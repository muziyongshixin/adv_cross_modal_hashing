import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import text_net
import loss
import image_net

from IPython import embed


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class SAEM(object):
    """
    """

    def __init__(self, opt):
        self.logger=None

        # Build Models
        self.grad_clip = opt.grad_clip
        self.txt_enc = text_net.BertMapping(opt)
        self.img_enc = image_net.TransformerMapping(opt)
        # self.img_enc = image_net.RnnMapping(opt.img_dim, opt.final_dims, 1)
        # self.img_enc = image_net.CnnMapping(opt.img_dim, opt.final_dims)

        if torch.cuda.is_available():
            self.txt_enc.cuda()
            self.img_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = loss.ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        self.criterion2 = loss.AngularLoss()
        self.re_phrase_criterion = loss.RePhraseLoss()
        self.adv_criterion = loss.AdversaryLoss(margin=opt.adv_margin)

        # self.criterion = loss.L2Loss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        # self.criterion2 = loss.AngularLoss()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params = filter(lambda p: p.requires_grad, params)
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        if isinstance(self.img_enc, nn.DataParallel):
            state_dict = [self.img_enc.modules.state_dict(), self.txt_enc.modules.state_dict()]
        else:
            state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def use_data_parallel(self):
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.img_enc = nn.DataParallel(self.img_enc)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def bert_data(self, images, input_ids, attention_mask, token_type_ids, lengths, ids):
        return images, input_ids, attention_mask, token_type_ids, lengths, ids

    def enhanced_bert_data(self, image, \
                           raw_input_ids, raw_input_mask, raw_input_type_ids, \
                           re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
                           adv_ids, adv_mask, adv_type_ids, index, img_id):
        return index, img_id, image, \
               raw_input_ids, raw_input_mask, raw_input_type_ids, \
               re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
               adv_ids, adv_mask, adv_type_ids

    def forward_emb(self, epoch, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        # images, input_ids, attention_mask, token_type_ids, lengths, ids = self.bert_data(*batch_data)
        index, img_id, images, \
        raw_input_ids, raw_input_mask, raw_input_type_ids, \
        re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
        adv_ids, adv_mask, adv_type_ids = self.enhanced_bert_data(*batch_data)

        B, R, L = re_phrase_ids.shape
        _, A, _ = adv_ids.shape

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()

            raw_input_ids = raw_input_ids.cuda()
            raw_input_mask = raw_input_mask.cuda()
            raw_input_type_ids = raw_input_type_ids.cuda()

            re_phrase_ids = re_phrase_ids.view(B * R, L).cuda()
            re_phrase_mask = re_phrase_mask.view(B * R, L).cuda()
            re_phrase_type_ids = re_phrase_type_ids.view(B * R, L).cuda()

            adv_ids = adv_ids.view(B * A, L).cuda()
            adv_mask = adv_mask.view(B * A, L).cuda()
            adv_type_ids = adv_type_ids.view(B * A, L).cuda()

            # input_ids = input_ids.cuda()
            # attention_mask = attention_mask.cuda()
            # token_type_ids = token_type_ids.cuda()
        # forward text
        # print('model input',input_ids.shape)
        raw_cap_code = self.txt_enc(raw_input_ids, raw_input_mask, raw_input_type_ids, None)
        if epoch  > 5:
            re_phrase_code = self.txt_enc(re_phrase_ids, re_phrase_mask, re_phrase_type_ids, None).view(B, R, -1)
            adv_code = self.txt_enc(adv_ids, adv_mask, adv_type_ids, None).view(B, A, -1)
        else:
            re_phrase_code=None
            adv_code=None

        cap_lens = None

        # forward image
        img_code = self.img_enc(images)

        return img_id, img_code, raw_cap_code, re_phrase_code, adv_code, cap_lens

    def forward_loss(self, epoch, img_emb, cap_emb, re_phrase_emb, adv_emb, cap_len, ids, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # alpha = 0
        loss1 = self.criterion(img_emb, cap_emb, cap_len, ids)


        if epoch>5:
            re_phrase_loss = self.re_phrase_criterion(cap_emb, re_phrase_emb)
            adversary_loss = self.adv_criterion(cap_emb, adv_emb)
        else:
            re_phrase_loss=torch.tensor(0.0)
            adversary_loss=torch.tensor(0.0)

        # alpha = 1
        if epoch > 20 or self.criterion2 is None:
            alpha = 0
            loss2 = torch.tensor(0.0)
        else:
            alpha = 0.5 * (0.1 ** (epoch // 5))
            # loss2 = self.criterion2(img_emb , cap_emb , cap_len, ids)
            loss2 = self.criterion2(img_emb / img_emb.norm(dim=1)[:, None], cap_emb / cap_emb.norm(dim=1)[:, None],
                                    cap_len, ids)

        self.logger.update('Loss1', loss1.item(), img_emb.size(0))
        self.logger.update('Loss2', loss2.item(), img_emb.size(0))
        self.logger.update('rep_Loss', re_phrase_loss.item(), img_emb.size(0))
        self.logger.update('adv_Loss', adversary_loss.item(), img_emb.size(0))

        l2_reg = torch.tensor(0., dtype=torch.float)
        if torch.cuda.is_available():
            l2_reg = l2_reg.cuda()
        no_decay = ['bias', 'gamma', 'beta']
        for n, p in self.img_enc.named_parameters():
            en = n.split('.')[-1]
            if en not in no_decay:
                l2_reg += torch.norm(p)
        # for n, p in self.txt_enc.mapping.named_parameters():
        #     en = n.split('.')[-1]
        #     if en not in no_decay:
        #         l2_reg += torch.norm(p)
        # for n, p in self.txt_enc.layer.named_parameters():
        #     en = n.split('.')[-1]
        #     if en not in no_decay:
        #         l2_reg += torch.norm(p)
        reg_loss = 0.01 * l2_reg

        beta = 0.5
        gamma = 0.5

        return loss1 + alpha * loss2 + beta * re_phrase_loss + gamma * adversary_loss + reg_loss
        # return loss2 + reg_loss

    def train_emb(self, epoch, batch_data, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_ids, img_code, raw_cap_code, re_phrase_code, adv_code, cap_lens = self.forward_emb(epoch, batch_data)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_code, raw_cap_code, re_phrase_code, adv_code, cap_lens, img_ids)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
