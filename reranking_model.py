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


class ReRankSAEM(object):
    """
    """
    def __init__(self, opt):
        self.logger = None

        # Build Models
        self.grad_clip = opt.grad_clip

        self.txt_enc = text_net.BertBinaryMapping(opt)
        self.img_enc = image_net.TransformerBinaryMapping(opt)
        self.concept_enc = nn.Sequential(nn.Linear(in_features=opt.final_dims, out_features=1024),
                                         nn.ReLU(),
                                         nn.Linear(in_features=1024, out_features=opt.concept_num),
                                         nn.Sigmoid()) if opt.need_concept_label else None

        # self.img_enc = image_net.RnnMapping(opt.img_dim, opt.final_dims, 1)
        # self.img_enc = image_net.CnnMapping(opt.img_dim, opt.final_dims)

        if torch.cuda.is_available():
            self.txt_enc.cuda()
            self.img_enc.cuda()
            self.concept_enc.cuda() if self.concept_enc is not None else None
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = loss.ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        self.criterion2 = loss.AngularLoss()
        self.re_phrase_criterion = loss.RePhraseLoss()
        self.concept_loss = nn.BCELoss()

        # self.adv_criterion = loss.AdversaryLoss(margin=opt.adv_margin)
        self.adv_criterion=loss.AdversaryLossWithImg(margin=opt.adv_margin)


        # self.criterion = loss.L2Loss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        # self.criterion2 = loss.AngularLoss()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        if self.concept_enc is not None :
            params+=list(self.concept_enc.parameters())
        params = filter(lambda p: p.requires_grad, params)
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        if isinstance(self.img_enc, nn.DataParallel):
            state_dict = [self.img_enc.modules.state_dict(), self.txt_enc.modules.state_dict()]
            if self.concept_enc is not None:
                state_dict.append(self.concept_enc.modules.state_dict())
        else:
            state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
            if self.concept_enc is not None:
                state_dict.append(self.concept_enc.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        if len(state_dict)>=3 and self.concept_enc is not None:
            self.concept_enc.load_state_dict(state_dict[2])

    def use_data_parallel(self):
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.img_enc = nn.DataParallel(self.img_enc)
        if self.concept_enc is not None:
            self.concept_enc = nn.DataParallel(self.concept_enc)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        if self.concept_enc is not None :
            self.concept_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        if self.concept_enc is not None :
            self.concept_enc.eval()

    def unpack_batch_data(self, *batch_data):

        if torch.cuda.is_available():
            tmp = []
            for x in batch_data:
                if isinstance(x, torch.Tensor):
                    tmp.append(x.cuda())
                else:
                    tmp.append(x)
        else:
            tmp = batch_data

        raw_image, roi_feature, concept_label, \
        raw_input_ids, raw_input_mask, raw_input_type_ids, \
        re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
        adv_ids, adv_mask, adv_type_ids, \
        img_id, sample_index = tmp

        return raw_image, roi_feature, concept_label, \
               raw_input_ids, raw_input_mask, raw_input_type_ids, \
               re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
               adv_ids, adv_mask, adv_type_ids, \
               img_id, sample_index

    def forward_emb(self, epoch, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        # images, input_ids, attention_mask, token_type_ids, lengths, ids = self.bert_data(*batch_data)
        raw_image, roi_feature, concept_label, \
        raw_input_ids, raw_input_mask, raw_input_type_ids, \
        re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
        adv_ids, adv_mask, adv_type_ids, \
        img_id, sample_index = self.unpack_batch_data(*batch_data)

        batch_size=len(img_id)
        raw_cap_code = None
        re_phrase_code = torch.zeros((batch_size,1))
        adv_code = torch.zeros((batch_size,1))
        img_code = None
        cap_lens = None

        # forward image
        img_code = self.img_enc(roi_feature)

        raw_cap_code = self.txt_enc(raw_input_ids, raw_input_mask, raw_input_type_ids, None)
        if epoch > self.opt.adversary_step:
            if self.opt.need_rephrase_data:
                B, R, L = re_phrase_ids.shape
                re_phrase_ids = re_phrase_ids.view(B * R, L)
                re_phrase_mask = re_phrase_mask.view(B * R, L)
                re_phrase_type_ids = re_phrase_type_ids.view(B * R, L)
                re_phrase_code = self.txt_enc(re_phrase_ids, re_phrase_mask, re_phrase_type_ids, None).view(B, R, -1)
            if self.opt.need_adversary_data:
                B, A, L = adv_ids.shape
                adv_ids = adv_ids.view(B * A, L)
                adv_mask = adv_mask.view(B * A, L)
                adv_type_ids = adv_type_ids.view(B * A, L)
                adv_code = self.txt_enc(adv_ids, adv_mask, adv_type_ids, None).view(B, A, -1)

        concept_pred=None
        if self.opt.need_concept_label:
            concept_pred = self.concept_enc(raw_cap_code)
        concept_data = [concept_pred, concept_label]

        return sample_index, img_code, raw_cap_code, re_phrase_code, adv_code, concept_data, cap_lens

    def forward_loss(self, epoch, img_emb, cap_emb, re_phrase_emb, adv_emb, cap_len, sample_index, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        alpha = 0
        beta = 0.5
        gamma = 0.5
        theta = 0.5

        loss1 = torch.tensor(0.0)
        loss2 = torch.tensor(0.0)
        re_phrase_loss = torch.tensor(0.0)
        adversary_loss = torch.tensor(0.0)
        concept_loss = torch.tensor(0.0)

        loss1 = self.criterion(img_emb, cap_emb, cap_len, sample_index)

        if epoch > self.opt.adversary_step and self.opt.need_rephrase_data:
            re_phrase_loss = self.re_phrase_criterion(img_emb, re_phrase_emb)# todo  测试将imgembedding 用于训练rephrase句子
        if epoch> self.opt.adversary_step and self.opt.need_adversary_data:
            adversary_loss = self.adv_criterion(cap_emb, adv_emb,img_emb)

        if self.opt.need_concept_label and 'concept_data' in kwargs:
            pred_concept, concept_label = kwargs['concept_data']
            concept_loss = self.concept_loss.forward(pred_concept, concept_label)

        # alpha = 1
        if epoch <= 20 and self.criterion2 is not None:
            alpha = 0.5 * (0.1 ** (epoch // 5))
            # loss2 = self.criterion2(img_emb , cap_emb , cap_len, ids)
            loss2 = self.criterion2(img_emb / img_emb.norm(dim=1)[:, None], cap_emb / cap_emb.norm(dim=1)[:, None],
                                    cap_len, sample_index)

        self.logger.update('Loss1', loss1.item(), img_emb.size(0))
        self.logger.update('Loss2', loss2.item(), img_emb.size(0))
        self.logger.update('rep_Loss', re_phrase_loss.item(), img_emb.size(0))
        self.logger.update('adv_Loss', adversary_loss.item(), img_emb.size(0))
        self.logger.update('concept_Loss', concept_loss.item(), img_emb.size(0))

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

        total_loss = loss1 + alpha * loss2 + reg_loss + \
                     beta * re_phrase_loss + gamma * adversary_loss + \
                     theta * concept_loss

        return total_loss
        # return loss2 + reg_loss

    def train_emb(self, epoch, batch_data, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        sample_index, img_code, raw_cap_code, re_phrase_code, adv_code, concept_data, cap_lens = self.forward_emb(epoch,
                                                                                                             batch_data)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_code, raw_cap_code, re_phrase_code, adv_code, cap_lens, sample_index,
                                 concept_data=concept_data)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
