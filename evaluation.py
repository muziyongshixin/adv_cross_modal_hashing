from __future__ import print_function
import os

import sys
from data import get_test_loader
import time
import numpy as np
import torch
from SAEM_model import SAEM
from collections import OrderedDict
import time
import logging
from utils import get_hamming_dist, save_similarity_matrix
from scipy.spatial.distance import cdist
import pickle

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def clear_all(self):
        del self.meters
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.avg, global_step=step)


def encode_data(model, data_loader, log_step=10, tb_logger=None):
    """Encode all images and captions loadable by `data_loader`, 数据中不包含 adversary data和 rephrase data
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None

    # max_n_word = 0
    # for i, (images, input_ids, attention_mask, token_type_ids, lengths, ids) in enumerate(data_loader):
    #     max_n_word = max(max_n_word, max(lengths))
    eval_start_time=time.time()
    for i, batch_data in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        ids, img_emb, cap_emb, re_phrase_emb, adv_emb, concept_data, cap_len = model.forward_emb(20, batch_data,
                                                                                                 volatile=True)
        # img_emb, cap_emb, cap_len, ids = model.forward_emb(20, batch_data, volatile=True)
        # print(img_emb)
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        ids = batch_data[-1]
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        # for j, nid in enumerate(ids):
        #     cap_lens[nid] = cap_len[j]

        # measure accuracy and record loss
        model.forward_loss(10, img_emb, cap_emb, re_phrase_emb, adv_emb, cap_len, ids, concept_data=concept_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logger.info('Test: [{0}/{1}]\t {e_log}\t Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                        .format(i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)))
        # del images, input_ids, attention_mask, token_type_ids
    if tb_logger is not None:
        model.logger.tb_log(tb_logger, prefix='val/', step=model.Eiters)
    logger.info('evaluation use time is {}'.format(time.time()-eval_start_time))
    return img_embs, cap_embs, cap_lens


def encode_data_with_adversary(model, data_loader, log_step=10, tb_logger=None):
    """Encode all images and captions loadable by `data_loader`，数据中包含adversary data
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = []
    cap_embs = []
    cap_lens = []

    for i, batch_data in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        ids, img_emb, cap_emb, re_phrase_emb, adv_emb, concept_data, cap_len = model.forward_emb(200, batch_data,
                                                                                                 volatile=True)
        #    B, dim   B,dim     B,R,dim       B,A,dim
        batch_size, code_dim = img_emb.shape

        # measure accuracy and record loss
        model.forward_loss(10, img_emb, cap_emb, re_phrase_emb, adv_emb, cap_len, ids, concept_data=concept_data)

        for j in range(batch_size):
            img_embs.append(img_emb[j].cpu().detach())
            tmp_cap = [cap_emb[j].cpu().detach(), re_phrase_emb[j].cpu().detach(), adv_emb[j].cpu().detach()]
            cap_embs.append(tmp_cap)

        del ids, img_emb, cap_emb, re_phrase_emb, adv_emb, cap_len
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logger.info('Test: [{0}/{1}]\t{e_log}\tTime {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                        .format(i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)))
        # del images, input_ids, attention_mask, token_type_ids
    if tb_logger is not None:
        model.logger.tb_log(tb_logger, prefix='val/', step=model.Eiters)
    return img_embs, cap_embs, cap_lens


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    每张图片查出五个gt句子，将gt句子中rank最高的作为结果
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    五个句子分别查，得到的结果取平均。
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_binary(matching_score_matrix, topk=(1, 5, 10, 50, 100, 200)):
    ''' 计算text查image 的结果。 每一行是一个text 和所有5000个image haming 距离
    :param matching_score_matrix:  5000*1000  tensor matrix
    :param topk: tuple to calculate different topk accuracy score
    :return: list
    '''
    #     assert matching_score_matrix.shape==(5000,1000)

    msm = matching_score_matrix.clone()
    msm = msm.transpose_(0, 1)  # size 5000,1000

    n_caps, n_imgs = msm.shape
    assert n_caps == n_imgs * 5

    result = {}
    for k in topk:
        max_score, max_score_idx = msm.topk(k=k, dim=1, largest=False)
        correct_count = 0
        for i in range(n_caps):
            cur_i_topk = max_score_idx[i]
            j = i // 5
            if j in cur_i_topk:
                correct_count += 1

        acc = correct_count / n_caps
        result[k] = acc
    return result


def i2t_binary(matching_score_matrix, topk=(1, 5, 10, 50, 100, 200)):
    '''计算image查text的结果。 每一行是一个image 和所有5000个captions  haming 距离
    :param matching_score_matrix:  n*n  tensor matrix
    :param topk: tuple to calculate different topk accuracy score
    :return: list
    '''
    msm = matching_score_matrix.clone()  # size 1000,5000
    n_imgs, n_caps = msm.shape
    assert n_imgs * 5 == n_caps
    result = {}
    for k in topk:
        max_score, max_score_idx = msm.topk(k=k, dim=1, largest=False)
        correct_count = 0
        for i in range(n_imgs):
            cur_i_topk = max_score_idx[i]
            for j in range(i * 5, i * 5 + 5):  # 五句话只要有一句在里面就可以认为命中
                if j in cur_i_topk:
                    correct_count += 1
                    break

        acc = correct_count / n_imgs
        result[k] = acc
    return result


def img2text_adversary(img_embs, cap_embs, **kwargs):
    '''
    :param img_embs: List, [one image code]* 5000
    :param cap_embs: List, [[ori_cap_code,reph_cap_code,adv_cap_code]]*5000
    :return:
    '''
    topk = (1, 5, 10, 50, 100, 200, 500)
    img_codes = torch.stack(img_embs, dim=0)[::5, ...]  # 1000,256
    cap_codes = []
    adv_num=-1
    for sample in cap_embs:
        cur = [sample[0].squeeze()]
        adv_num=len(sample[2])
        for adv in sample[2]:
            cur.append(adv)
        cap_codes += cur
    cap_codes = torch.stack(cap_codes, dim=0)  # 11*5*1000, 256

    img_codes = torch.sign(img_codes).long()
    cap_codes = torch.sign(cap_codes).long()

    sims = get_hamming_dist(img_codes, cap_codes)  # hamming distance matrix  1000*55000

    n_imgs, n_caps = sims.shape
    assert n_imgs * 5 * (adv_num+1) == n_caps

    base_matrix = sims[:, ::(adv_num + 1)]  # 1000，5000
    print('base matrix shape is ', base_matrix.shape)
    result = {}
    for k in topk:
        # max_score, max_score_idx = sims.topk(k=k, dim=1, largest=False)
        correct_count = 0
        for i in range(n_imgs):
            cur_base_row = base_matrix[i]
            t_l = i * 5 * (adv_num + 1)
            t_r = (i + 1) * 5 * (adv_num + 1)

            tmp = []
            for j in range(t_l + 1, t_r, (adv_num + 1)):
                tmp.append(sims[i, j:j + adv_num])
            cur_adv_row = torch.cat(tmp, dim=-1)
            cur_row = torch.cat([cur_base_row, cur_adv_row], dim=-1)  # 5000+55
            scores, cur_i_topk = cur_row.topk(k=k, dim=-1, largest=False)
            for j in range(i * 5, (i + 1) * 5):  # 五句话只要有一句在里面就可以认为命中, 每个image有55个句子，其中5个是gt
                if j in cur_i_topk:
                    correct_count += 1
                    break
        acc = correct_count / n_imgs
        result[k] = acc
    logger.info("Adversary retrieval (Add 10 adversary sentences for each original caption)\n" +
                "Image to text: {}".format(str(result)))

    if 'save_sm_matrix_dir' in kwargs:
        save_path = os.path.join(kwargs['save_sm_matrix_dir'], 'adv_retrieval_i2t_sm_matrix.npz')
        save_similarity_matrix(sims.cpu().detach().numpy(), save_path)

    return result


def text2image_rephrase(img_embs, cap_embs, **kwargs):
    ''' 使用re-phrase的句子作为query来查图片
    :param img_embs: List, [one image code]* 5000
    :param cap_embs: List, [[ori_cap_code,reph_cap_code,adv_cap_code]]*5000
    :return:
    '''
    topk = (1, 5, 10, 50, 100, 200, 500)
    img_codes = torch.stack(img_embs, dim=0)[::5, ...]  # 1000,256
    cap_codes = []
    for sample in cap_embs:
        cur = []
        for reph in sample[1]:
            cur.append(reph)
        cap_codes += cur
    cap_codes = torch.stack(cap_codes, dim=0)  # 2*5*1000, 256

    img_codes = torch.sign(img_codes).long()
    cap_codes = torch.sign(cap_codes).long()

    sims = get_hamming_dist(cap_codes, img_codes)  # hamming distance matrix  10000,1000

    n_caps, n_imgs = sims.shape
    assert n_caps == n_imgs * 5 * 2

    result = {}
    for k in topk:
        max_score, max_score_idx = sims.topk(k=k, dim=1, largest=False)
        correct_count = 0
        for i in range(n_caps):
            cur_i_topk = max_score_idx[i]
            j = i // 10
            if j in cur_i_topk:
                correct_count += 1

        acc = correct_count / n_caps
        result[k] = acc

    logger.info("Adversary retrieval (Using 2 re-phrase sentences as query)\n" +
                "Text to image: {}".format(str(result)))

    if 'save_sm_matrix_dir' in kwargs:
        save_path = os.path.join(kwargs['save_sm_matrix_dir'], 'adv_retrieval_t2i_sm_matrix.npz')
        save_similarity_matrix(sims.cpu().detach().numpy(), save_path)
    return result


def validate(opt, val_loader, model, **kwargs):
    tb_logger = kwargs['tb_logger'] if 'tb_logger' in kwargs else None
    # compute the encoding for all the validation images and captions，使用的是连续向量
    start = time.time()
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, tb_logger)
    end = time.time()
    print("calculate backbone time:", end - start)

    print(img_embs.shape, cap_embs.shape)
    # save_vec_path = os.path.join(opt.exp_dir, 'saved_vector.pkl')
    # save_vector_to_file(data={'img_vec': img_embs, 'cap_vec': cap_embs}, file_name=save_vec_path)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = 1 - cdist(img_embs, cap_embs, metric='cosine')
    end = time.time()
    logger.info("calculate similarity time:{}".format(end - start))

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def validate_binary(opt, val_loader, model, **kwargs):
    tb_logger = kwargs['tb_logger'] if 'tb_logger' in kwargs else None
    adv_i2t, adv_t2i = None, None
    ori_i2tb, ori_t2ib = None, None

    save_code_flag = True if 'save_hash_code' in kwargs and kwargs['save_hash_code'] is True else False

    # compute the encoding for all the validation images and captions
    if opt.need_adversary_data or opt.need_rephrase_data:
        img_embs, cap_embs, cap_lens = encode_data_with_adversary(model, val_loader, opt.log_step, tb_logger)

        ori_i2tb, ori_t2ib = original_retrieval_with_adversary_data(img_embs, cap_embs, **kwargs)

        if opt.need_adversary_data:
            adv_i2t = img2text_adversary(img_embs, cap_embs, **kwargs)
        if opt.need_rephrase_data:
            adv_t2i = text2image_rephrase(img_embs, cap_embs, **kwargs)

    else:
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, tb_logger)

        ori_i2tb, ori_t2ib = original_retrieval(img_embs, cap_embs, **kwargs)

    if tb_logger is not None:
        for k, val in ori_i2tb.items():
            tb_logger.add_scalar('val/ori_i2tb_top{}'.format(k), val, global_step=model.Eiters)
        for k, val in ori_t2ib.items():
            tb_logger.add_scalar('val/ori_t2ib_top{}'.format(k), val, global_step=model.Eiters)
        if adv_i2t is not None:
            for k, val in adv_i2t.items():
                tb_logger.add_scalar('val/adv_i2t_top{}'.format(k), val, global_step=model.Eiters)
        if adv_t2i is not None:
            for k, val in adv_t2i.items():
                tb_logger.add_scalar('val/adv_t2i_top{}'.format(k), val, global_step=model.Eiters)

    if save_code_flag:
        save_path = os.path.join(opt.exp_dir, 'saved_hash_code.pickle')
        save_hashcodes(img_embs, cap_embs, save_path)

    currscore = [ri2t for k, ri2t in ori_i2tb.items() if k < 50] + [rt2i for k, rt2i in ori_t2ib.items() if k < 50]

    # currscore=[s for k ,s in adv_i2t.items() if k<100] #仅考虑adversary 效果最好的时候保存最优checkpoint

    r_sum = sum(currscore)
    return r_sum


def original_retrieval(img_embs, cap_embs, **kwargs):
    img_codes = torch.from_numpy(img_embs[::5, ...])  # 1000,256
    cap_codes = torch.from_numpy(cap_embs)  # 5*1000, 256

    img_codes = torch.sign(img_codes).long()
    cap_codes = torch.sign(cap_codes).long()

    sims = get_hamming_dist(img_codes, cap_codes)  # hamming distance matrix  1000*5000

    # caption retrieval
    topk_r_i2tb = i2t_binary(sims)
    logger.info("Original retrieval, Image to text: {}".format(str(topk_r_i2tb)))
    # image retrieval
    topk_r_t2ib = t2i_binary(sims)
    logger.info("Original retrieval, Text to image: {}".format(str(topk_r_t2ib)))

    if 'save_sm_matrix_dir' in kwargs:
        save_path = os.path.join(kwargs['save_sm_matrix_dir'], 'original_retrieval_i2t_sm_matrix.npz')
        save_similarity_matrix(sims.cpu().detach().numpy(), save_path)

    return topk_r_i2tb, topk_r_t2ib


def original_retrieval_with_adversary_data(img_embs, cap_embs, **kwargs):
    '''
    :param img_embs:
    :param cap_embs: 得到的数据中包含adversary的data
    :param topk:
    :return:
    '''
    img_codes = torch.stack(img_embs, dim=0)[::5, ...]  # 1000,256
    cap_codes = []
    for sample in cap_embs:
        cap_codes.append(sample[0].squeeze())
    cap_codes = torch.stack(cap_codes, dim=0)  # 5*1000, 256

    img_codes = torch.sign(img_codes).long()
    cap_codes = torch.sign(cap_codes).long()

    sims = get_hamming_dist(img_codes, cap_codes)  # hamming distance matrix  1000*5000

    # caption retrieval
    topk_r_i2tb = i2t_binary(sims)
    logger.info("Original retrieval, Image to text: {}".format(str(topk_r_i2tb)))
    # image retrieval
    topk_r_t2ib = t2i_binary(sims)
    logger.info("Original retrieval, Text to image: {}".format(str(topk_r_t2ib)))

    if 'save_sm_matrix_dir' in kwargs:
        save_path = os.path.join(kwargs['save_sm_matrix_dir'], 'original_retrieval_i2t_sm_matrix.npz')
        save_similarity_matrix(sims.cpu().detach().numpy(), save_path)
    return topk_r_i2tb, topk_r_t2ib


def save_hashcodes(img_embeds, cap_embeds, save_path):
    if isinstance(img_embeds, np.ndarray):
        img_embeds = torch.from_numpy(img_embeds)
        cap_embeds = torch.from_numpy(cap_embeds)
    img_hash_code = torch.sign(img_embeds).cpu().numpy().astype(np.int8)
    cap_hash_code = torch.sign(cap_embeds).cpu().numpy().astype(np.int8)

    hash_code = {'img_code': img_hash_code, 'cap_code': cap_hash_code}
    pickle.dump(hash_code, open(save_path, 'wb'))
    logger.info('save hash code to file {} successfully.'.format(save_path))
