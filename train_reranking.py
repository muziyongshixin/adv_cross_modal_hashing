# -*- coding: utf-8 -*-
import argparse
import logging
import logging.config
import os
import shutil
import socket
import time
from pprint import pformat

import torch
from torch.utils.tensorboard import SummaryWriter

import enhanced_data as data
from binary_SAEM_model import BinarySAEM
from evaluation import AverageMeter, LogCollector
from evaluation import validate, validate_binary
from SAEM_model import SAEM
from screen_model import BinaryBaseModel
from reranking_model import ReRankSAEM
from utils import init_logging

logger = logging.getLogger(__name__)
global tb_logger

def main():
    global tb_logger
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/scan_data/',
                        help='path to datasets')

    ####################################### coco data #############################################
    parser.add_argument('--image_root_dir', default='data/mscoco/',
                        help='path to coco_root_dir')
    parser.add_argument('--concept_file_path', default='data/coco_concept/coco_imgid_to_rela+obj+categ_vec.pickle', #'/S4/MI/liyz/data/coco_concept/new_imgid_2_concept_idxs.pkl',
                        help='path concept label file')
    parser.add_argument('--concept_num', default=642 + 1000 + 91, type=int, help='caption的 concept标签类别数')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f30k}_precomp')

    ####################################### flickr data #############################################
    # parser.add_argument('--image_root_dir', default='data/f30k/flickr30/images',
    #                     help='path to coco_root_dir')
    # parser.add_argument('--concept_file_path', default='data/f30k_concept/f30k_imgid_to_obj_rela_concept_vec.pickle',
    #                     help='path concept label file')
    # parser.add_argument('--concept_num', default=2000, type=int,
    #                     help='caption的 concept标签类别数')
    # parser.add_argument('--data_name', default='f30k_precomp',
    #                     help='{coco,f30k}_precomp')


    parser.add_argument('--need_raw_image', default=0,type=int,
                        help='是否使用原始图片作为输入，1表示需要，0表示不需要')
    parser.add_argument('--need_roi_feature', default=1,type=int,
                        help='是否需要使用faster rcnn提取的roi feature作为输入')
    parser.add_argument('--need_adversary_data', default=0,type=int,
                        help='是否使用adversary的文本数据进行训练')
    parser.add_argument('--need_rephrase_data', default=0,type=int,
                        help='是否使用rephrase的文本数据进行训练')
    parser.add_argument('--need_concept_label', default=1,type=int,
                        help='是否使用文本的concept label进行训练')

    parser.add_argument('--part_train_data',default='', type=str,help='和hash方法比较的时候只使用1w训练集')

    parser.add_argument('--adversary_step', default=-1, type=int,
                        help='After how many epochs to start adversary training')
    parser.add_argument('--adversary_num', default=10, type=int,
                        help='use how many adversary sentences in training')
    parser.add_argument('--adversary_type', default='noun', type=str,
                        help='the adversary sample type {noun,num,rela,mixed}')
    parser.add_argument('--adv_margin', default=0.5, type=float,
                        help='the adversary loss margin')

    parser.add_argument('--image_size', default=256, type=int,
                        help='the raw image size to feed into the image network')
    parser.add_argument('--model_name', default='rerank_model',
                        help='{rerank,screen,binary_saem}_model')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--data_eager', default=False,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', default=True, action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--final_dims', default=256, type=int,
                        help='dimension of final codes.')
    parser.add_argument('--max_words', default=32, type=int,
                        help='maximum number of words in a sentence.')
    parser.add_argument("--bert_path",
                        default='data/bert_ckpt/uncased_L-12_H-768_A-12/',
                        type=str,
                        help="The BERT model path.")
    parser.add_argument("--txt_stru", default='cnn',
                        help="Whether to use pooling or cnn or rnn")
    parser.add_argument("--trans_cfg", default='t_cfg.json',
                        help="config file for image transformer")
    parser.add_argument("--remark", default='',
                        help="description about the experiments")
    parser.add_argument("--binary", default='True',
                        help="generate binary hash code?")


    opt = parser.parse_args()



    # create experiments dir
    exp_root_dir = 'runs/'
    cur_time = time.localtime(int(time.time()))
    time_info = time.strftime('%Y_%m_%d,%H_%M_%S', cur_time)
    host_name = socket.gethostname()
    exp_name = '{data_name}/{time}_{host}_{remark}'.format(data_name=opt.data_name, time=time_info, host=host_name,
                                                           remark=opt.remark)
    exp_dir = os.path.join(exp_root_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    opt.exp_dir = exp_dir

    init_logging(opt.exp_dir)

    # pprint(vars(opt))
    logger.info(pformat(vars(opt)))

    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO,filename=os.path.join(exp_dir,'info.log'),filemode='a')
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # opt.logger_name = opt.logger_name + TIMESTAMP

    # tb_logger.configure(opt.logger_name, flush_secs=5)
    tb_dir = os.path.join(exp_dir, 'tensor_board')
    os.makedirs(tb_dir)
    tb_logger = SummaryWriter(log_dir=tb_dir)

    opt.vocab_file = os.path.join(opt.bert_path, 'vocab.txt')
    opt.bert_config_file = os.path.join(opt.bert_path, 'bert_config.json')
    opt.init_checkpoint = os.path.join(opt.bert_path, 'pytorch_model.bin')
    opt.do_lower_case = True

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, opt.batch_size, opt.workers, opt)

    # Construct the model
    if opt.model_name == 'rerank_model':
        model = ReRankSAEM(opt)
        logger.info('Model name is ReRankSAEM model')
    elif opt.model_name == 'screen_model':
        model = BinaryBaseModel(opt)
        logger.info('Model name is Binary Base model')
    elif opt.model_name == 'binary_saem_model':
        logger.warning('Training SAEM binary models !!!!!!!!!!.....')
        model = BinarySAEM(opt)
    else:
        model = SAEM(opt)

    start_epoch = 0
    best_rsum = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            logger.error("=> no checkpoint found at '{}'".format(opt.resume))

    if torch.cuda.device_count() > 1:
        model.use_data_parallel()
        logger.info('=> using data parallel...')

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate_binary(opt, val_loader, model, tb_logger=tb_logger)
        #     rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=os.path.join(exp_dir, 'checkpoints'))


def train(opt, train_loader, model, epoch, val_loader):
    logger.info('=================== start training epoch {} ================'.format(epoch))
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    # make sure train logger is used
    model.logger = train_logger

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # Update the model
        model.train_emb(epoch, train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logger.info('[{0}][{1}/{2}] {e_log}'.format(epoch, i, len(train_loader), e_log=str(model.logger)))


        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate_binary(opt, val_loader, model, tb_logger=tb_logger)
            # else:
            #     rsum = validate(opt, val_loader, model)
            # validate(opt, val_loader, model)
            # validate_binary(opt, val_loader, model)
    # Record logs in tensorboard
    tb_logger.add_scalar('train/epoch', epoch, global_step=model.Eiters)
    tb_logger.add_scalar('train/batch_time', batch_time.avg, global_step=epoch)
    tb_logger.add_scalar('train/data_time', data_time.avg, global_step=epoch)
    model.logger.tb_log(tb_logger, prefix='train/', step=epoch)


#
# def validate(opt, val_loader, model):
#     # compute the encoding for all the validation images and captions
#     img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt, opt.log_step, logger.info)
#
#     img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
#
#     start = time.time()
#     sims = 1 - cdist(img_embs, cap_embs, metric='cosine')
#     end = time.time()
#     logger.info("calculate similarity time:{}".format(end - start))
#
#     # caption retrieval
#     (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
#     logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))
#     # image retrieval
#     (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
#     logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))
#     # sum of recalls to be used for early stopping
#     currscore = r1 + r5 + r10 + r1i + r5i + r10i
#
#     # record metrics in tensorboard
#     tb_logger.add_scalar('val/r1_i2t', r1, global_step=model.Eiters)
#     tb_logger.add_scalar('val/r5_i2t', r5, global_step=model.Eiters)
#     tb_logger.add_scalar('val/r10_i2t', r10, global_step=model.Eiters)
#     tb_logger.add_scalar('val/medr_i2t', medr, global_step=model.Eiters)
#     tb_logger.add_scalar('val/meanr_i2t', meanr, global_step=model.Eiters)
#     tb_logger.add_scalar('val/r1i_t2i', r1i, global_step=model.Eiters)
#     tb_logger.add_scalar('val/r5i_t2i', r5i, global_step=model.Eiters)
#     tb_logger.add_scalar('val/r10i_t2i', r10i, global_step=model.Eiters)
#     tb_logger.add_scalar('val/medri_t2i', medri, global_step=model.Eiters)
#     tb_logger.add_scalar('val/meanr_t2i', meanr, global_step=model.Eiters)
#     tb_logger.add_scalar('val/rsum_t2i', currscore, global_step=model.Eiters)
#
#     return currscore


# def validate_binary(opt, val_loader, model):
#     # compute the encoding for all the validation images and captions
#     img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt, opt.log_step, logger.info)
#
#     img_embs = img_embs[::5, ...]
#     img_embs = torch.sign(torch.from_numpy(img_embs)).long()
#     cap_embs = torch.sign(torch.from_numpy(cap_embs)).long()
#
#     start = time.time()
#     sims = get_hamming_dist(img_embs, cap_embs)  # hamming distance matrix  1000*5000
#     end = time.time()
#     logger.info("calculate similarity time:{}".format(end - start))
#
#     # caption retrieval
#     topk_r_i2tb = i2t_binary(sims, topk=(1, 5, 10, 50, 100, 200))
#     logger.info("Image to text: {}".format(str(topk_r_i2tb)))
#     # image retrieval
#     topk_r_t2ib = t2i_binary(sims, topk=(1, 5, 10, 50, 100, 200))
#     logger.info("Text to image: {}".format(str(topk_r_t2ib)))
#     # sum of recalls to be used for early stopping
#     currscore = [ri2t for k, ri2t in topk_r_i2tb.items()] + [rt2i for k, rt2i in topk_r_t2ib.items()]
#     currscore = sum(currscore)
#
#     # record metrics in tensorboard
#     for k, recall in topk_r_i2tb.items():
#         tb_logger.add_scalar('val/i2t_{}'.format(k), recall, global_step=model.Eiters)
#     for k, recall in topk_r_t2ib.items():
#         tb_logger.add_scalar('val/t2i_{}'.format(k), recall, global_step=model.Eiters)
#
#     return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 3
    error = None
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    save_path = os.path.join(prefix, filename)
    best_path = os.path.join(prefix, 'model_best.pth.tar')
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, save_path)
            logger.info('save checkpoint to {}'.format(save_path))
            if is_best:
                shutil.copyfile(save_path, best_path)
                logger.info('copy best checkpoint to {}'.format(best_path))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logging.error('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""

    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
