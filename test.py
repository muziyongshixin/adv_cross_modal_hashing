import argparse
import logging
import os
import pickle
import socket
import time
from datetime import datetime
from pprint import pformat

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

import enhanced_data as data
from binary_SAEM_model import BinarySAEM
from screen_model import BinaryBaseModel
from evaluation import validate_binary, validate
from SAEM_model import SAEM
from reranking_model import ReRankSAEM

logger = logging.getLogger(__name__)
from utils import init_logging, get_hamming_dist

global tb_logger


def main():
    global tb_logger
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/scan_data/',
                        help='path to datasets')

    ####################################### coco data #############################################
    # parser.add_argument('--image_root_dir', default='data/mscoco/',
    #                     help='path to coco_root_dir')
    # parser.add_argument('--concept_file_path', default='data/coco_concept/coco_imgid_to_rela+obj+categ_vec.pickle', #'/S4/MI/liyz/data/coco_concept/new_imgid_2_concept_idxs.pkl',
    #                     help='path concept label file')
    # parser.add_argument('--concept_num', default=642 + 1000 + 91, type=int, help='caption的 concept标签类别数')
    # parser.add_argument('--data_name', default='coco_precomp',
    #                     help='{coco,f30k}_precomp')
    ####################################### above coco data #############################################


    ####################################### flickr data #############################################
    parser.add_argument('--image_root_dir', default='data/f30k/flickr30/images',
                        help='path to coco_root_dir')
    parser.add_argument('--concept_file_path', default='data/f30k_concept/f30k_imgid_to_obj_rela_concept_vec.pickle',
                        help='path concept label fi le')
    parser.add_argument('--concept_num', default=2000, type=int,
                        help='caption的 concept标签类别数')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    ####################################### above flickr data #############################################

    parser.add_argument('--need_raw_image', default=0, type=int,
                        help='是否使用原始图片作为输入，1表示需要，0表示不需要')
    parser.add_argument('--need_roi_feature', default=1, type=int,
                        help='是否需要使用faster rcnn提取的roi feature作为输入')
    parser.add_argument('--need_adversary_data', default=0, type=int,
                        help='是否使用adversary的文本数据进行训练')
    parser.add_argument('--need_rephrase_data', default=0, type=int,
                        help='是否使用rephrase的文本数据进行训练')
    parser.add_argument('--need_concept_label', default=0, type=int,
                        help='是否使用文本的concept label进行训练')

    parser.add_argument('--part_train_data', default='', type=str, help='和hash方法比较的时候只使用1w训练集')

    parser.add_argument('--adversary_step', default=-1, type=int,
                        help='After how many epochs to start adversary training')
    parser.add_argument('--adversary_num', default=10, type=int,
                        help='After how many epochs to start adversary training')
    parser.add_argument('--adversary_type', default='noun', type=str,
                        help='the adversary sample type {noun,num,rela,mixed}')
    parser.add_argument('--adv_margin', default=0.5, type=float,
                        help='the adversary loss margin')

    parser.add_argument('--image_size', default=256, type=int,
                        help='the raw image size to feed into the image network')
    parser.add_argument('--model_name', default='rerank_model',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
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

    parser.add_argument('--test_split', default='test', help='test data split name [test/testall]')
    opt = parser.parse_args()
    logger.info(pformat(vars(opt)))

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    opt.logger_name = opt.logger_name + TIMESTAMP

    # create experiments dir
    exp_root_dir = 'runs/testing/'
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

    tb_dir = os.path.join(exp_dir, 'tensor_board')
    os.makedirs(tb_dir)
    tb_logger = SummaryWriter(log_dir=tb_dir)

    opt.vocab_file = opt.bert_path + 'vocab.txt'
    opt.bert_config_file = opt.bert_path + 'bert_config.json'
    opt.init_checkpoint = opt.bert_path + 'pytorch_model.bin'
    opt.do_lower_case = True

    # Load data loaders
    test_loader = data.get_test_loader(opt.test_split, opt.data_name, opt.batch_size, opt.workers, opt)

    # Construct the modea
    if opt.model_name == 'screen_model':
        model = BinaryBaseModel(opt)
    elif opt.model_name == 'rerank_model':
        model = ReRankSAEM(opt)
    elif opt.model_name == 'binary_saem':
        model = BinarySAEM(opt)
    else:
        model = SAEM(opt)

    if os.path.isfile(opt.resume):
        logger.info("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        # Eiters is used to show logs as the continuation of another
        # training
        model.Eiters = checkpoint['Eiters']
        logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(opt.resume, start_epoch, best_rsum))
        if opt.binary == 'True':
            logger.info('validation in the binary mode....')
            validate_binary(opt, test_loader, model, save_sm_matrix_dir=exp_dir,save_hash_code=False)
        else:
            validate(opt, test_loader, model, save_sm_matrix_dir=exp_dir)
    else:
        logger.error("=> no checkpoint found at '{}'".format(opt.resume))


if __name__ == '__main__':
    main()
