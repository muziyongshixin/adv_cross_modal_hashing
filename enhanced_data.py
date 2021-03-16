# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import tokenization
import json
import logging
from tqdm import tqdm
import pickle
from IPython import embed
import random

logger = logging.getLogger(__name__)


def convert_to_feature(raw, seq_length, tokenizer):
    line = tokenization.convert_to_unicode(raw)
    tokens_a = tokenizer.tokenize(line)
    # Modifies `tokens_a` in place so that the total
    # length is less than the specified length.
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    # tokens.append("[CLS]")
    # input_type_ids.append(0)
    # for token in tokens_a:
    #     tokens.append(token)
    #     input_type_ids.append(0)
    # tokens.append("[SEP]")
    # input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Zero-pad up to the sequence length.
    input_ids += [0] * (seq_length - len(input_ids))
    # while len(input_ids) < seq_length:
    #     input_ids.append(0)
    #     input_mask.append(0)
    #     input_type_ids.append(0)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * seq_length
    if len(tokens) < seq_length:
        input_mask[-(seq_length - len(tokens)):] = [0] * (seq_length - len(tokens))

    input_type_ids = [0] * seq_length

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)

    return tokens, input_ids, input_mask, input_type_ids


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, opt, eager_execution=False, **kwargs):

        self.data_split=data_split
        self.opt=opt
        self.need_raw_image = opt.need_raw_image
        self.need_concept_label = opt.need_concept_label
        self.transform = kwargs['transform'] if 'transform' in kwargs else None
        self.roi_feature = None
        self.concept_label = None
        self.part_idx2ori_idx=None

        self.img_ids = []
        if 'coco' in data_path:
            img_ids_file=os.path.join(data_path, '{}_ids.txt'.format(data_split))
        elif 'f30' in data_path:
            img_ids_file=os.path.join(data_path,'{}_imgids.txt'.format(data_split))
        with open(img_ids_file) as f:
            for line in f:
                cur_img_id = line.strip()
                self.img_ids.append(cur_img_id)

        logger.info('>>>construct dataset for {}, split is {}'.format(data_path,data_split))

        if self.need_concept_label:
            self.concept_label = pickle.load(open(opt.concept_file_path, 'rb'))

        # captions_file_path = os.path.join(data_path, '{}_caps+rephrase+adv.json'.format(data_split))
        captions_file_path = os.path.join(data_path, '{}_caps+rephrase+30advs.json'.format(data_split))

        data = json.load(open(captions_file_path))
        logger.info('cur data split is {}. captions samples number is {}'.format(data_split, len(data)))

        self.tokenizer = tokenization.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        self.eager_execution = eager_execution
        self.max_words = opt.max_words

        self.need_adversary_data = opt.need_adversary_data
        self.need_rephrase_data = opt.need_rephrase_data
        self.adv_num=opt.adversary_num
        # Captions
        # if self.eager_execution:
        #     logger.info('data eager execution is activated,preprocessing captions to tensor...')
        #     self.captions = {}
        #     for key, caps in tqdm(data.items()):
        #
        #         raw_caption = caps['raw'].strip()
        #         raw_data = convert_to_feature(raw_caption, self.max_words, self.tokenizer)
        #
        #         re_phrase_captions = [caps['re-pharse'][0][0], caps['re-pharse'][1][0]]
        #         re_phrase_data = []
        #         for re_phr in re_phrase_captions:
        #             re_phr = re_phr.strip()
        #             re_phr = re_phr.replace('.', ' .')
        #             re_phrase_data.append(convert_to_feature(re_phr, self.max_words, self.tokenizer))
        #
        #         adversary_captions = caps['adversary']
        #         adversary_data = []
        #         for adv_cap in adversary_captions:
        #             adv_cap = adv_cap.strip()
        #             adversary_data.append(convert_to_feature(adv_cap, self.max_words, self.tokenizer))
        #
        #         tmp = {'raw': raw_data, 're_phrase': re_phrase_data, 'adversary': adversary_data}
        #         self.captions[key] = tmp
        # else:
        #     self.captions = data

        self.captions = data

        if self.need_raw_image:
            # imgid to image file path mapping
            mapping_file_path = os.path.join(data_path, 'id2filename.json')
            self.imgid2filepath = json.load(open(mapping_file_path))
            self.img_paths = []
            for cur_img_id in self.img_ids:
                self.img_paths.append(os.path.join(opt.image_root_dir, self.imgid2filepath[str(cur_img_id)]))

        self.need_roi_feature = opt.need_roi_feature
        if self.need_roi_feature:
            # Image features
            self.roi_feature = np.load(os.path.join(data_path, '%s_ims.npy' % data_split))
            logger.info('faster rcnn image feature loading finished...')

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if (self.roi_feature is not None and self.roi_feature.shape[0] != self.length) or len(self.img_ids)!=self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        logger.info('cur data split is {}, self.im_div={}'.format(data_split,self.im_div))

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        if data_split=='train' and opt.part_train_data != '':
            self.part_idx2ori_idx={}
            part_train_ids=set(json.load(open(opt.part_train_data))['split2ids']['train'])
            for i,img_id in enumerate(self.img_ids):
                if img_id in part_train_ids:
                    self.part_idx2ori_idx[len(self.part_idx2ori_idx)]=i
            self.length=len(self.part_idx2ori_idx)*5
            logger.info('using training img number is {}, self.length={}'.format(len(self.part_idx2ori_idx),self.length))


    def __getitem__(self, index):
        if self.part_idx2ori_idx is not None :
            part_sample_idx=index//5
            sent_shift=index%5
            ori_idx=self.part_idx2ori_idx[part_sample_idx]
            index=ori_idx*5+sent_shift
        # handle the image redundancy
        sample_index = int(index / self.im_div)
        img_id = self.img_ids[sample_index]

        raw_image = []
        roi_feature = []
        concept_label = []

        if self.need_raw_image:
            img_path = self.img_paths[sample_index]
            raw_image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                raw_image = self.transform(raw_image)

        if self.need_roi_feature:
            roi_feature = torch.tensor(self.roi_feature[sample_index])

        if  self.need_concept_label:
            if int(img_id) not in self.concept_label:
                concept_label=torch.zeros(self.opt.concept_num)
            else:
                concept_label = torch.from_numpy(self.concept_label[int(img_id)]).float()

        raw_input_ids, raw_input_mask, raw_input_type_ids, \
        re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
        adv_ids, adv_mask, adv_type_ids = self.get_cap_tensor(index)

        return raw_image, roi_feature, concept_label, \
               raw_input_ids, raw_input_mask, raw_input_type_ids, \
               re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
               adv_ids, adv_mask, adv_type_ids, \
               img_id, sample_index

    def get_cap_tensor(self, index):
        '''
        :param index:数据的index
        :return: caption 相关的tensor数据
        '''
        re_phrase_ids, re_phrase_mask, re_phrase_type_ids = [], [], []
        adv_ids, adv_mask, adv_type_ids = [], [], []

        # if self.eager_execution:
        #     caption_data = self.captions[str(index)]
        #     raw_tokens, raw_input_ids, raw_input_mask, raw_input_type_ids = caption_data['raw']
        #
        #     if self.need_rephrase_data:
        #         re_phrase_tokens, re_phrase_ids, re_phrase_mask, re_phrase_type_ids = [], [], [], []
        #         for r_to, r_ids, r_mask, r_type_ids in caption_data['re_phrase']:
        #             re_phrase_tokens.append(r_to)
        #             re_phrase_ids.append(r_ids)
        #             re_phrase_mask.append(r_mask)
        #             re_phrase_type_ids.append(r_type_ids)
        #
        #         re_phrase_ids = torch.stack(re_phrase_ids, dim=0)
        #         re_phrase_mask = torch.stack(re_phrase_mask, dim=0)
        #         re_phrase_type_ids = torch.stack(re_phrase_type_ids, dim=0)
        #     if self.need_adversary_data:
        #         adv_tokens, adv_ids, adv_mask, adv_type_ids = [], [], [], []
        #         for a_to, a_ids, a_mask, a_type_ids in caption_data['adversary']:
        #             adv_tokens.append(a_to)
        #             adv_ids.append(a_ids)
        #             adv_mask.append(a_mask)
        #             adv_type_ids.append(a_type_ids)
        #
        #         adv_ids = torch.stack(adv_ids, dim=0)
        #         adv_mask = torch.stack(adv_mask, dim=0)
        #         adv_type_ids = torch.stack(adv_type_ids, dim=0)
        #
        # else:
        caption = self.captions[str(index)]
        raw_tokens, raw_input_ids, raw_input_mask, raw_input_type_ids = convert_to_feature(caption['raw'],
                                                                                           self.max_words,
                                                                                           self.tokenizer)

        if self.need_rephrase_data:
            re_phrase_tokens, re_phrase_ids, re_phrase_mask, re_phrase_type_ids = [], [], [], []
            for re_phrase_caption in [caption['re-pharse'][0][0], caption['re-pharse'][1][0]]:
                re_phrase_caption = re_phrase_caption.strip()
                re_phrase_caption = re_phrase_caption.replace('.', ' .')
                r_to, r_ids, r_mask, r_type_ids = convert_to_feature(re_phrase_caption, self.max_words,
                                                                     self.tokenizer)
                re_phrase_tokens.append(r_to)
                re_phrase_ids.append(r_ids)
                re_phrase_mask.append(r_mask)
                re_phrase_type_ids.append(r_type_ids)

            re_phrase_ids = torch.stack(re_phrase_ids, dim=0)
            re_phrase_mask = torch.stack(re_phrase_mask, dim=0)
            re_phrase_type_ids = torch.stack(re_phrase_type_ids, dim=0)


        if self.need_adversary_data:
            adv_tokens, adv_ids, adv_mask, adv_type_ids = [], [], [], []
            all_adv_caps=caption['adversary']
            if self.opt.adversary_type=='noun':
                used_adv_caps=all_adv_caps['noun']
            elif self.opt.adversary_type=='num':
                used_adv_caps=all_adv_caps['num']
            elif self.opt.adversary_type=='rela':
                used_adv_caps=all_adv_caps['rela']
            elif self.opt.adversary_type=='mixed':
                used_adv_caps=all_adv_caps['noun']+all_adv_caps['num']+all_adv_caps['rela']
                random.shuffle(used_adv_caps)
            else:
                used_adv_caps=[]

            for adversary_caption in used_adv_caps:
                adversary_caption = adversary_caption.strip()
                a_to, a_ids, a_mask, a_type_ids = convert_to_feature(adversary_caption, self.max_words,
                                                                     self.tokenizer)
                adv_tokens.append(a_to)
                adv_ids.append(a_ids)
                adv_mask.append(a_mask)
                adv_type_ids.append(a_type_ids)

            adv_ids = torch.stack(adv_ids, dim=0)
            adv_mask = torch.stack(adv_mask, dim=0)
            adv_type_ids = torch.stack(adv_type_ids, dim=0)
            if self.adv_num>0:
                adv_ids=adv_ids[:self.adv_num]
                adv_mask=adv_mask[:self.adv_num]
                adv_type_ids=adv_type_ids[:self.adv_num]

        return raw_input_ids, raw_input_mask, raw_input_type_ids, \
               re_phrase_ids, re_phrase_mask, re_phrase_type_ids, \
               adv_ids, adv_mask, adv_type_ids

    def __len__(self):
        return self.length


def collate_fn_bert(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: sum(x[2]), reverse=True)
    images, input_ids, input_mask, input_type_ids, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [torch.sum(cap) for cap in input_mask]
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    input_type_ids = torch.stack(input_type_ids, 0)

    ids = np.array(ids)

    return images, input_ids, input_mask, input_type_ids, lengths, ids




def get_precomp_loader(data_path, data_split, opt, batch_size=64, shuffle=True, num_workers=2, transform=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              # collate_fn=collate_fn_bert,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    if opt.need_raw_image:
        image_size = opt.image_size
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform = None

    train_loader = get_precomp_loader(dpath, 'train', opt, batch_size, True, workers, transform=train_transform)

    val_loader = get_precomp_loader(dpath, 'test', opt, batch_size // 4, False, workers, transform=val_transform)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    image_size = opt.image_size
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_loader = get_precomp_loader(dpath, split_name, opt, batch_size, False, workers, transform=val_transform)
    return test_loader
