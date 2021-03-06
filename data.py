import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json
import tokenization


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

    return tokens, input_ids, input_mask, input_type_ids


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, opt, **kwargs):
        loc = data_path + '/'
        image_root_dir = opt.image_root_dir

        self.need_ori_image = opt.need_ori_image
        self.transform = kwargs['transform'] if 'transform' in kwargs else None
        if self.need_ori_image:
            # imgid to image file path mapping
            mapping_file_path = os.path.join(data_path, 'id2filename.json')
            self.imgid2filepath = json.load(open(mapping_file_path))

            # image_ids
            self.img_ids = []
            self.img_paths = []
            with open(os.path.join(loc, '{}_ids.txt'.format(data_split))) as f:
                for line in f:
                    cur_img_id = line.strip()
                    self.img_ids.append(cur_img_id)
                    self.img_paths.append(os.path.join(image_root_dir, self.imgid2filepath[str(cur_img_id)]))

        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.tokenizer = tokenization.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        self.max_words = opt.max_words

    def __getitem__(self, index):
        # handle the image redundancy
        sample_index = int(index / self.im_div)

        img_id = self.img_ids[sample_index]
        img_path = self.img_paths[sample_index]
        ori_image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            ori_image = self.transform(ori_image)

        rois = torch.Tensor(self.images[sample_index])
        caption = self.captions[index]
        tokens, input_ids, input_mask, input_type_ids = convert_to_feature(caption, self.max_words, self.tokenizer)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        return ori_image, rois, input_ids, input_mask, input_type_ids, index, sample_index

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
    ori_images, rois, input_ids, input_mask, input_type_ids, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    ori_images = torch.stack(ori_images, 0)
    images = torch.stack(rois, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [torch.sum(cap) for cap in input_mask]
    input_ids = torch.stack(input_ids, 0)
    input_mask = torch.stack(input_mask, 0)
    input_type_ids = torch.stack(input_type_ids, 0)

    ids = np.array(ids)

    return ori_images, images, input_ids, input_mask, input_type_ids, lengths, ids


def get_precomp_loader(data_path, data_split, opt, batch_size=100, shuffle=True, num_workers=2, transform=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn_bert,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    if opt.need_ori_image:
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

    val_loader = get_precomp_loader(dpath, 'dev', opt, batch_size // 4, False, workers, transform=val_transform)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt, batch_size, False, workers)
    return test_loader
