import os
import json
from tqdm import tqdm
import  torch

# Compare the results with English-Russian round-trip translation:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
print(paraphrase)
assert paraphrase == 'PyTorch is a great interface!'



def check_drive():
    try:
        path = 'drive/scan_data_caps/f30k/test_caps.txt'
        if not os.path.exists(path):
            return False
        with open(path, 'r') as f:
            for line in f.readlines():
                print(line.strip())
                break
        test_data = {'hello': 100}
        save_path = 'drive/scan_data_caps/f30k/debug.json'
        json.dump(test_data, open(save_path, 'w'))
        if os.path.exists(save_path):
            os.remove(save_path)
            return True
        else:
            return False
    except:
        return False


def read_ori_files(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            sentences.append(line)
    return sentences


def back_translation(all_sents, tl1, tl2):
    result = {}
    beam_size = [3]
    batch_size=500
    batch_cnt=len(all_sents)/batch_size
    batch_cnt=int(batch_cnt) if batch_cnt*batch_size==len(all_sents) else int(batch_cnt)+1
    print("cur file total sentences number is {}".format(len(all_sents)))
    for i in tqdm(range(batch_cnt)):
        cur_batch_sent=all_sents[i*batch_size:(i+1)*batch_size]
        cur_result = {sent:[] for sent in cur_batch_sent}
        for bs in beam_size:
            dest_rt = tl1.translate(cur_batch_sent, beam=bs)
            back_batch_sent = tl2.translate(dest_rt, beam=bs)
            for j in range(len(cur_batch_sent)):
                src_sent=cur_batch_sent[j]
                back_sent=back_batch_sent[j]
                cur_result[src_sent].append(back_sent)
        result.update(cur_result)
    return result


def main(all_file_path, translator1, translator2):
    for file_path in all_file_path:
        print("begin processing file {}".format(file_path))
        all_sents = read_ori_files(file_path)

        result_path = file_path + '_{}.json'.format(BACK_NAME)
        result = back_translation(all_sents, tl1=translator1, tl2=translator2)
        json.dump(result, open(result_path, 'w'))
        print('save result json file to {}'.format(result_path))

    print('All file processing finished')


all_file_paths = ['drive/scan_data_caps/f30k/test_caps.txt',
                  'drive/scan_data_caps/f30k/dev_caps.txt',
                  'drive/scan_data_caps/f30k/train_caps.txt',
                  'drive/scan_data_caps/coco/test_caps.txt',
                  'drive/scan_data_caps/coco/testall_caps.txt',
                  'drive/scan_data_caps/coco/dev_caps.txt',
                  'drive/scan_data_caps/coco/train_caps.txt']

BACK_NAME = 'en2ru2en'

translator1 = en2ru.cuda()
translator2 = ru2en.cuda()
main(all_file_paths, translator1, translator2)
