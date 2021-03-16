import json
import os
from tqdm import tqdm
import random

DATANAME = ['f30k_precomp']

root_dir = '/S4/MI/liyz/data/scan_data'
en2de_back_trans_dir = '/S4/MI/liyz/data/scan_data/back_translation/en2de2en/'
en2ru_back_trans_dir = '/S4/MI/liyz/data/scan_data/back_translation/en2ru2en/'


def main(data_name):
    print('current dataname is {}'.format(data_name))
    data_split = ['test', 'dev', 'train']
    adversary_type = ['_ex_noun', '_ex_num', '_ex_rela']

    if data_name == 'coco_precomp':
        data_split.append('testall')
    data_set_dir = os.path.join(root_dir, data_name)

    for spt in data_split:
        cur_raw_captions_file = os.path.join(data_set_dir, '{}_caps.txt'.format(spt))
        print('begin execute file is: {}'.format(cur_raw_captions_file))

        en2de_back_file = os.path.join(en2de_back_trans_dir, "{}/{}_caps_en2de2en.json".format(data_name, spt))
        en2de_back_trans = json.load(open(en2de_back_file))  # 德语回翻的re-phrase句子

        en2ru_back_file = os.path.join(en2ru_back_trans_dir, '{}/{}_caps_en2ru2en.json'.format(data_name, spt))
        en2ru_back_trans = json.load(open(en2ru_back_file))  # 俄语回翻的re-phrase句子

        noun_adv_dir = os.path.join(data_set_dir, '{}_ex_noun'.format(spt))
        num_adv_dir = os.path.join(data_set_dir, '{}_ex_num'.format(spt))
        rela_adv_dir = os.path.join(data_set_dir, '{}_ex_rela'.format(spt))

        with open(cur_raw_captions_file) as f:
            all_captions = f.readlines()  # 原始句子

        result = {}
        for i in tqdm(range(len(all_captions))):
            tmp = {}

            cur_cap = all_captions[i].strip()
            tmp['raw'] = cur_cap
            try:
                en2de = en2de_back_trans[cur_cap]
            except:
                print("WARNING!!!   {} not in the en2de file".format(cur_cap))
                en2de=cur_cap
            try:
                en2ru = en2ru_back_trans[cur_cap]
            except:
                print("WARNING!!!   {} not in the en2ru file".format(cur_cap))
                en2ru = cur_cap

            tmp['re-pharse'] = [en2de, en2ru]

            adv_caps = {}

            noun_adv_file = os.path.join(noun_adv_dir, '{}.txt'.format(i))
            with open(noun_adv_file) as f:
                all_adv_noun_caps = f.readlines()
            random.shuffle(all_adv_noun_caps)
            noun_adv=[x.strip() for x in all_adv_noun_caps[:10]]
            while len(noun_adv)<10:
                noun_adv.append('unk')

            num_adv_file = os.path.join(num_adv_dir, '{}.txt'.format(i))
            with open(num_adv_file) as f:
                all_adv_num_caps = f.readlines()
            random.shuffle(all_adv_num_caps)
            num_adv=[x.strip() for x in all_adv_num_caps[:10]]
            while len(num_adv)<10:
                num_adv.append('unk')

            rela_adv_file = os.path.join(rela_adv_dir, '{}.txt'.format(i))
            with open(rela_adv_file) as f:
                all_adv_rela_caps =f.readlines()
            random.shuffle(all_adv_rela_caps)
            rela_adv=[x.strip() for x in all_adv_rela_caps[:10]]
            while len(rela_adv)<10:
                rela_adv.append('unk')

            adv_caps={'noun':noun_adv,'num':num_adv,'rela':rela_adv}

            # j = 0
            # while len(adv_caps) < 5 and j < len(all_adv_noun_caps):
            #     adv_caps.append(all_adv_noun_caps[j].strip())
            #     j += 1
            #
            # j = 0
            # while len(adv_caps) < 8 and j < len(all_adv_rela_caps):
            #     adv_caps.append(all_adv_rela_caps[j].strip())
            #     j += 1
            #
            # j = 0
            # while len(adv_caps) < 10 and j < len(all_adv_num_caps):
            #     adv_caps.append(all_adv_num_caps[j].strip())
            #     j += 1
            #
            # if len(adv_caps) < 10:  # 如果小于10个句子，从所有的adv句子中随机选补齐，否则的话使用unk当做一个句子补齐10句
            #     all_adv_caps = all_adv_noun_caps + all_adv_rela_caps + all_adv_num_caps
            #     need_caps = 10 - len(adv_caps)
            #     t_caps = random.choices(all_adv_caps, k=need_caps) if len(all_adv_caps) > 0 else ['<unk>'] * need_caps
            #     adv_caps += t_caps
            #
            # assert len(adv_caps) == 10

            tmp['adversary'] = adv_caps

            result[i] = tmp

        save_file_path = os.path.join(data_set_dir, '{}_caps+rephrase+30advs.json'.format(spt))
        json.dump(result, open(save_file_path, 'w'))
        print('save processed file into {}'.format(save_file_path))


if __name__ == '__main__':
    for data_name in DATANAME:
        main(data_name)
