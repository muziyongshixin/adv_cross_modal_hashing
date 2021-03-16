from googletrans import Translator
import  time
import json
import random

import torch
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')




dest_langs=['zh-cn','de','fr','es','ru']

useful_proxies=["118.89.234.236:8787","118.99.95.234:8080",'120.26.208.102:88', '39.97.234.82:3128', '201.48.183.180:3128', '39.137.69.7:80', '182.61.179.157:8888', '47.106.124.179:80', '101.37.118.54:8888', '117.131.119.116:80', '116.114.19.211:443', '118.25.35.202:9999', '39.137.107.98:80', '39.137.107.98:8080', '101.231.104.82:80', '14.21.69.222:3128', '183.146.213.198:80']


def read_ori_files(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            sentences.append(line)
    return sentences


def translate(input_sentences, translator_pool,batch_size=20,target_lang='zh-cn'):
    result={}

    i=0
    t_idx=0
    while i<len(input_sentences):
        batch_sents = input_sentences[i:i + batch_size]
        translator = translator_pool[t_idx % len(translator_pool)]
        t_idx = i
        try:
            target_rt = translator.translate(batch_sents, src='en', dest=target_lang)
            trans_sents = [rt.text for rt in target_rt]
            back_trans = translator.translate(trans_sents, src=target_lang, dest='en')
            back_trans_sents=[x.text for x in back_trans]
            for j in range(len(batch_sents)):
                result[batch_sents[j]] = back_trans_sents[j]
            if len(result) % 500 == 0:
                print('{} sentences have been back translated'.format(len(result)))
                print(trans_sents)
                print(back_trans_sents)
            print(i)
            i += batch_size
        except:
            print("proxy {} failed".format(translator.session.proxies))
            t_idx=random.randint(0,100)

    return result

if __name__ == '__main__':
    translator_pool=[Translator(service_urls=['translate.google.cn'])]
    # for pro in useful_proxies:
    #     translator = Translator(service_urls=['translate.google.cn'],proxies={'https':pro})
    #     translator_pool.append(translator)

    translator=translator_pool[0]
    rt=translator.translate('hello my name is jack', dest='zh-cn')
    print(rt.text)
    file_list=['C:/Users/木子-勇士心/Desktop/caps/test_caps.txt',
               'C:/Users/木子-勇士心/Desktop/caps/dev_caps.txt',
               'C:/Users/木子-勇士心/Desktop/caps/train_caps.txt']
    for file in file_list:
        all_sents = read_ori_files(file_path=file)
        print('{} file read finished'.format(file))
        result=translate(all_sents, translator_pool)
        save_path=file.split('.')[0]+"_trans.json"
        json.dump(result,open(save_path,'w'))
        print('save translated result into {}'.format(save_path))






    LANGUAGES = {
        'af': 'afrikaans',
        'sq': 'albanian',
        'am': 'amharic',
        'ar': 'arabic',
        'hy': 'armenian',
        'az': 'azerbaijani',
        'eu': 'basque',
        'be': 'belarusian',
        'bn': 'bengali',
        'bs': 'bosnian',
        'bg': 'bulgarian',
        'ca': 'catalan',
        'ceb': 'cebuano',
        'ny': 'chichewa',
        'zh-cn': 'chinese (simplified)',
        'zh-tw': 'chinese (traditional)',
        'co': 'corsican',
        'hr': 'croatian',
        'cs': 'czech',
        'da': 'danish',
        'nl': 'dutch',
        'en': 'english',
        'eo': 'esperanto',
        'et': 'estonian',
        'tl': 'filipino',
        'fi': 'finnish',
        'fr': 'french',
        'fy': 'frisian',
        'gl': 'galician',
        'ka': 'georgian',
        'de': 'german',
        'el': 'greek',
        'gu': 'gujarati',
        'ht': 'haitian creole',
        'ha': 'hausa',
        'haw': 'hawaiian',
        'iw': 'hebrew',
        'hi': 'hindi',
        'hmn': 'hmong',
        'hu': 'hungarian',
        'is': 'icelandic',
        'ig': 'igbo',
        'id': 'indonesian',
        'ga': 'irish',
        'it': 'italian',
        'ja': 'japanese',
        'jw': 'javanese',
        'kn': 'kannada',
        'kk': 'kazakh',
        'km': 'khmer',
        'ko': 'korean',
        'ku': 'kurdish (kurmanji)',
        'ky': 'kyrgyz',
        'lo': 'lao',
        'la': 'latin',
        'lv': 'latvian',
        'lt': 'lithuanian',
        'lb': 'luxembourgish',
        'mk': 'macedonian',
        'mg': 'malagasy',
        'ms': 'malay',
        'ml': 'malayalam',
        'mt': 'maltese',
        'mi': 'maori',
        'mr': 'marathi',
        'mn': 'mongolian',
        'my': 'myanmar (burmese)',
        'ne': 'nepali',
        'no': 'norwegian',
        'ps': 'pashto',
        'fa': 'persian',
        'pl': 'polish',
        'pt': 'portuguese',
        'pa': 'punjabi',
        'ro': 'romanian',
        'ru': 'russian',
        'sm': 'samoan',
        'gd': 'scots gaelic',
        'sr': 'serbian',
        'st': 'sesotho',
        'sn': 'shona',
        'sd': 'sindhi',
        'si': 'sinhala',
        'sk': 'slovak',
        'sl': 'slovenian',
        'so': 'somali',
        'es': 'spanish',
        'su': 'sundanese',
        'sw': 'swahili',
        'sv': 'swedish',
        'tg': 'tajik',
        'ta': 'tamil',
        'te': 'telugu',
        'th': 'thai',
        'tr': 'turkish',
        'uk': 'ukrainian',
        'ur': 'urdu',
        'uz': 'uzbek',
        'vi': 'vietnamese',
        'cy': 'welsh',
        'xh': 'xhosa',
        'yi': 'yiddish',
        'yo': 'yoruba',
        'zu': 'zulu',
        'fil': 'Filipino',
        'he': 'Hebrew'
    }

