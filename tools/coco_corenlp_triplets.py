from stanza.server import CoreNLPClient
import json
import os
from tqdm import  tqdm
file_paths=['/S4/MI/data/mscoco/annotations/captions_val2014.json','/S4/MI/data/mscoco/annotations/captions_train2014.json']
save_path='/S4/MI/liyz/coco_triplets_corenlp.json'
result={}


with CoreNLPClient(annotators=['openie'], timeout=30000, memory='64G',threads=32) as client:
    for file in file_paths:
        print('executing file {}'.format(file_paths))
        data=json.load(open(file))
        captions=data['annotations']
        for cap in tqdm(captions):
            img_id=cap['image_id']
            if img_id not in result:
                result[img_id]=[]
            raw_sent = cap['caption']
            ann=client.annotate(raw_sent)
            parse_result=ann.sentence[0].openieTriple
            for triplet in parse_result:
                tmp=[triplet.subject,triplet.relation,triplet.object]
                result[img_id].append(tmp)
    json.dump(result,open(save_path,'w'))
    print('save result file to {}'.format(save_path))



