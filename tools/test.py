import json
from tqdm import tqdm
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import torch
import pickle


def get_imgid2object():
    input_file_path = '/m/liyz/sg_matching/s2g/spice/coco_output/merged.json'

    data = json.load(open(input_file_path))
    print('total caption number is {}'.format(len(data)))
    all_obj = {}
    imgid2objs = {}

    for info in tqdm(data):
        img_id = info['image_id'].split('_')[0]
        ref_tuples = info['ref_tuples']
        if img_id not in imgid2objs:
            imgid2objs[img_id] = set()

        for tuple in ref_tuples:
            if len(tuple['tuple']) == 1:
                obj = tuple['tuple'][0]
                imgid2objs[img_id].add(obj)
                all_obj[obj] = all_obj.get(obj, 0) + 1

    print('total image id number is {}'.format(len(imgid2objs)))
    print('total objects number is {}'.format(len(all_obj)))

    for imgid, objs in imgid2objs.items():
        objs = list(objs)
        imgid2objs[imgid] = objs

    imgid2objs_save_path = './coco_imgid2objs.json'
    json.dump(imgid2objs, open(imgid2objs_save_path, 'w'))
    print('save to {} successfully.'.format(imgid2objs_save_path))

    all_obj_save_path = './coco_obj_freq.json'
    json.dump(all_obj, open(all_obj_save_path, 'w'))
    print('save to {} successfully.'.format(all_obj_save_path))


def kmeans(data, k_cluster=1000):
    # X, y = make_blobs(n_samples=100, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
    #                   cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
    # print(X.shape)
    # plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
    # plt.show()
    # X=torch.randn(500,200).numpy()
    print('the input data shape is: ', data.shape)
    X = data
    y_pred = KMeans(n_clusters=k_cluster, random_state=9).fit_predict(X)
    print('output result shape is ', y_pred.shape)
    # print(y_pred)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()
    return y_pred


def load_vector_dict(vector_file_path):
    print('loading glove vector file...')
    pickle_file_path = vector_file_path + '_pickle.pkl'
    word2vector = {}
    if os.path.exists(pickle_file_path):
        word2vector = pickle.load(open(pickle_file_path, 'rb'))
        print('load from pickle directly')
    else:
        with open(vector_file_path, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                infos = line.split()
                word = infos[0]
                vec = np.array([float(x) for x in infos[1:]])
                word2vector[word] = vec
        pickle.dump(word2vector, open(pickle_file_path, 'wb'))
        print('save dict file into pickle file: {}'.format(pickle_file_path))
    vec_dim = word2vector['hello'].shape[0]
    print('reading glove vector file finished...    vector dimension is {}'.format(vec_dim))
    # print(len(word2vector),word2vector['hello'])
    # print(word2vector['hammer'])
    return word2vector, vec_dim


def get_all_obj(obj_freq_file, word2vec, threshold):
    obj2freq = json.load(open(obj_freq_file))
    # print(word2vec.keys())
    used_obj = []
    used_vectors = []
    for obj, cnt in obj2freq.items():
        if cnt>=threshold and obj in word2vec:
            # print(obj)
            used_obj.append(obj)
            used_vectors.append(word2vec[obj])
    print(len(used_obj),len(used_vectors))
    print('using threshold {}, the useful object number is {}'.format(threshold, len(used_obj)))
    used_vectors = np.stack(used_vectors, axis=0)
    return used_obj, used_vectors


def get_clustered_result(glove_file_path, obj_freq_file_path, save_word2clus_id_path, save_clus_id2words,
                         k_cluster=1000):
    word2vec,vec_dim = load_vector_dict(vector_file_path=glove_file_path)
    used_obj, used_vectors = get_all_obj(obj_freq_file=obj_freq_file_path, word2vec=word2vec, threshold=10)

    clustered_idxs = kmeans(used_vectors, k_cluster=1000)

    word2clus_id = {}
    clus_id2words = {i: [] for i in range(k_cluster)}
    for i in range(len(used_obj)):
        word = used_obj[i]
        idx = int(clustered_idxs[i])
        word2clus_id[word] = idx
        clus_id2words[idx].append(word)

    json.dump(word2clus_id, open(save_word2clus_id_path, 'w'))
    json.dump(clus_id2words, open(save_clus_id2words, 'w'))

    print('finished.........')



glove_file_path='/S4/MI/liyz/data/glove/glove.6B.200d.txt'
obj_freq_file_path='/S4/MI/liyz/saem_retrieval/data/cocoid2obj/coco_obj_freq.json'
save_word2clus_id_path='/S4/MI/liyz/saem_retrieval/data/cocoid2obj/obj_to_clustered_id.json'
save_clus_id2words='/S4/MI/liyz/saem_retrieval/data/cocoid2obj/clustered_id_to_obj.json'
get_clustered_result(glove_file_path,obj_freq_file_path,save_word2clus_id_path,save_clus_id2words,k_cluster=1000)

