import os
import json
from tqdm import tqdm
import pickle
import numpy as np
import json
from tqdm import tqdm
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import torch
from numpy import linalg
import time


# val_path = '/S4/MI/data/mscoco/annotations/captions_val2014.json'
# train_path = '/S4/MI/data/mscoco/annotations/captions_train2014.json'
#
# save_path = '/S4/MI/liyz/data/scan_data/coco_precomp/id2filename.json'
#
# all_paths = [val_path, train_path]
# id2filename = {}
#
# for file in all_paths:
#     data = json.load(open(file))
#     images = data['images']
#     print('cur file path is {}'.format(file))
#     for ele in tqdm(images):
#         id = ele['id']
#         file_path = ele['file_name']
#         if 'train' in file_path:
#             file_path = os.path.join('train2014', file_path)
#         elif 'val' in file_path:
#             file_path = os.path.join('val2014', file_path)
#         else:
#             print(file_path)
#         id2filename[id] = file_path
#
# json.dump(id2filename, open(save_path, 'w'))
# print('save mapping file into {}'.format(save_path))

# image_json_info_dir = '/S4/MI/liyz/data/coco_concept/new_tuples'
# concept_to_cls_file = '/S4/MI/liyz/data/coco_concept/tuple_2_cls.pkl'
# save_path = '/S4/MI/liyz/data/coco_concept/imgid_2_concept_idxs.pkl'
#
#
# result={}
#
# def get_all_paths(dir):
#     paths = []
#     for file in os.listdir(dir):
#         cur = os.path.join(dir, file)
#         paths.append(cur)
#     return paths
#
# tuple2idx=pickle.load(open(concept_to_cls_file,'rb'))
#
# all_paths=get_all_paths(image_json_info_dir)
#
# zero_cnt=0
# for file in tqdm(all_paths):
#     cur=json.load(open(file))
#     img_id=cur['id']
#     cur_label=np.zeros((642),dtype=np.int)
#     concept_tuple=cur['tuple']
#     for concept in concept_tuple:
#         tmp=' '.join(concept)
#         cls=tuple2idx[tmp] if tmp in tuple2idx else []
#         for idx in cls:
#             cur_label[idx]=1
#
#     if cur_label.sum()==0:
#         zero_cnt+=1
#         if zero_cnt%1000==0:
#             print(zero_cnt,'============')
#     result[img_id]=cur_label
#
#
# pickle.dump(result,open(save_path,'wb'))
# print('finished')
#

# concept_to_cls_file = '/S4/MI/liyz/data/coco_concept/tuple_2_cls.pkl'
# save_path='/S4/MI/liyz/data/coco_concept/class_id_2_concepts.json'
# tuple2idx=pickle.load(open(concept_to_cls_file,'rb'))
#
# result={i:[] for i in range(643)}
# for concepts,idxs in tqdm(tuple2idx.items()):
#     for id in idxs :
#         result[id].append(concepts)
#
# json.dump(result,open(save_path,'w'))


def kmeans():
    X, y = make_blobs(n_samples=100, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
    print(X.shape)
    plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
    plt.show()
    # X=torch.randn(1000,200).numpy()
    y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
    print(y_pred.shape)
    print(y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()


# kmeans()


def hamming_distance_v1(A, b):
    r = (1 << np.arange(8))[:, None]
    na = A.shape[0]
    H = np.zeros((na), dtype=np.float32)
    for i in range(0, na):
        a = A[i, :]
        c = np.bitwise_xor(a, b)
        d = c & r
        H[i] = np.count_nonzero(d != 0)
    return H


def hamming_distance_v2(A, b):
    db_size = A.shape[0]
    r = np.expand_dims(np.expand_dims((1 << np.arange(8)), axis=0).repeat(db_size, axis=0), axis=1)
    result = np.count_nonzero(np.expand_dims(np.bitwise_xor(A, b), axis=-1) & r != 0, axis=-1).sum(axis=-1)
    return result





def hamming_distance_v3(A, b):
    import gmpy2
    from gmpy2 import mpz, hamdist, pack
    na = len(A)
    H = np.zeros((na), dtype=np.float32)
    b = pack(b, 64)
    for i in range(0, na):
        a = A[i]
        a = pack(a, 64)
        H[i] = gmpy2.hamdist(a, b)
        # H[i] = gmpy2.popcount(np.bitwise_xor(a,b))
    return H


table = np.array(
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3,
     3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
     3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
     4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
     3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6,
     6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5,
     4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8],dtype=np.uint8)

def hamming_distance_v4(A, b):
    query_n, word_n = b.shape
    db_size, word_n1 = A.shape

    assert word_n == word_n1
    Dh = np.zeros((query_n, db_size), 'uint16')

    for i in range(query_n):
        c = np.bitwise_xor(A, b[i])  # db_size，word_n
        for n in range(word_n1):
            cur_idx = c[:, n]
            Dh[i,:]=Dh[i,:]+table[cur_idx]
    return Dh


A = np.zeros((100, 4), dtype=np.uint8)
b=np.zeros((4,4),dtype=np.uint8)
result=hamming_distance_v4(A,b)
print(result)
print(result.shape)
#
# start = time.time()
# for i in range(2):
#     b = np.zeros((4,), dtype=np.int64)
#     result = hamming_distance_v1(A, b)
# print('100 test on hamming 256 bit time={} function v1'.format(time.time() - start))
exit(0)


def get_cosine_similarity_v1(A, b):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1 = A
    x2 = b
    db_size = x1.shape[0]
    result = np.zeros((db_size,), dtype=np.float32)
    x1_norm = x1 / linalg.norm(x2, axis=1)[:, None]
    x2_norm = x2 / linalg.norm(x2, axis=1)[:, None]
    for i in range(db_size):
        cur = np.matmul(x1_norm[i, :], x2_norm.T)
        result[i] = cur.squeeze()
    return result


def get_cosine_similarity_v2(A, b):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1 = A
    x2 = b
    x1_norm = x1 / linalg.norm(x2, axis=1)[:, None]
    x2_norm = x2 / linalg.norm(x2, axis=1)[:, None]
    res = np.matmul(x1_norm, x2_norm.T)
    res = res.squeeze()
    return res


def get_l2_distance(A, b):
    diff = ((A - b) ** 2).sum(axis=1)
    diff = diff ** 0.5
    # print(diff.shape)
    return diff


A = np.random.randn(100000, 512)
start = time.time()
for i in range(100):
    b = np.random.randn(1, 512)
    result = hamming_distance_v4(A, b)
print('100 test on continue vec {} dimension 1024  cos_sim function v2'.format(time.time() - start))
exit(0)

#
# A=np.array([[1,3,1,0],[77,0,1,0]],dtype=np.int64)
# b=np.array([[1,3,1,0]],dtype=np.int64)
# H2=hamming_distance_v2(A,b)
# print(H2)
# b=np.array([1,3,1,0],dtype=np.int64)
# H=hamming_distance_v3(A,b)
# print(H)


# SIZES=[]
SIZES = [200, 5000, 10000, 50000, 100000]
QUERY_NUMBER = 100

for DATABASE_SIZE in SIZES:
    print('cur data base size is {}=============================='.format(DATABASE_SIZE))
    # test for conitnus vector
    #
    A = np.random.randn(DATABASE_SIZE, 256)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.random.randn(1, 256)
        result = get_cosine_similarity_v1(A, b)
    print('100 test on continue vec {} dimension 256  cos_sim function v1'.format(time.time() - start))

    A = np.random.randn(DATABASE_SIZE, 256)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.random.randn(1, 256)
        result = get_cosine_similarity_v2(A, b)
    print('100 test on continue vec {} dimension 256  cos_sim function v2'.format(time.time() - start))

    A = np.random.randn(DATABASE_SIZE, 256)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.random.randn(1, 256)
        result = get_l2_distance(A, b)
    print('100 test on continue vec {} dimension 256  l2 function v2'.format(time.time() - start))

    A = np.zeros((DATABASE_SIZE, 4), dtype=np.int64)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.zeros((4,), dtype=np.int64)
        result = hamming_distance_v1(A, b)
    print('100 test on hamming 256 bit time={} function v1'.format(time.time() - start))

    A = np.zeros((DATABASE_SIZE, 4), dtype=np.int64)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.zeros((1, 4), dtype=np.int64)
        result = hamming_distance_v2(A, b)
    print('100 test on hamming 256 bit time={} function v2'.format(time.time() - start))

    A = np.zeros((DATABASE_SIZE, 4), dtype=np.int64)
    A = [[int(x) for x in A[i]] for i in range(DATABASE_SIZE)]
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.zeros((4,), dtype=np.int64)
        b = [int(x) for x in b]
        result = hamming_distance_v3(A, b)
    print('100 test on hamming 256 bit time={} function v3'.format(time.time() - start))

    A = np.zeros((DATABASE_SIZE, 32), dtype=np.int64)
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.zeros((1, 32), dtype=np.int64)
        result = hamming_distance_v2(A, b)
    print('100 test on hamming 2048 bit time={} function v2'.format(time.time() - start))

    A = np.zeros((DATABASE_SIZE, 32), dtype=np.int64)
    A = [[int(x) for x in A[i]] for i in range(DATABASE_SIZE)]
    start = time.time()
    for i in range(QUERY_NUMBER):
        b = np.zeros((32,), dtype=np.int64)
        b = [int(x) for x in b]
        result = hamming_distance_v3(A, b)
    print('100 test on hamming 2048 bit time={} function v3'.format(time.time() - start))
