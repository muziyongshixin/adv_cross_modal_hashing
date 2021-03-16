import logging
import logging.config
import os
import torch
import  pickle
import numpy as np

logger=logging.getLogger(__name__)


def init_logging(exp_dir, config_path='config/logging_config.yaml'):
    """
    initial logging module with config
    :param config_path:
    :return:
    """
    import yaml, sys
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        config["handlers"]["info_file_handler"]["filename"] = os.path.join(exp_dir, "info.log")
        config["handlers"]["time_file_handler"]["filename"] = os.path.join(exp_dir, "time.log")
        config["handlers"]["error_file_handler"]["filename"] = os.path.join(exp_dir, "error.log")

        logging.config.dictConfig(config)
    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)


def get_hamming_dist(img_code, cap_code):
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    code_len = img_code.shape[1]
    similarity_matrix = []
    for i in range(0, img_code.shape[0], 10):  # 分片计算防止爆内存
        cur_query_code = img_code[i:i + 10].to(device) # size(10,code_len)
        cur_matrix=[]
        for j in range(0,cap_code.shape[0],1000):
            cur_ref_code=cap_code[j:j+1000].to(device)
            cur_part=(code_len - (cur_query_code.unsqueeze(1) * cur_ref_code.unsqueeze(0)).sum(dim=-1)) / 2 # size(10,1000)
            cur_part=cur_part.cpu()
            cur_matrix.append(cur_part)
        cur_matrix = torch.cat(cur_matrix,dim=-1).cpu()
        similarity_matrix.append(cur_matrix)
    similarity_matrix = torch.cat(similarity_matrix, dim=0).cpu()
    return similarity_matrix



def save_vector_to_file(data, file_name):
    pickle.dump(data, open(file_name, 'wb'))
    logger.info('save vector file to {}'.format(file_name))


def save_similarity_matrix(matrix_data,save_path):
    np.save(save_path,matrix_data)
    logger.info('save similarity matrix data into file: {}'.format(save_path))

