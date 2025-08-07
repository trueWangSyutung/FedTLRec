import json

import pandas as pd
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import argparse
from mlp import MLPEngine
from data import SampleGenerator
from utils import *


# Training settings
parser = argparse.ArgumentParser()

# cora
# 尝试不同的LoRA配置
parser.add_argument('--r', type=int, default=16)  # 增加r值从8到16
parser.add_argument('--lora_alpha', type=int, default=32)  # 增加alpha值从16到32
parser.add_argument('--server_round', type=int, default=10)  # 增加服务端训练轮数
parser.add_argument('--offline_rate', type=float, default=0.1)
parser.add_argument('--server_lr', type=float, default=1e-3)  # 降低服务端学习率
parser.add_argument('--use_transfermer', type=bool, default=False)  # 启用Transformer
parser.add_argument('--use_kan_transfermer', type=bool, default=False)
parser.add_argument('--function', type=str, default="relu")

parser.add_argument('--use_kmean', type=bool, default=True)  # 启用K-means聚类
parser.add_argument('--use_kan', type=bool, default=False)
parser.add_argument('--transfermer_block_num', type=int, default=2)  # 增加Transformer块数量

parser.add_argument('--full_train', type=bool, default=False)
# user_num,max_line
parser.add_argument('--max_line', type=int, default=128)


parser.add_argument('--alias', type=str, default='fedgraph')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=50)  # 增加联邦训练轮数
parser.add_argument('--local_epoch', type=int, default=1)  # 增加本地训练轮数

parser.add_argument('--neighborhood_size', type=int, default=0)
parser.add_argument('--neighborhood_threshold', type=float, default=1.)
parser.add_argument('--mp_layers', type=int, default=1)
parser.add_argument('--similarity_metric', type=str, default='cosine')
parser.add_argument('--reg', type=float, default=0.1)  # 降低正则化参数
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=128)  # 调整批次大小
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-3)  # 降低学习率
parser.add_argument('--dataset', type=str, default='100k', help='ml-1m, lastfm-2k, amazon, filmtrust, 100k')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items', type=int)
# 增加网络层数和宽度
parser.add_argument('--latent_dim', type=int, default=16)  # 增加潜在维度
parser.add_argument('--num_negative', type=int, default=8)  # 增加负样本数量
parser.add_argument('--layers', type=str, default='32,16,8')  # 增加层数和宽度
parser.add_argument('--dp', type=float, default=0.1)
parser.add_argument('--use_cuda', type=bool, default=False)  # 启用CUDA
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--model_dir', type=str, default='checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model')
args = parser.parse_args()

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)
def train(args):
    # Model.
    config = vars(args)
    # config['layers'] 不是 list
    if not isinstance(config['layers'], list):
        if len(config['layers']) > 1:
            config['layers'] = [int(item) for item in config['layers'].split(',')]
        else:
            config['layers'] = int(config['layers'])
    if config['dataset'] == 'ml-1m':
        config['num_users'] = 6040
        config['num_items'] = 3706
    elif config['dataset'] == '100k':
        config['num_users'] = 943
        config['num_items'] = 1682
    elif config['dataset'] == 'lastfm-2k':
        config['num_users'] = 1600
        config['num_items'] = 12454
    elif config['dataset'] == 'amazon':
        config['num_users'] = 8072
        config['num_items'] = 11830
    elif config['dataset'] == "filmtrust":
        config['num_users'] = 1002
        config['num_items'] = 2042

        pass
    engine = MLPEngine(config)



    # Load Data
    dataset_dir = "data/" + config['dataset'] + "/" + "ratings.dat"
    if config['dataset'] == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "100k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "lastfm-2k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "amazon":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif config['dataset'] == "filmtrust":
        rating = pd.read_csv(dataset_dir, sep=" ", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    else:
        pass
    # Reindex
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=rating)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data

    hit_ratio_list = []
    ndcg_list = []
    val_hr_list = []
    val_ndcg_list = []
    train_loss_list = []
    test_loss_list = []
    val_loss_list = []
    best_val_hr = 0
    final_test_round = 0
    for round in range(config['num_round']):
        # break
        logging.info('-' * 80)
        logging.info('Round {} starts !'.format(round))

        all_train_data = sample_generator.store_all_train_data(config['num_negative'])
        logging.info('-' * 80)
        logging.info('Training phase!')
        tr_loss = engine.fed_train_a_round(all_train_data, round_id=round)
        # break
        train_loss_list.append(tr_loss)

        logging.info('-' * 80)
        logging.info('Testing phase!')
        hit_ratio, ndcg, te_loss = engine.fed_evaluate(test_data)
        test_loss_list.append(te_loss)
        # break
        logging.info('[Testing Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, hit_ratio, ndcg))
        hit_ratio_list.append(hit_ratio)
        ndcg_list.append(ndcg)

        logging.info('-' * 80)
        logging.info('Validating phase!')
        val_hit_ratio, val_ndcg, v_loss = engine.fed_evaluate(validate_data)
        val_loss_list.append(v_loss)
        logging.info(
            '[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round, val_hit_ratio, val_ndcg))
        val_hr_list.append(val_hit_ratio)
        val_ndcg_list.append(val_ndcg)

        if val_hit_ratio >= best_val_hr:
            best_val_hr = val_hit_ratio
            final_test_round = round

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    json_obj = {
        'config' : config,
        'hit_list': hit_ratio_list,
        'ndcg_list': ndcg_list,
        'val_hit_list': val_hr_list,
        'val_ndcg_list': val_ndcg_list,

        'best_val_hr': best_val_hr,
        'best_val_ndcg' : val_ndcg_list[val_hr_list.index(best_val_hr)],
        'final_test_round': final_test_round,
        'final_test_hr': hit_ratio_list[final_test_round],
        'final_test_ndcg': ndcg_list[final_test_round],
    }
    file_name = "sh_result/"+'-'+config['dataset']+".txt"
    with open(file_name, 'a') as file:
        file.write( json.dumps(json_obj) + '\n' )

    logging.info('fedgraph')
    logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr: {}, dataset: {}, layers: {}, negatives: {}, '
                 'neighborhood_size: {}, neighborhood_threshold: {}, mp_layers: {}, similarity_metric: {}'.
                 format(config['clients_sample_ratio'], config['lr_eta'], config['batch_size'], config['lr'],
                        config['dataset'], config['layers'], config['num_negative'], config['neighborhood_size'],
                        config['neighborhood_threshold'], config['mp_layers'], config['similarity_metric']))

    logging.info('hit_list: {}'.format(hit_ratio_list))
    logging.info('ndcg_list: {}'.format(ndcg_list))
    logging.info('Best test hr: {}, ndcg: {} at round {}'.format(hit_ratio_list[final_test_round],
                                                                 ndcg_list[final_test_round],
                                                                 final_test_round))


times = 10
for i in range(times):
    train(args)