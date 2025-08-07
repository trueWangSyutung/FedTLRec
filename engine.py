import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from fast_pytorch_kmeans import KMeans

from fedmodel import ServiceModel
from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace
import tqdm

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        self.server_model = ServiceModel(
             embedding_dim=self.config['latent_dim'],
             output_dim=self.config['latent_dim'],
             r=self.config['r'],
             user_num= self.config['num_users'],
             max_line=self.config['max_line'],
        )
        self.server_model_optimizer =  torch.optim.SGD(self.server_model.parameters(),
                                     lr=config['lr'])

        self.server_loss = torch.nn.MSELoss()
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.top_k = 10
        self.k_mean = KMeans(n_clusters=4, max_iter=100, tol=0.0001, verbose=1, mode='euclidean', init_method='random')
        self.round = 0
        self.clusters = None

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, user):
        """train a batch and return an updated model."""
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.lora_A'][user].data)
        optimizer, optimizer_u, optimizer_i = optimizers
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        regularization_term = compute_regularization(model_client, reg_item_embedding)
        loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        return model_client, loss.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        item_embedding = np.zeros((len(round_user_params), self.config['r'] * self.config['latent_dim']), dtype='float32')
        user_list = list(round_user_params.keys())
        for i, user in enumerate(user_list):
            item_embedding[i] = round_user_params[user]['embedding_item.lora_A'].numpy().flatten()
        print('item_embedding: {}'.format(item_embedding.shape))

        # 使用简化的方法将参数分为4部分，避免花费过多时间
        # 随机分配用户到4个组中
        np.random.seed(0)  # 保持结果的一致性
        groups = np.random.choice(4, len(user_list))

        # 分别处理每个组
        group_item_embeddings = []
        group_user_lists = []
        for i in range(4):
            group_indices = np.where(groups == i)[0]
            if len(group_indices) > 0:
                group_item_embedding = item_embedding[group_indices]
                group_item_embeddings.append(group_item_embedding)
                group_user_list = [user_list[idx] for idx in group_indices]
                group_user_lists.append(group_user_list)
                print(f'Group {i}: {len(group_indices)} users')

        # 为每个组分别训练服务模型
        self.server_model.train()
        group_outputs = []
        for i, group_item_embedding in enumerate(group_item_embeddings):
            for round in tqdm.tqdm(range(self.config['server_round'])):
                self.server_model_optimizer.zero_grad()
                output = self.server_model(group_item_embedding)

                # 只要 前 user_num 个 维度
                server_loss = self.server_loss(output, torch.from_numpy(group_item_embedding).float())
                server_loss.backward()
                self.server_model_optimizer.step()
            group_outputs.append(output.detach().numpy())

        self.server_model.eval()

        # 合并所有组的输出
        merged_output = np.zeros_like(item_embedding)
        for i, (group_item_embedding, group_user_list) in enumerate(zip(group_item_embeddings, group_user_lists)):
            group_output = group_outputs[i]
            for j, user in enumerate(group_user_list):
                user_index = user_list.index(user)
                merged_output[user_index] = group_output[j]

        # 将结果分配给用户
        for i, user in enumerate(user_list):
            o2 = torch.from_numpy(merged_output[i]).reshape(self.config['r'], self.config['latent_dim']).data
            self.server_model_param['embedding_item.lora_A'][user] = copy.deepcopy(o2)

    def aggregate_clients_params_k(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        item_embedding = np.zeros((len(round_user_params), self.config['r'] * self.config['latent_dim']), dtype='float32')
        user_list = list(round_user_params.keys())
        for i, user in enumerate(user_list):
            item_embedding[i] = round_user_params[user]['embedding_item.lora_A'].numpy().flatten()
        print('item_embedding: {}'.format(item_embedding.shape))
        # 聚类的时候 使用 每一个  用户 的 item_embedding 的均值，即维度 [user_num, 1]

        if self.round % 10 == 0:
            item_tensor = torch.tensor(item_embedding,device="cpu")
            self.clusters = self.k_mean.fit_predict(item_tensor)

        # 分别处理每个聚类
        cluster_item_embeddings = []
        cluster_user_lists = []
        for i in tqdm.tqdm(range(4)):
            cluster_indices = np.where(self.clusters == i)[0]
            if len(cluster_indices) > 0:
                cluster_item_embedding = item_embedding[cluster_indices]
                cluster_item_embeddings.append(cluster_item_embedding)
                cluster_user_list = [user_list[idx] for idx in cluster_indices]
                cluster_user_lists.append(cluster_user_list)
                print(f'Cluster {i}: {len(cluster_indices)} users')

        # 为每个聚类分别训练服务模型
        self.server_model.train()
        cluster_outputs = []
        for i, cluster_item_embedding in enumerate(cluster_item_embeddings):
            for round in tqdm.tqdm(range(self.config['server_round'])):
                self.server_model_optimizer.zero_grad()
                output = self.server_model(cluster_item_embedding)

                # 只要 前 user_num 个 维度
                server_loss = self.server_loss(output, torch.from_numpy(cluster_item_embedding).float())
                server_loss.backward()
                self.server_model_optimizer.step()
            cluster_outputs.append(output.detach().numpy())

        self.server_model.eval()

        # 合并所有聚类的输出
        merged_output = np.zeros_like(item_embedding)
        for i, (cluster_item_embedding, cluster_user_list) in enumerate(zip(cluster_item_embeddings, cluster_user_lists)):
            cluster_output = cluster_outputs[i]
            for j, user in enumerate(cluster_user_list):
                user_index = user_list.index(user)
                merged_output[user_index] = cluster_output[j]

        # 将结果分配给用户
        for i, user in enumerate(user_list):
            o2 = torch.from_numpy(merged_output[i]).reshape(self.config['r'], self.config['latent_dim']).data
            self.server_model_param['embedding_item.lora_A'][user] = copy.deepcopy(o2)

    def aggregate_clients_params_back(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        item_embedding = np.zeros((len(round_user_params), self.config['r'] * self.config['latent_dim']), dtype='float32')
        for user in round_user_params.keys():
            item_embedding[user] = round_user_params[user]['embedding_item.lora_A'].numpy().flatten()
        print('item_embedding: {}'.format(item_embedding.shape))
        # 开始训练 服务模型
        self.server_model.train()
        for round in tqdm.tqdm(range(self.config['server_round'])):
            self.server_model_optimizer.zero_grad()
            output = self.server_model(item_embedding)

            # 只要 前 user_num 个 维度
            server_loss = self.server_loss(output, torch.from_numpy(item_embedding).float())
            server_loss.backward()
            self.server_model_optimizer.step()
        self.server_model.eval()
        output = self.server_model(item_embedding)
        for user in round_user_params.keys():
            #             item_embedding[user] = round_user_params[user]['embedding_item.lora_A'].numpy().flatten()
            # output[user].reshape(self.config['r'], self.config['latent_dim'])
            o2 = output[user].reshape(self.config['r'], self.config['latent_dim']).data
            self.server_model_param['embedding_item.lora_A'][user] = copy.deepcopy(o2)
    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # 从 训练数据中 选取 offlin_rate * num_users 个用户，作为离线用户
        offline_user = random.sample(range(self.config['num_users']), int(self.config['num_users'] * self.config['offline_rate']))
        # store users' model parameters of current round.
        round_participant_params = {}
        self.round = round_id

        # initialize server parameters for the first round.
        if round_id == 0:
            self.server_model_param['embedding_item.lora_A'] = {}
            for user in participants:
                self.server_model_param['embedding_item.lora_A'][user] = copy.deepcopy(self.model.state_dict()['embedding_item.lora_A'].data.cpu())
            self.server_model_param['embedding_item.lora_A']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.lora_A'].data.cpu())
        for user in tqdm.tqdm(participants, desc='round {}'.format(round_id)):
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated user embedding and aggregated item embedding from server
            # and use local updated mlp parameters from last round.
            if round_id != 0:
                # for participated users, load local updated parameters.
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
                user_param_dict['embedding_item.lora_A'] = copy.deepcopy(self.server_model_param['embedding_item.lora_A'][user].data)
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating mlp parameters. attention_layers
            base =  [{"params": model_client.fc_layers.parameters()}, {"params": model_client.affine_output.parameters()}]
            if self.config['use_transfermer'] is True:
                base.append({"params": model_client.attention_layers.parameters()})
            optimizer = torch.optim.SGD(
                base,
                lr=self.config['lr'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding.
            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])  # User optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer
            optimizers = [optimizer, optimizer_u, optimizer_i]
            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)
            # print('[User {}]'.format(user))
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' user embedding using a dict.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            # del round_participant_params[user]['embedding_user.weight']
            round_participant_params[user] = {}
            if user in offline_user:
                # offline user embedding, r * latent_dim 全零
                round_participant_params[user]['embedding_item.lora_A'] = copy.deepcopy( torch.zeros(self.config['r'], self.config['latent_dim']))
            else:
                round_participant_params[user]['embedding_item.lora_A'] = copy.deepcopy(self.client_model_params[user]['embedding_item.lora_A'])
                if self.config['dp'] > 0:
                    round_participant_params[user]['embedding_item.lora_A'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.lora_A'].shape).sample()
        # aggregate client models in server side.
        if self.config['use_kmean'] is True:
            self.aggregate_clients_params_k(round_participant_params)
        else:
            self.aggregate_clients_params(round_participant_params)
        return participants

    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
            # user_param_dict['embedding_item.lora_A'] = copy.deepcopy(
            #     self.server_model_param['embedding_item.lora_A']['global'].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_user = negative_users[user * 99: (user + 1) * 99]
                negative_item = negative_items[user * 99: (user + 1) * 99]
                # perform model prediction.
                test_score = user_model(test_item)
                negative_score = user_model(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                # print(ratings_pred,ratings)
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
