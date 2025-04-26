from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim, Tensor
from torch_sparse import matmul
from tqdm import tqdm
import numpy as np
import dataloader
import evaluator
import datetime
import warnings
import models
import torch
import utils
import os

warnings.filterwarnings("ignore")


def train(model, optimizer, di_train_edge_index, user_sparse_edge_index, train_pos_intactions, config):
    model.train()
    s = utils.negative_sampling(train_pos_intactions, di_train_edge_index, config['neg_cnt'])
    s = s.to(config['device'])
    users, pos_items, neg_items = s[:, 0], s[:, 1], s[:, 2:]
    neg_items = neg_items.reshape((neg_items.size()[0], config['neg_cnt']))

    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // config['batch_size'] + 1
    avg_loss = 0
    for (batch_i, (b_users, b_pos, b_neg)) in enumerate(
            utils.minibatch(config['batch_size'], users, pos_items, neg_items)):
        items_emb_final, users_emb_final = model(user_sparse_edge_index)
        loss = model.cal_loss(users_emb_final, items_emb_final, config['l2-reg'], 1e-4, config['neg_cnt'],
                              b_users, b_pos, b_neg)

        avg_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss /= total_batch
    return round(avg_loss.item(), 4)


def evaluate(model, di_test_edge_index, user_sparse_edge_index, exclude_edge_indices,
             test_pos_interactions, config,topk, tail_res):
    model.eval()
    neg_cnt = 1
    config['topk'] = topk
    s = utils.negative_sampling(test_pos_interactions, di_test_edge_index, neg_cnt)
    s = s.to(config['device'])
    users, pos_items, neg_items = s[:, 0], s[:, 1], s[:, 2]
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    neg_items = neg_items.reshape((neg_items.size()[0], neg_cnt))
    with torch.no_grad():
        item_embedding, user_embedding = model(user_sparse_edge_index)
        loss = model.cal_loss(user_embedding, item_embedding, config['l2-reg'], 1e-4, 1, users, pos_items, neg_items)

        f = torch.nn.Sigmoid()
        rating = f(torch.matmul(user_embedding, item_embedding.T))
        for exclude_edge_index in exclude_edge_indices:
            exclude_edge_index = exclude_edge_index.cpu().numpy()
            user_pos_items = dataloader.get_positive_intractions(exclude_edge_index)

            exclude_users = []
            exclude_items = []
            for user, items in enumerate(user_pos_items):
                exclude_users.extend([user] * len(items))
                exclude_items.extend(items)

            rating[exclude_users, exclude_items] = -(1 << 10)

        _, top_K_items = torch.topk(rating, k=config['topk'])

        users = list(range(0, num_users))

        top_K_items = top_K_items.cpu().numpy()
        r = [] 
        for user in users:
            label = list(map(lambda x: x in test_pos_interactions[user], top_K_items[user]))
            r.append(label)

        if len(exclude_edge_indices)==1: # validation
            new_r = [r[i] for i in val_on]
            new_test_pos_interactions = [test_pos_interactions[i] for i in val_on]
            new_users = list(range(0, len(val_on)))
            rating_ = rating.cpu().numpy()
            new_rating = torch.Tensor([rating_[i] for i in val_on])
            new_topk = np.array([top_K_items[i] for i in val_on])
 
        if len(exclude_edge_indices)==2: # test
            new_r = [r[i] for i in test_on]
            new_test_pos_interactions = [test_pos_interactions[i] for i in test_on]
            new_users = list(range(0, len(test_on)))
            rating_ = rating.cpu().numpy()
            new_rating = torch.Tensor([rating_[i] for i in test_on])
            new_topk = np.array([top_K_items[i] for i in test_on])

        rating = new_rating
        r = torch.Tensor(np.array(new_r).astype('float'))
        precision, recall = evaluator.PrecisionRecall_ATk(new_test_pos_interactions, r, config['topk'])
        ndcg = evaluator.NDCGatK_r(new_test_pos_interactions, r, config['topk'])
        if len(exclude_edge_indices) == 2:
            coverage = 0
            pru = evaluator.PRU_AtK(new_users, rating, new_test_pos_interactions, exclude_items)
            pri = evaluator.PRI_AtK(rating, new_test_pos_interactions, exclude_items)
            mrr = evaluator.MRR_AtK(r)
            ma_precision = evaluator.MAP_AtK(new_test_pos_interactions, r)
            novelty = 0
            eo_ = evaluator.EO_AtK(long_tails, popular_items, new_test_pos_interactions, new_topk)
        else:
            coverage = 0  
            pru = 0  
            pri = 0
            mrr = 0
            ma_precision = 0
            novelty = 0
            eo_ = 0

        splited_result = {}
        if tail_res and len(exclude_edge_indices) == 2:
            diff = [popular_items, long_tails]
            mark = ['unpop', 'pop']
            for i in range(2):
                temp_test_pos_interactions = {}
                for user in users:
                    items = list(set(test_pos_interactions[user]) - set(diff[i]))
                    if len(items) > 0:
                        temp_test_pos_interactions[user] = items
                r = []
                for user in list(temp_test_pos_interactions.keys()):
                    label = list(map(lambda x: x in temp_test_pos_interactions[user], top_K_items[user]))
                    r.append(label)
                r = torch.Tensor(np.array(r).astype('float'))
                precision_, recall_ = evaluator.PrecisionRecall_ATk(list(temp_test_pos_interactions.values()), r,
                                                          config['topk'])
                splited_result[f"{mark[i]}_Precision"] = precision_
                splited_result[f"{mark[i]}_Recall"] = recall_
                splited_result[f"{mark[i]}_NDCG"] = evaluator.NDCGatK_r(list(temp_test_pos_interactions.values()), r,
                                                              config['topk'])
                splited_result[f"{mark[i]}_MAP"] = evaluator.MAP_AtK(list(temp_test_pos_interactions.values()), r)
                splited_result[f"{mark[i]}_MRR"] = evaluator.MRR_AtK(r)

    return round(loss.item(),
                 4), precision, recall, ndcg, mrr, ma_precision, coverage, pru, pri, novelty, eo_, splited_result


if __name__ == "__main__":
    wo_val = True
    tail_result = False
    model_name = "mymodel"
    # data_path = 'data/bookcrossing'
    # data_path = 'data/cds'
    data_path = 'data/electronics'
    dataset = 'modified_data_20'
    config_path = os.path.join(os.path.dirname(__file__), 'configs')
    config = utils.read_config(config_path, model_name)
    config['i_i_edge_count'] = 0
    config['i_i_edge_type'] = "category_based"
    config['neg_cnt'] = 1
    config['seed'] = 2020
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(config['seed'])

    if wo_val:
        path = f'{data_path}/{dataset}'
    else:
        path = f'{data_path}/{dataset}'

    train_path = f'{path}/train.txt'
    test_path = f'{path}/test.txt'
    category_dic = {}
    categories = []
    with open(os.path.join(os.path.dirname(__file__), f'{data_path}/{dataset}/val_on.txt'), 'r') as f1, open(os.path.join(os.path.dirname(__file__), f'{data_path}/{dataset}/test_on.txt'), 'r') as f2:
        val_on = list(map(int, f1.read().split(' ')))
        test_on = list(map(int, f2.read().split(' ')))

    user_mapping, item_mapping = dataloader.load_node_csv(train_path)
    num_users, num_items = len(user_mapping), len(item_mapping)

    di_train_edge_index, undi_train_edge_index = dataloader.load_edge_csv(train_path, user_mapping, item_mapping)

    di_test_edge_index, undi_test_edge_index = dataloader.load_edge_csv(test_path, user_mapping, item_mapping)

    popular_items, long_tails, popularity, num_click = utils.get_items_popularity(di_train_edge_index)

    train_pos_items = dataloader.get_positive_intractions(di_train_edge_index)
    test_pos_items = dataloader.get_positive_intractions(di_test_edge_index)

    sprs_size = (num_items, num_items)

    sparse_size = (num_users + num_items, num_users + num_items)
    user_train_sparse_edge_index = utils.create_sparse_tensor(undi_train_edge_index[0], undi_train_edge_index[1],
                                                        sparse_size)
    user_test_sparse_edge_index = utils.create_sparse_tensor(undi_test_edge_index[0], undi_test_edge_index[1],
                                                       sparse_size)

    user_train_sparse_edge_index = user_train_sparse_edge_index.to(config['device'])
    user_test_sparse_edge_index = user_test_sparse_edge_index.to(config['device'])

    val_path = f'{data_path}/{dataset}/val.txt'
    di_val_edge_index, undi_val_edge_index = dataloader.load_edge_csv(val_path, user_mapping, item_mapping)
    val_pos_items = dataloader.get_positive_intractions(di_val_edge_index)
    val_sparse_edge_index = utils.create_sparse_tensor(undi_val_edge_index[0], undi_val_edge_index[1], sparse_size)
    val_sparse_edge_index = val_sparse_edge_index.to(config['device'])

    model = getattr(models, model_name)(num_users, num_items, config).to(config['device'])
    model = model.to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    current_datetime = datetime.datetime.now()
    current_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_filename = f"{model.name}_{dataset}_batch_{config['batch_size']}_lr_{config['lr']}_neg_" \
                          f"{config['neg_cnt']}_ii_edge_{config['i_i_edge_count']}_topk_{config['topk']}" \
                          f"_lmbd_{config['l2-reg']}"
    tensorboard_filename = f"{model.name}_batch_{config['batch_size']}_lr_{config['lr']}_neg_{config['neg_cnt']}" \
                           f"_ii_edge_{config['i_i_edge_count']}_topk_{config['topk']}_lmbd_{config['l2-reg']}"
    
    save_file_20 = f"./checkpoints/{checkpoint_filename}_date_{current_datetime}_ndcg_best_model_20.pth.tar"
    save_file_50 = f"./checkpoints/{checkpoint_filename}_date_{current_datetime}_ndcg_best_model_50.pth.tar"
    save_file_100 = f"./checkpoints/{checkpoint_filename}_date_{current_datetime}_ndcg_best_model_100.pth.tar"
    save_file_300 = f"./checkpoints/{checkpoint_filename}_date_{current_datetime}_ndcg_best_model_300.pth.tar"

    monitor_20 = utils.Monitor(model, optimizer, 'higher', patience=20, filename=save_file_20)
    monitor_50 = utils.Monitor(model, optimizer, 'higher', patience=20, filename=save_file_50)
    monitor_100 = utils.Monitor(model, optimizer, 'higher', patience=20, filename=save_file_100)
    monitor_300 = utils.Monitor(model, optimizer, 'higher', patience=20, filename=save_file_300)
    monitors = [monitor_20, monitor_50, monitor_100, monitor_300]
    monitor_dict = {20:monitor_20, 50:monitor_50, 100:monitor_100, 300:monitor_300}

    for epoch in tqdm(range(1, config['epochs'] + 1)):
        train_loss = train(model, optimizer, di_train_edge_index,
                           user_train_sparse_edge_index, train_pos_items, config)
        print(f'EPOCH {epoch}/{config["epochs"]} loss: {train_loss}')
        if epoch % 1 == 0:
            model.eval()
            topk_list = list(monitor_dict.keys())
            if len(topk_list) ==0:
                break

            for tpk in topk_list:
                val_loss, precision, recall, ndcg, mrr, ma_p, coverage, pru, pri, nov, eo, pop_unpop_metrics = \
                    evaluate(model, di_val_edge_index, user_train_sparse_edge_index,[di_train_edge_index], 
                            val_pos_items, config, tpk, tail_result)
                
                print(f"Eval {tpk}: Loss: {val_loss}, Precision:{precision}, Recall:{recall}, NDCG: {ndcg}, "
                    f"MRR: {mrr}, MAP: {ma_p}, Coverage:{coverage}, Novelty:{nov}, EO:{eo}, PRU:{pru}, PRI:{pri}")

                if not monitor_dict[tpk].coninue_check(epoch, ndcg, val_loss=val_loss, precision=precision, recall=recall, ndcg=ndcg,
                                            MRR=mrr, MAP=ma_p, coverage=coverage, Novelty=nov, EO=eo, PRU=pru, PRI=pri,
                                            Split=pop_unpop_metrics):
                    print("Training Stopped due to early stop mechanism ", tpk)
                    monitor_dict[tpk].save_best_model()
                    monitor_dict.pop(tpk)

    # Test
    print("\nTest:\n")
    monitor_dict = {20:monitor_20, 50:monitor_50, 100:monitor_100, 300:monitor_300}
    topk_list = list(monitor_dict.keys())
    for tpk in topk_list:
        print(f'Top {tpk}:\n')
        model = globals()[model_name](num_users, num_items, config).to(config['device'])
        model.load_state_dict(monitor_dict[tpk].load_best_model())
        
        val_loss, precision, recall, ndcg, mrr, ma_p, coverage, pru, pri, nov, eo, pop_unpop_metrics = \
            evaluate(model, di_test_edge_index, user_train_sparse_edge_index, [di_train_edge_index, di_val_edge_index],
                    test_pos_items, config, tpk, tail_result)

        print(f"Eval: Loss: {val_loss}, Precision:{precision}, Recall:{recall}, NDCG: {ndcg}, "
            f"MRR: {mrr}, MAP: {ma_p}, Coverage:{coverage}, Novelty:{nov}, EO:{eo} PRU:{pru}, PRI:{pri}")
