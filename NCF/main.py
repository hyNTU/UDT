import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, movielens',
	default = 'yelp')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--seed', 
	type = int,
	help = 'seed for reproducibility',
	default = 2019)
parser.add_argument("--gpu", 
	type=str,
	default="0",
	help="gpu card ID")
parser.add_argument("--epoch_eval", 
    type = int,
	default=10,
	help="epoch to start evaluation")
parser.add_argument("--top_k", 
    type = list,
	default= [50, 100],
	help="compute metric @topk")
parser.add_argument("--batch_size", 
    type = int,
	default= 2048,
	help="batch size")
parser.add_argument("--temp1", 
    type = float,
	default= 0.2,
	help="temp 1")
parser.add_argument("--temp2", 
    type = float,
	default= 0.5,
	help="temp 2")
parser.add_argument("--userfact1", 
    type = float,
	default= 0.5,
	help="user factor 1")
parser.add_argument("--userfact2", 
    type = float,
	default= 0.0,
	help="user factor 2")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
np.random.seed(args.seed) #numpy
random.seed(args.seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


data_path = f'../data/{args.dataset}/'
model_path = f'./models/'

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, valid_pos, user_pos, user_num ,item_num, train_mat, train_valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, 1, 0, train_data_noisy)
valid_dataset = data_utils.NCFData(
		valid_data, item_num, train_mat, 1, 1)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=2048, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

########################### CREATE MODEL #################################

model = model.NCF(user_num, item_num, 32, 3, 
						0.0, f'{args.model}', None, None)

model.cuda()
BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_mat_dense = torch.tensor(train_mat.toarray()).cpu()
train_mat_sum = train_mat_dense.sum(1)



########################### Temp Softmax #####################################
def temperature_scaled_softmax(logits,temperature):
    logits = logits / torch.unsqueeze(temperature,1)
    return torch.softmax(logits, dim=1)
    
########################### Eval #####################################
def eval(model, valid_pos, mat, best_recall, count):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15,users_in_valid):
        users_valid = list(users_valid)
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        users_valid_torch = torch.tensor(users_valid).repeat_interleave(item_num).cuda()  
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_valid)).cuda() 
        prediction = model(users_valid_torch, items_full)
        _, indices = torch.topk(prediction.view(len(users_valid),-1)+mat[users_valid].cuda()*-9999, max(top_k))
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    epoch_recall = recall[0]

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")

    if epoch_recall > best_recall:
        best_recall = epoch_recall
        count = 0
        torch.save(model.state_dict(), model_path+f'{args.model}_{args.dataset}_{args.temp1}_{args.temp2}_{args.userfact1}_{args.userfact2}.pth')
    else: 
        count += 1
    return best_recall, count

########################### Test #####################################
def test(model, test_data_pos, mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_test = list(test_data_pos.keys())
    for users_test in toolz.partition_all(15,users_in_test):
        users_test = list(users_test)
        GroundTruth.extend([test_data_pos[u] for u in users_test])
        users_test_torch = torch.tensor(users_test).repeat_interleave(item_num).cuda()  
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_test)).cuda()
        prediction = model(users_test_torch, items_full)
        _, indices = torch.topk(prediction.view(len(users_test),-1)+mat[users_test].cuda()*-9999, max(top_k))
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)


    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))




    ########################### TRAINING #####################################
count, best_hr = 0, 0
best_recall = 0.0
top_k = args.top_k
user_temperature = torch.linspace(args.temp1, args.temp2, steps=user_num)
user_factor = torch.linspace(args.userfact1, args.userfact2, steps=user_num)
for epoch in range(1000):
    model.train() # Enable dropout (if have).
    train_loss = 0

    start_time = time.time()
    train_loader.dataset.ng_sample() # negative sampling is done here

    loss_mat = float('inf') - torch.zeros_like(train_mat_dense)

    user_loss = torch.zeros_like(train_mat_sum)
    

    for user, item, label, _ in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        # for user interaction level

        batch_pos_user = user[label > 0.]
        batch_pos_item = item[label > 0.]

        model.zero_grad()
        prediction = model(user, item)
        loss = BCE_loss(prediction, label)

        # for user level
        user_loss[user.cpu()] += loss.cpu()
        
        
        with torch.no_grad():
            if epoch <= 1:
                mul_factor = torch.ones_like(loss)
            else: 
                mul_factor = ui_factor[user.cpu(), item.cpu()].cuda()

        # for updating user iteraction level
        loss_ui = loss[label > 0.].cpu()
        loss_mat[batch_pos_user.cpu(), batch_pos_item.cpu()] = loss_ui

        batch_loss = torch.mean(mul_factor * loss)


        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss

    print("epoch: {}, loss:{}".format(epoch,train_loss))
    
    if epoch >= args.epoch_eval:
        best_recall, count = eval(model, valid_pos, train_mat_dense, best_recall, count)
    model.train()
    if count == 10:
        break

   
        
    all_user_loss = user_loss/train_mat_sum # avg of different number of interactions
    sorted_users = torch.argsort(all_user_loss) # users with lowest loss to highest loss

    temp_fact = torch.zeros_like(user_temperature)
    temp_fact[sorted_users] = user_temperature


    user_fact = torch.zeros_like(user_temperature)
    user_fact[sorted_users] = user_factor
    
    # ui level
    ui_factor = temperature_scaled_softmax(-loss_mat, temp_fact) * torch.unsqueeze(train_mat_sum,1) #multiply to make it even
    ui_factor[ui_factor==0] = 1.0


    ui_factor *=  torch.unsqueeze(user_fact,1)

print("############################## Training End. ##############################")
model.load_state_dict(torch.load(model_path+f'{args.model}_{args.dataset}_{args.temp1}_{args.temp2}_{args.userfact1}_{args.userfact2}.pth'))
model.cuda()

    ########################### Logs #####################################

train_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu() 
test(model, test_data_pos, train_mat_dense)
