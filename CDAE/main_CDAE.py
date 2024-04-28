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
import torch.nn.functional as F

import copy

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'yelp')
parser.add_argument('--seed', 
	type = int,
	help = 'seed for reproducibility',
	default = 2019)
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
parser.add_argument("--epoch_eval", 
    type = int,
	default=10,
	help="epoch to start evaluation")
parser.add_argument("--batch_size", 
    type = int,
	default=2048,
	help="epoch to start evaluation")
parser.add_argument("--top_k", 
    type = list,
	default= [50, 100],
	help="compute metrics @k")
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

train_data, valid_data, train_data_pos, valid_data_pos, test_data_pos, user_pos, user_num ,item_num, train_mat, valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)



train_mat_dense = train_mat.toarray()
users_list = np.array([i for i in range(user_num)])
train_dataset = data_utils.DenseMatrixUsers(users_list ,train_mat_dense)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

valid_mat_dense = valid_mat.toarray()
valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
valid_loader = data.DataLoader(valid_dataset, batch_size=4096, shuffle=True)

########################### CREATE MODEL #################################

model = model.CDAE(user_num, item_num, 32, 0.2)
model.cuda()
BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
num_ns = 1 # negative samples
optimizer = optim.Adam(model.parameters(), lr=0.001)


########################### Temp Softmax #####################################
def temperature_scaled_softmax(logits,temperature):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)

########################### Eval #####################################
def eval(model, valid_loader, valid_data_pos, train_mat, best_recall, count):
    top_k = args.top_k
    model.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(user_num, top_k[-1], dtype=torch.long) # predictions
    GroundTruth = list(valid_data_pos.values()) # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda()
            prediction = model(user_valid, prediction_input_from_train) # prediction of the batch from train matrix
            valid_data_mask = train_mat[user_valid.cpu()] * -9999# depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()

    predictedIndices = predictedIndices_all[list(valid_data_pos.keys())]
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print(f"Recall:{recall} NDCG: {NDCG}")
    if recall[0] > best_recall:
        best_recall = recall[0]
        count = 0
        torch.save(model.state_dict(), model_path+f'CDAE_{args.dataset}_{args.temp1}_{args.temp2}_{args.userfact1}_{args.userfact2}.pth')
    else: 
        count += 1
    return best_recall, count
    

########################### Test #####################################
def test(model, test_data_pos, train_mat, valid_mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = [] # predictions
    GroundTruth = list(test_data_pos.values())

    for users in toolz.partition_all(1000, list(test_data_pos.keys())): # looping through users in test set
        user_id = torch.tensor(list(users)).cuda()
        data_value_test = torch.tensor(train_mat[list(users)]).cuda()
        predictions = model(user_id, data_value_test) # model prediction for given data
        test_data_mask = (train_mat_dense[list(users)] + valid_mat[list(users)]) * -9999

        predictions = predictions + torch.tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(predictions, top_k[-1]) # returns sorted index based on highest probability
        indices = indices.cpu().numpy().tolist()
        predictedIndices += indices # a list of top 100 predicted indices

    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))



########################### Training #####################################
top_k = args.top_k
all_user_loss = [float(0) for _ in range(user_num)] # user level loss
careless_users = []
best_recall = 0
count = 0 

user_temperature = np.linspace(args.temp1, args.temp2, user_num)
user_factor = np.linspace(args.userfact1, args.userfact2, user_num)

for epoch in range(1000):
    model.train()
    train_loss = 0

    for user, data_value in train_loader:
        user = user.cuda()
        data_value = data_value.cuda()
        prediction = model(user, data_value)
        #negative sampling
        with torch.no_grad():
            num_ns_per_user = data_value.sum(1) * num_ns
            negative_samples = []
            users = []
            for u in range(data_value.size(0)):
                batch_interaction = torch.randint(0, item_num, (int(num_ns_per_user[u].item()),))
                negative_samples.append(batch_interaction)
                users.extend([u] * int(num_ns_per_user[u].item()))


        negative_samples = torch.cat(negative_samples, 0)
        users = torch.LongTensor(users)
        mask = data_value.clone()
        mask[users, negative_samples] = 1
        groundtruth = data_value[mask > 0.]
        pred = prediction [mask > 0.]

        loss = BCE_loss(pred, groundtruth)
        # section the loss base on mask sum
        split_loss = torch.split(loss,mask.sum(1, dtype=torch.int).tolist(), dim=0)
        split_groundtruth = torch.split(groundtruth,mask.sum(1, dtype=torch.int).tolist(), dim=0)

        with torch.no_grad():
            for i , user_id in enumerate(user):
                # user level loss
                user_loss = torch.mean(split_loss[i], dim = 0)
                all_user_loss[user_id] = user_loss.item()

                # per user interaction loss
                if epoch <= 1:
                    ui_factor = torch.ones_like(split_loss[i])

                else:                    
                    temp = user_temperature[np.where(sorted_users==user[i].item())[0][0]]

                    fact = user_factor[np.where(sorted_users==user[i].item())[0][0]] # user level factor
                    ui_factor = torch.ones_like(split_loss[i])
                    ui_factor[split_groundtruth[i]==1] = temperature_scaled_softmax(-split_loss[i][split_groundtruth[i] == 1], temp) * mask[i].sum() / 2.0 

                    ui_factor *= fact

                if i == 0:
                    ui_loss_factor = ui_factor
                else:
                    ui_loss_factor = torch.cat((ui_loss_factor, ui_factor))

        
        ui_loss_factor = ui_loss_factor.cuda()

        # calculating loss

        batch_loss = torch.mean(ui_loss_factor * loss)
        

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss

    user_loss_np = np.array(all_user_loss)
    sorted_users = np.argsort(user_loss_np) # lowest to highest loss

    
    print(f"Epoch: {epoch} Train loss: {train_loss}") 
    if epoch%20==0 or epoch >=args.epoch_eval:
        # validation
        best_recall, count = eval(model, valid_loader, valid_data_pos, train_mat_dense, best_recall, count)

        if count == 10:
            break

print("############################## Training End. ##############################")
model.load_state_dict(torch.load(model_path+f'CDAE_{args.dataset}_{args.temp1}_{args.temp2}_{args.userfact1}_{args.userfact2}.pth'))
model.cuda()

test(model, test_data_pos, train_mat_dense, valid_mat_dense)
