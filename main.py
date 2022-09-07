from cmath import inf
import os
import numpy as np
import pandas as pd

import albumentations as A
import cv2
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
# import pretrainedmodels
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
import torchvision
from glob import glob
from torch.utils.data import Dataset,DataLoader
from model import VitModel
from dataset import GUIETrainDataSet,GUIEValDataSet,give_sop_datasets,split_database_dict
from config import CFG
import warnings  
import metrics
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# scheduler and optimizer
def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

# dataset and dataloader
data_transforms = {
    "train": A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        ], p=1.0),
    
    "valid": A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST)
        ], p=1.0)
}

# loss
criterion = torch.nn.CrossEntropyLoss()

# data preprocess
train_image_dict,val_image_dict = give_sop_datasets('/home/lijw/GUIE/data/Stanford_Online_Products')
database_image_dict,query_image_dict  = split_database_dict(val_image_dict)

config_ = {'batch_size':4000,'samples_per_class':4}
train_dataset = GUIETrainDataSet(train_image_dict,config_,transforms=data_transforms['train'])
database_dataset = GUIEValDataSet(database_image_dict,transforms=data_transforms['valid'])
query_dataset = GUIEValDataSet(query_image_dict,transforms=data_transforms['valid'])

trainloader = DataLoader(train_dataset,batch_size = CFG.train_bs,num_workers = 8,pin_memory=True,sampler=torch.utils.data.SequentialSampler(train_dataset))
databaseloader = DataLoader(database_dataset,batch_size = CFG.train_bs,num_workers = 8,shuffle=False,pin_memory=True)
queryloader = DataLoader(query_dataset,batch_size = CFG.train_bs,num_workers = 8,shuffle=False,pin_memory=True)

model = VitModel(in_features=1000)
model.to(CFG.device)

optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)

max_mP_at_5 = -inf

for i in range(CFG.epochs):
    # train
    model.train()
    trainloader.dataset.reshuffle()
    for idx,(images,labels) in enumerate(trainloader):
        images = images.to(torch.float32).to(CFG.device)
        labels = labels.to(torch.long).to(CFG.device)
        embed_features = model(images,labels) # batch size * CFG.embed
        output = torch.softmax(embed_features,dim=1)
        # output = embed_features
        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'step:{idx},loss:{loss}')

    # val
    model.eval()
    with torch.no_grad():
        database_target_labels, db_features = [],[]
        final_iter = tqdm(databaseloader, desc='Computing Database Embeded features...')
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[0].to(torch.float), inp[1].to(torch.long)
            database_target_labels.extend(target.numpy().tolist())
            out = model.feature_extractor(input_img.to(CFG.device))
            db_features.extend(out.cpu().detach().numpy().tolist())
        database_target_labels = np.hstack(database_target_labels).reshape(-1,1)
        db_features  = np.vstack(db_features).astype('float32')
        print(f'Database features nums : {len(db_features)}')

        q_target_labels, q_features = [],[]
        final_iter = tqdm(queryloader, desc='Computing query embeded features...')
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[0].to(torch.float), inp[1].to(torch.long)
            q_target_labels.extend(target.numpy().tolist())
            out = model.feature_extractor(input_img.to(CFG.device))
            q_features.extend(out.cpu().detach().numpy().tolist())
        q_target_labels = np.hstack(q_target_labels).reshape(-1,1)
        q_features  = np.vstack(q_features).astype('float32')
        print(f'Query features nums : {len(q_features)}')
        # compute distance
        # db_features : db_nums,embed nums
        # q_features  : q_nums ,embed_nums
        distance_list = []
        topK_index    = []
        for i in range(len(q_features)):
            temp_distance = []
            for j in range(len(db_features)):
                temp_distance.append(np.sum((q_features[i,:]-db_features[j])**2))
            topK_index.append([database_target_labels[i] for i in sorted(range(len(temp_distance)),key=lambda k: temp_distance[k])])
            distance_list.append(temp_distance)
        topK_index = np.array(topK_index).reshape(len(q_target_labels),len(database_target_labels))
        mP_at_5 = metrics.compute_mP_at_K(q_target_labels,topK_index)
        del topK_index
        del q_features
        del db_features
        if max_mP_at_5<=mP_at_5:
            print(f'Compute Completed. mP@5 improved from {max_mP_at_5} to {mP_at_5}...')
            max_mP_at_5 = mP_at_5
            torch.save(model,'best_model.pt')
        else:
            print(f'Compute Completed. mP@5 did not improved...')

        


        

