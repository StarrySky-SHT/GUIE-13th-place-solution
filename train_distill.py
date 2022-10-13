from cmath import inf
from distutils.command.config import config
import os
import numpy as np
import pandas as pd
import albumentations as A
import cv2
import torch
import torch.nn as nn
from tfrecord.torch.dataset import MultiTFRecordDataset
import torch.optim as optim
# import pretrainedmodels
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
import torchvision
from glob import glob
from torch.utils.data import Dataset,DataLoader
from model import CreateModel, CreateDolgModel,CreateClipRNModel,CreateClipVitModel,CreateClipVitDyMarginModel,CreateClipOpenAiModel
from dataset import get_dataset,give_sop_datasets,split_database_dict,fetch_scheduler
from config import CFG
import warnings  
import metrics
from torch.cuda import amp
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import faiss
import logging
import torch.nn.functional as F
from losses import RKdAngle,RkdDistance

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]- %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
set_seed(42)
# loss
dist_criterion = RkdDistance()
angle_criterion = RKdAngle()

#dataset and dataloader
train_dataset = get_dataset(dataType='merge_train')[-1]
database_dataset = get_dataset(dataType='gldv2_database')[-1]
query_dataset = get_dataset(dataType='gldv2_query')[-1]
trainloader = DataLoader(train_dataset ,batch_size = CFG.train_bs,num_workers = 8,pin_memory=True,shuffle=True)

databaseloader = DataLoader(database_dataset ,batch_size = CFG.train_bs,num_workers = 8,pin_memory=True,shuffle=False)
queryloader = DataLoader(query_dataset ,batch_size = CFG.train_bs,num_workers = 8,pin_memory=True,shuffle=False)

#create model
teacher_model = CreateClipOpenAiModel(in_features=1024)
teacher_model.load_state_dict(torch.load('/root/autodl-tmp/trained_models/clip_vitH_14_openai_newData_state_dict.pt',map_location=CFG.device))
teacher_model.cuda().float()

student_model = CreateClipOpenAiModel(in_features=1024)
student_model.bn = nn.Sequential(nn.BatchNorm1d(CFG.embed),
                                 nn.Linear(CFG.embed,64),
                                 nn.BatchNorm1d(64))
student_model.cuda().float()

optimizer = optim.Adam(student_model.parameters(),lr=CFG.lr,weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer,len(trainloader))
# train
max_mP_at_5 = -inf
logger = get_logger('/root/GUIE/GUIE/logs/exp_distill_clipVit_H14_newData.log')
for i in range(CFG.epochs):
    # train
    student_model.train()
    scaler = amp.GradScaler()
    for idx,(images,labels) in enumerate(trainloader):
        images = images.to(torch.float32).to(CFG.device)
        labels = labels.to(torch.long).to(CFG.device)

        with torch.no_grad():
            t_features = teacher_model.feature_extractor(images)

        with amp.autocast(enabled=True):
            s_features = student_model.feature_extractor(images)
            loss   = dist_criterion(s_features,t_features)+2*angle_criterion(s_features,t_features)
            loss   = loss / CFG.n_accumulate
        
        scaler.scale(loss).backward()
        
        if (idx + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.info('epoch:[{}/{}]  step:[{}/{}]  lr={:.7f}  loss={:.5f}'.format(i,CFG.epochs,idx , len(trainloader), lr, loss))


    #val
    student_model.eval()
    with torch.no_grad():
        database_target_labels, db_features = [],[]
        final_iter = tqdm(databaseloader, desc='Computing Database Embeded features...')
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[0].to(torch.float), inp[1].to(torch.long)
            database_target_labels.extend(target.numpy().tolist())
            out = F.normalize(student_model.feature_extractor(input_img.to(CFG.device)))
            db_features.extend(out.cpu().detach().numpy().tolist())
        database_target_labels = np.hstack(database_target_labels).reshape(-1,1)
        db_features  = np.vstack(db_features).astype('float32')
        print(f'Database features nums : {len(db_features)}')

        q_target_labels, q_features = [],[]
        final_iter = tqdm(queryloader, desc='Computing query embeded features...')
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[0].to(torch.float), inp[1].to(torch.long)
            q_target_labels.extend(target.numpy().tolist())
            out = F.normalize(student_model.feature_extractor(input_img.to(CFG.device)))
            q_features.extend(out.cpu().detach().numpy().tolist())
        q_target_labels = np.hstack(q_target_labels).reshape(-1,1)
        q_features  = np.vstack(q_features).astype('float32')
        print(f'Query features nums : {len(q_features)}')
        # compute distance
        # db_features : db_nums,embed nums
        # q_features  : q_nums ,embed_nums
        index = faiss.IndexFlatL2(64)
        index.add(db_features)
        D,I = index.search(q_features,k=5)
        topK_Class = database_target_labels[I].reshape(I.shape)
        mP_at_5 = metrics.compute_mP_at_K(q_target_labels,database_target_labels,topK_Class)
        del q_features
        del db_features
        if max_mP_at_5<=mP_at_5:
            logger.info('Compute Completed. mP@5 improved from {} to {}...'.format(max_mP_at_5 ,mP_at_5))
            max_mP_at_5 = mP_at_5
        else:
            print(f'Compute Completed. mP@5 did not improved...')
    torch.save(student_model.state_dict(),'/root/autodl-tmp/trained_models/distill_clipVit_H14_newData.pt')