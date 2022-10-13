from sklearn import datasets
import torch
import cv2
import os
import random
import copy
from config import CFG
import pandas as pd
import albumentations as A
import torchvision
from tfrecord.torch.dataset import MultiTFRecordDataset
from glob import glob
from torch.optim import lr_scheduler
import numpy as np
import open_clip
from torchvision import transforms
flatten = lambda l: [item for sublist in l for item in sublist]

# scheduler and optimizer
def fetch_scheduler(optimizer,steps_per_epoch):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.epochs*steps_per_epoch/CFG.n_accumulate, 
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
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

# dataset and dataloader
data_transforms = {
    "train": A.Compose([
        A.RandomResizedCrop(*CFG.img_size,scale=(0.9, 1.0), ratio=(0.75, 1.3333),interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.3),
        A.Cutout(max_h_size=int(CFG.img_size[0] * 0.4), max_w_size=int(CFG.img_size[1] * 0.4), num_holes=1, p=0.5),
        A.HorizontalFlip(p=0.5)
        ], p=1.0),
    
    "valid": A.Compose([
        A.RandomResizedCrop(*CFG.img_size,scale=(0.9, 1.0), ratio=(0.75, 1.3333),interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ], p=1.0)
}
# data_transforms_336 = {
#     "train": A.Compose([
#         A.RandomResizedCrop(336,336,scale=(0.9, 1.0), ratio=(0.75, 1.3333),interpolation=cv2.INTER_CUBIC),
#         A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
#         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.3),
#         A.Cutout(max_h_size=int(CFG.img_size[0] * 0.4), max_w_size=int(CFG.img_size[1] * 0.4), num_holes=1, p=0.5),
#         A.HorizontalFlip(p=0.5)
#         ], p=1.0),
    
#     "valid": A.Compose([
#         A.RandomResizedCrop(*CFG.img_size,scale=(0.9, 1.0), ratio=(0.75, 1.3333),interpolation=cv2.INTER_CUBIC),
#         A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
#         ], p=1.0)
# }


def get_cls_dataset(dataType='merge_train'):
    if dataType == 'merge_train':
        merge_list = ['gldv2','deepfashion','products10k']
        df_merge = pd.DataFrame()
        id_accumulate = 0
        for i in range(len(merge_list)):
            df,df_rootPath = get_cls_dataset(merge_list[i])[0:2]
            df['image_path'] = df.image_path.apply(lambda x:df_rootPath+x)
            df['id'] = df.id.apply(lambda x:x+id_accumulate)
            id_accumulate += df['id'].max() + 1
            df_merge = pd.concat([df_merge,df])
        merge_dataset = DFClsDataset(df_merge,'',data_transforms['train'])
        return None,None,merge_dataset
    elif dataType == 'merge_val':
        merge_list = ['gldv2_query','deepfashion_query']
        df_merge = pd.DataFrame()
        id_accumulate = 0
        for i in range(len(merge_list)):
            df,df_rootPath = get_cls_dataset(merge_list[i])[0:2]
            df['image_path'] = df.image_path.apply(lambda x:df_rootPath+x)
            df['id'] = df.id.apply(lambda x:x+id_accumulate)
            id_accumulate += df['id'].max() + 1
            df_merge = pd.concat([df_merge,df])
        merge_dataset = DFClsDataset(df_merge,'',data_transforms['train'])
        return None,None,merge_dataset
    
    elif dataType == 'products10k':
        df = pd.read_csv(CFG.products10k_dfPath)
        train_dataset = DFClsDataset(df,CFG.products10k_rootPath,data_transforms['train'])
        return df,CFG.products10k_rootPath,train_dataset
    elif dataType == 'gldv2':
        df = pd.read_csv(CFG.gldv2_dfPath)
        train_dataset = DFClsDataset(df,CFG.gldv2_rootPath,data_transforms['train'])
        return df,CFG.gldv2_rootPath,train_dataset
    elif dataType == 'deepfashion':
        df = pd.read_csv(CFG.deepfashion_dfPath)
        train_dataset = DFClsDataset(df,CFG.deepfashion_rootPath,data_transforms['train'])
        return df,CFG.deepfashion_rootPath,train_dataset

    elif dataType == 'products10k_val':
        df = pd.read_csv(CFG.products10k_val_dfPath)
        train_dataset = DFClsDataset(df,CFG.products10k_val_rootPath,data_transforms['valid'])
        return df,CFG.products10k_rootPath,train_dataset
    elif dataType == 'gldv2_query':
        df = pd.read_csv(CFG.gldv2_query_dfPath)
        query_dataset = DFClsDataset(df,CFG.gldv2_query_rootPath,data_transforms['valid'])
        return df,CFG.gldv2_query_rootPath,query_dataset
    elif dataType == 'deepfashion_query':
        df = pd.read_csv(CFG.deepfashion_query_dfPath)
        query_dataset = DFClsDataset(df,CFG.deepfashion_query_rootPath,data_transforms['valid'])
        return df,CFG.deepfashion_query_rootPath,query_dataset
    else:
        raise Exception("Invalid dataset type")


def get_dataset(dataType='gldv2'):
    if dataType == 'gldv2':
        df = pd.read_csv(CFG.gldv2_dfPath)
        train_dataset = DFDataset(df,CFG.gldv2_rootPath,data_transforms['train'])
        return df,CFG.gldv2_rootPath,train_dataset
    elif dataType == 'deepfashion':
        df = pd.read_csv(CFG.deepfashion_dfPath)
        train_dataset = DFDataset(df,CFG.deepfashion_rootPath,data_transforms['train'])
        return df,CFG.deepfashion_rootPath,train_dataset
    elif dataType == 'products10k':
        df = pd.read_csv(CFG.products10k_dfPath)
        train_dataset = DFDataset(df,CFG.products10k_rootPath,data_transforms['train'])
        return df,CFG.products10k_rootPath,train_dataset
    elif dataType == 'gldv2_subset':
        df = pd.read_csv(CFG.gldv2_subset_dfPath)
        train_dataset = DFDataset(df,CFG.gldv2_subset_rootPath,data_transforms['train'])
        return df,CFG.gldv2_subset_rootPath,train_dataset
    elif dataType == 'furniture':
        df = pd.read_csv(CFG.furniture_dfPath)
        train_dataset = DFDataset(df,CFG.furniture_rootPath,data_transforms['train'])
        return df,CFG.furniture_rootPath,train_dataset
    elif dataType == 'storefronts':
        df = pd.read_csv(CFG.storefronts_dfPath)
        train_dataset = DFDataset(df,CFG.storefronts_rootPath,data_transforms['train'])
        return df,CFG.storefronts_rootPath,train_dataset
    elif dataType == 'art':
        df = pd.read_csv(CFG.art_dfPath)
        train_dataset = DFDataset(df,CFG.art_rootPath,data_transforms['train'])
        return df,CFG.art_rootPath,train_dataset
    elif dataType == 'dishes':
        df = pd.read_csv(CFG.dishes_dfPath)
        train_dataset = DFDataset(df,CFG.dishes_rootPath,data_transforms['train'])
        return df,CFG.dishes_rootPath,train_dataset
    elif dataType == 'sop':
        train_image_dict,val_image_dict = give_sop_datasets('/root/autodl-tmp/sop/')
        database_image_dict,query_image_dict  = split_database_dict(val_image_dict)
        # config_ = {'batch_size':4000,'samples_per_class':4}
        train_dataset = SOPDataSet(train_image_dict,transforms=data_transforms['train'])
        return train_dataset 

    elif dataType == 'sop_query':
        train_image_dict,val_image_dict = give_sop_datasets('/root/autodl-tmp/sop/')
        database_image_dict,query_image_dict  = split_database_dict(val_image_dict)
        query_dataset = SOPDataSet(query_image_dict,transforms=data_transforms['valid'])
        return query_dataset
    elif dataType == 'sop_database':
        train_image_dict,val_image_dict = give_sop_datasets('/root/autodl-tmp/sop/')
        database_image_dict,query_image_dict  = split_database_dict(val_image_dict)
        database_dataset = SOPDataSet(database_image_dict,transforms=data_transforms['valid'])
        return database_dataset

    elif dataType == 'gldv2_database':
        df = pd.read_csv(CFG.gldv2_database_dfPath)
        database_dataset = DFDataset(df,CFG.gldv2_database_rootPath,data_transforms['valid'])
        return df,CFG.gldv2_database_rootPath,database_dataset
    elif dataType == 'gldv2_query':
        df = pd.read_csv(CFG.gldv2_query_dfPath)
        query_dataset = DFDataset(df,CFG.gldv2_query_rootPath,data_transforms['valid'])
        return df,CFG.gldv2_query_rootPath,query_dataset

    elif dataType == 'deepfashion_database':
        df = pd.read_csv(CFG.deepfashion_database_dfPath)
        database_dataset = DFDataset(df,CFG.deepfashion_database_rootPath,data_transforms['valid'])
        return df,CFG.deepfashion_database_rootPath,database_dataset
    elif dataType == 'deepfashion_query':
        df = pd.read_csv(CFG.deepfashion_query_dfPath)
        query_dataset = DFDataset(df,CFG.deepfashion_query_rootPath,data_transforms['valid'])
        return df,CFG.deepfashion_query_rootPath,query_dataset

    elif dataType == 'merge_train':
        df_merge = pd.DataFrame()
        id_accumulate = 0
        for i in range(len(CFG.merge_list)):
            df,df_rootPath = get_dataset(CFG.merge_list[i])[0:2]
            df['image_path'] = df.image_path.apply(lambda x:df_rootPath+x)
            df['id'] = df.id.apply(lambda x:x+id_accumulate)
            id_accumulate = df['id'].max() + 1
            df_merge = pd.concat([df_merge,df])
        merge_dataset = DFDataset(df_merge,'',data_transforms['train'])
        return df_merge,None,merge_dataset

    elif dataType == 'merge_database':
        df_merge = pd.DataFrame()
        id_accumulate = 0
        for i in range(len(CFG.merge_list)):
            df,df_rootPath = get_dataset(CFG.merge_list[i]+'_database')[0:2]
            df['image_path'] = df.image_path.apply(lambda x:df_rootPath+x)
            df['id'] = df.id.apply(lambda x:x+id_accumulate)
            id_accumulate += df['id'].max() + 1
            df_merge = pd.concat([df_merge,df])
        merge_dataset = DFDataset(df_merge,'',data_transforms['valid'])
        return None,None,merge_dataset

    elif dataType == 'merge_query':
        df_merge = pd.DataFrame()
        id_accumulate = 0
        for i in range(len(CFG.merge_list)):
            df,df_rootPath = get_dataset(CFG.merge_list[i]+'_query')[0:2]
            df['image_path'] = df.image_path.apply(lambda x:df_rootPath+x)
            df['id'] = df.id.apply(lambda x:x+id_accumulate)
            id_accumulate += df['id'].max() + 1
            df_merge = pd.concat([df_merge,df])
        merge_dataset = DFDataset(df_merge,'',data_transforms['valid'])
        return None,None,merge_dataset

    elif dataType == 'tfrec':
        def parse_tfrecord(single_record):  #解码tfrecord文件并将其转化成训练时的张量，这个其实是对数据transform
            image, label = single_record    #和tensorflow解码类似
            image = torch.tensor(single_record["image/encoded"]) #先将其转化为向量
            
            label = torch.tensor(single_record["image/class/label"]).squeeze() 
            #label读出[[label1], [label2],...]，如果不降维，你每次取label就直接是一个元组[label1]，无法进行训练
            #降维之后就是[label1, label2,...]
            
            image = torchvision.io.decode_jpeg(image).float()
            image = torchvision.transforms.Resize([224,224])(image)/255.0
            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)
            #将image读出来重组成jpeg，操作和tensorflow类似，.float()是因为网络权重是float类型，两者必须相同
            
            return (image, label)

        dataList = glob('/root/autodl-tmp/GUIE-data/GUIE-TFRecords/*')
        splits={}
        for i in range(len(dataList)):
            splits[dataList[i].split('/')[-1].split('.')[0]] = 1
        tfrecord_pattern = "/root/autodl-tmp/GUIE-data/GUIE-TFRecords/{}.tfrec"
        index_pattern = "/root/autodl-tmp/GUIE-data/GUIE-Indexes/{}.index"
        description = {"image/encoded": "byte", "image/class/label": "int"} 
        train_dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern,splits, description,transform=parse_tfrecord,infinite=False)
        return train_dataset
    else:
        raise Exception("Invalid dataset type")

def split_database_dict(image_dict,num_query_classes=10000,sample_per_classes = 1):
    '''
    image_dict: the val_image_dict. split 
    '''
    query_dict = {}
    sample_items = range(len(image_dict))
    for idx,item in enumerate(sample_items):
        if sample_per_classes<len(image_dict[item]):
            query_dict[idx] = image_dict[item][:sample_per_classes]
            image_dict[item] = image_dict[item][sample_per_classes:]
    return image_dict,query_dict

def give_sop_datasets(path):
	train_image_dict, test_image_dict = {}, {}
	train_data = open(os.path.join(path, 'Ebay_train.txt'), 'r').read().splitlines()[1:]
	test_data = open(os.path.join(path, 'Ebay_test.txt'), 'r').read().splitlines()[1:]
	for entry in train_data:
		info = entry.split(' ')
		class_id = info[1]
		im_path = os.path.join(path, info[3])
		if class_id not in train_image_dict.keys():
			train_image_dict[class_id] = []
		train_image_dict[class_id].append(im_path)
	for entry in test_data:
		info = entry.split(' ')
		class_id = info[1]
		im_path = os.path.join(path, info[3])
		if class_id not in test_image_dict.keys():
			test_image_dict[class_id] = []
		test_image_dict[class_id].append(im_path)

	new_train_dict = {}
	class_ind_ind = 0
	for cate in train_image_dict:
		new_train_dict[class_ind_ind] = train_image_dict[cate]
		class_ind_ind += 1
	train_image_dict = new_train_dict
	new_test_dict = {}
	class_ind_ind = 0
	for cate in test_image_dict:
		new_test_dict[class_ind_ind] = test_image_dict[cate]
		class_ind_ind += 1
	test_image_dict = new_test_dict
	return train_image_dict,test_image_dict

class GUIETrainDataSet(torch.utils.data.Dataset):
    def __init__(self,image_dict,config,transforms=None):
        self.image_dict = image_dict
        self.batch_size = config['batch_size']
        self.samples_per_class = config['samples_per_class']
        self.transforms = transforms
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.reshuffle()
        
    def reshuffle(self):
        '''
        sample_per_class:每个类采样多少个
        '''
        image_dict = copy.deepcopy(self.image_dict) 
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])
        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >=self.samples_per_class) and (len(batch) < self.batch_size/self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:] 
            if len(batch) == self.batch_size/self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        image_path = self.dataset[index][1]
        label = self.dataset[index][0]
        img = cv2.imread(image_path)
        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        img = img.transpose(2,0,1)
        
        return torch.tensor(img),torch.tensor(label)

class SOPDataSet(torch.utils.data.Dataset):
    def __init__(self,image_dict,transforms=None):
        self.image_dict = image_dict
        self.transforms = transforms
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.image_list = []
        for item in self.image_dict:
            self.image_list+=self.image_dict[item]
        self.image_dict = self.image_list.copy()
        del self.image_list

    def __len__(self):
        return len(self.image_dict)
    
    def __getitem__(self,index):
        image_path = self.image_dict[index][1]
        label = self.image_dict[index][0]
        img = cv2.imread(image_path)
        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        img = img.transpose(2,0,1)
        
        return torch.tensor(img),torch.tensor(label)

class TFrecordDataset(torch.utils.data.Dataset):
    def __init__(self,tfrec_list,transforms=None):
        self.image_dict = tfrec_list
        self.transforms = transforms
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.image_list = []
        for item in self.image_dict:
            self.image_list+=self.image_dict[item]
        self.image_dict = self.image_list.copy()
        del self.image_list

    def __len__(self):
        return len(self.image_dict)
    
    def __getitem__(self,index):
        image_path = self.image_dict[index][1]
        label = self.image_dict[index][0]
        img = cv2.imread(image_path)
        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        img = img.transpose(2,0,1)
        
        return torch.tensor(img),torch.tensor(label)

class DFDataset(torch.utils.data.Dataset):
    def __init__(self,df,root_path,transforms=None) -> None:
        self.root_path = root_path
        self.ImageList = list(df.image_path)
        self.LabelList = list(df.id)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ImageList)
    
    def __getitem__(self,index):
        img = cv2.imread(self.root_path+self.ImageList[index])
        label = self.LabelList[index]
        assert img is not None
        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        img = img.transpose(2,0,1)
        return torch.tensor(img),torch.tensor(label)

class DFClsDataset(torch.utils.data.Dataset):
    def __init__(self,df,root_path,transforms=None) -> None:
        self.root_path = root_path
        self.ImageList = list(df.image_path)
        self.LabelList = list(df.id)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ImageList)
    
    def __getitem__(self,index):
        img = cv2.imread(self.root_path+self.ImageList[index])
        if 'deepfashion' in  self.root_path+self.ImageList[index]:
            label = 0
        elif 'gldv2' in  self.root_path+self.ImageList[index]:
            label = 1
        else:
            label = 2
        assert img is not None
        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        
        return torch.tensor(img),torch.tensor(label)