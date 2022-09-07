import torch
import cv2
import os
import random
import copy
from config import CFG

flatten = lambda l: [item for sublist in l for item in sublist]

def split_database_dict(image_dict,num_query_classes=100,sample_per_classes = 1):
    '''
    image_dict: the val_image_dict. split 
    '''
    database_dict = {}
    sample_items = random.sample(range(len(image_dict)),num_query_classes)
    for idx,item in enumerate(sample_items):
        if sample_per_classes<len(image_dict[item]):
            database_dict[idx] = image_dict[item][:sample_per_classes]
            image_dict[item] = image_dict[item][sample_per_classes:]
    return image_dict,database_dict

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

class GUIEValDataSet(torch.utils.data.Dataset):
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
