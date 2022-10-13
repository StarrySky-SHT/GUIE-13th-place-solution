import numpy as np
import pandas as pd
from glob import glob
import random

# df_gldv2 = pd.read_csv('/root/GUIE/GUIE/CSVfiles/gldv2_train.csv')
# sample_class = random.sample(list(df_gldv2.id.unique()),k=200)
# df_gldv2_subset = df_gldv2[df_gldv2.id.isin(sample_class)]
# del df_gldv2_subset['new_landmark_id']
# del df_gldv2_subset['Unnamed: 1']
# origin_id = list(df_gldv2_subset.id.unique())
# new_id = list(range(len(origin_id)))
# f_ = {}
# for i in range(len(new_id)):
#     f_[origin_id[i]] = new_id[i]
# df_gldv2_subset['id'] =  df_gldv2_subset.id.apply(lambda x:f_[x])
# df_gldv2_subset.to_csv('/root/GUIE/GUIE/CSVfiles/gldv2_subset_train.csv')
classList = glob('/root/autodl-tmp/GUIE-data/gldv2_subset/*')
image_path = []
id_list = []
for i in range(len(classList)):
    imgList = glob(classList[i]+'/*')
    for j in range(len(imgList)):
        image_path.append(imgList[j])
        id_list.append(i)
df = pd.DataFrame(list(zip(image_path,id_list)),columns=['image_path','id'])
df.to_csv('./CSVfiles/gldv2_subset.csv')
