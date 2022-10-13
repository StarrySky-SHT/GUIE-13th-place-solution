import numpy as np
import pandas as pd
from glob import glob
import random

classList = glob('/root/autodl-tmp/GUIE-data/storefronts/*')
image_path = []
id_list = []
cnt=0
for i in range(len(classList)):
    imgList = glob(classList[i]+'/*')
    for j in range(len(imgList)):
        if 'jpg' not in imgList[j] and 'jpeg' not in imgList[j]:
            cnt-=1
            break
        image_path.append(imgList[j])
        id_list.append(cnt)
    cnt+=1
df = pd.DataFrame(list(zip(image_path,id_list)),columns=['image_path','id'])
df.to_csv('./CSVfiles/store_fronts.csv')
