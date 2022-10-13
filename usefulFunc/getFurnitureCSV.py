import numpy as np
import pandas as pd
from glob import glob

classList = glob('/root/autodl-tmp/GUIE-data/furniture/*')
image_path = []
id_list = []
cnt=0
for i in range(len(classList)):
    imgList = glob(classList[i]+'/*')
    if len(imgList)==0:
        cnt-=1
    for j in range(len(imgList)):
        image_path.append(imgList[j])
        id_list.append(cnt)
    cnt+=1
df = pd.DataFrame(list(zip(image_path,id_list)),columns=['image_path','id'])
df.to_csv('./CSVfiles/furniture.csv')

