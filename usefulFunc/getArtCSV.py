import numpy as np
import pandas as pd
from glob import glob

classList = glob('/root/autodl-tmp/GUIE-data/art/*')
image_path = []
id_list = []
for i in range(len(classList)):
    imgList = glob(classList[i]+'/*')
    for j in range(len(imgList)):
        image_path.append(imgList[j])
        id_list.append(i)
df = pd.DataFrame(list(zip(image_path,id_list)),columns=['image_path','id'])
df.to_csv('./CSVfiles/art.csv')

