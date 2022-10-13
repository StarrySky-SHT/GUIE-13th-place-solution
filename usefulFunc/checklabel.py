from genericpath import exists
from glob import glob
import os
import pandas as pd
import random
import shutil

if not os.path.exists('/root/GUIE/GUIE/usefulFunc/checklabelfolder'):
    os.mkdir('/root/GUIE/GUIE/usefulFunc/checklabelfolder/')

rootPath = ''
df = pd.read_csv('/root/GUIE/GUIE/CSVfiles/furniture.csv')
imgList = list(df.image_path)

sample_id = random.randint(0,99)
sampled_image = list(df.query('id==@sample_id').image_path)
for i in range(len(sampled_image)):
    shutil.copy(rootPath+sampled_image[i],'/root/GUIE/GUIE/usefulFunc/checklabelfolder/'+str(i)+'.jpg')
