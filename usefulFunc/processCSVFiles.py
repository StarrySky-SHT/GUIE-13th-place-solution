import pandas as pd
import numpy as np

df = pd.read_csv('/root/autodl-tmp/GUIE-data/products10k/train.csv')
df['image_path'] = df.name.apply(lambda x:'/root/autodl-tmp/GUIE-data/products10k/train/'+x)
df.rename(columns={'class':'id'},inplace=True)
del df['name']
df.to_csv('/root/GUIE/GUIE/CSVfiles/products10k.csv')