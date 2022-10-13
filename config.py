# CFG
import torch
class CFG:
    seed          = 101
    train_bs      = 80
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 10
    lr            = 1e-5
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-7
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    num_classes   = 10645
    embed         = 64
    device = torch.device('cuda')
    n_accumulate  = 8
    torch.set_default_tensor_type('torch.FloatTensor')

    gldv2_rootPath = '/root/autodl-tmp/GUIE-data/gldv2_mini/'   # 9.56867
    gldv2_dfPath = '/root/GUIE/GUIE/CSVfiles/gldv2_train.csv'  # 8533 classes 85341 images 0.276418 3.6177 0.378077

    gldv2_query_rootPath = '/root/autodl-tmp/GUIE-data/gldv2_micro/images/'
    gldv2_query_dfPath = '/root/autodl-tmp/GUIE-data/gldv2_micro/query.csv'

    gldv2_database_rootPath = '/root/autodl-tmp/GUIE-data/gldv2_micro/images/'
    gldv2_database_dfPath = '/root/autodl-tmp/GUIE-data/gldv2_micro/database.csv'

    gldv2_subset_rootPath = ''
    gldv2_subset_dfPath = '/root/GUIE/GUIE/CSVfiles/gldv2_subset.csv' # 415 classes 23898 images

    furniture_rootPath = ''
    furniture_dfPath = '/root/GUIE/GUIE/CSVfiles/furniture.csv' #280 classes 3688 images

    storefronts_rootPath = ''
    storefronts_dfPath = '/root/GUIE/GUIE/CSVfiles/store_fronts.csv' #140 classes 4112 images

    art_rootPath = ''
    art_dfPath = '/root/GUIE/GUIE/CSVfiles/art.csv' # 51 classes 1187 images

    dishes_rootPath = ''
    dishes_dfPath = '/root/GUIE/GUIE/CSVfiles/dishes.csv' # 67 classes 670 images

    deepfashion_rootPath = '' # 处理的时候已经加入了rootpath
    deepfashion_dfPath = '/root/GUIE/GUIE/CSVfiles/deepfashion_train.csv' # 8000 classes 81461 images 0.264852 3.77569 0.394588

    products10k_rootPath = '' # 处理的时候已经加入了rootpath
    products10k_dfPath = '/root/GUIE/GUIE/CSVfiles/products10k.csv' # 9691 classes 141930 images cls 0.45971 2.17528 0.22733

    deepfashion_database_rootPath = '' # 处理的时候已经加入了rootpath
    deepfashion_database_dfPath = '/root/GUIE/GUIE/CSVfiles/deepfashion_val_database.csv'

    deepfashion_query_rootPath = '' # 处理的时候已经加入了rootpath
    deepfashion_query_dfPath = '/root/GUIE/GUIE/CSVfiles/deepfashion_val_query.csv'

    merge_list = ['products10k','gldv2_subset','storefronts','furniture','art','dishes']

    dilations = [4,8,12]
    arcface_s = 45
    arcface_m_x =  0.45
    arcface_m_y = 0.05