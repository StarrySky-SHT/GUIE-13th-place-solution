# CFG
import torch
class CFG:
    seed          = 101
    train_bs      = 192
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 50
    lr            = 2e-2
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    num_classes   = 11318
    embed         = 256
    device = "cuda:0"
    n_accumulate  = 16
    torch.set_default_tensor_type('torch.FloatTensor')