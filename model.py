from cgitb import reset
import torch.nn as nn
import torch
import torch.nn.functional as F
from losses import ArcMarginProduct
from config import CFG
from torchvision import models

# creat model
from transformers import CLIPProcessor, CLIPVisionModel, CLIPFeatureExtractor
# clip_vit = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
backbone = models.resnet50(pretrained=True)

class VitModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = backbone
        self.linear = nn.Linear(in_features=in_features,out_features=CFG.embed)
        self.bn = nn.BatchNorm1d(CFG.embed)
        self.dropout = nn.Dropout(p=0.2)
        self.margin = ArcMarginProduct(
            in_feature = CFG.embed,
            out_feature = CFG.num_classes, 
            s = 30, 
            m = 0.5
            )
        self._init_params()
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        x = self.margin(x,label)
        return x

    def feature_extractor(self,x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        return x