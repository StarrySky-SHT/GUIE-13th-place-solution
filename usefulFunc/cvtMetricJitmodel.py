import numpy as np
import torch
import timm
import torch.nn as nn
import clip
import  sys
sys.path.append('/root/GUIE/GUIE')
from config import CFG
from losses import *
from dataset import get_dataset
import open_clip

class CreateClipModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = clip.clip.load('ViT-L/14',jit=True)[0].visual
        self.linear = nn.Linear(in_features=in_features,out_features=CFG.embed)
        self.bn = nn.BatchNorm1d(CFG.embed)
        self.dropout = nn.Dropout(p=0.2)
        # self.margin = ArcMarginProduct(
        #     in_feature = CFG.embed,
        #     out_feature = CFG.num_classes, 
        #     s = 30, 
        #     m = 0.5
        #     )
            
        self._init_params()
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        # x = self.margin(x,label)
        return x

    def feature_extractor(self,x):
        x_ = self.backbone.conv1(x.cuda())  # shape = [*, width, grid, grid]
        x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
        x_ = self.backbone.ln_pre(x_)

        x_ = x_.permute(1, 0, 2)  # NLD -> LND
        x_ = self.backbone.transformer.resblocks(x_)
        x_ = x_.permute(1, 0, 2)  # LND -> NLD
        x_ = self.backbone.ln_post(x_[:, 0, :])
        if self.backbone.proj is not None:
            x_ = x_ @ self.backbone.proj

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateClipVitDyMarginModel(nn.Module):
    def __init__(self,in_features=1000,df=None):
        super().__init__()
        self.backbone = clip.clip.load('ViT-L/14',jit=True)[0].visual
        # self.backbone = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')[0].visual
        self.linear = nn.Linear(in_features=in_features,out_features=CFG.embed)
        self.bn = nn.BatchNorm1d(CFG.embed)
        self.dropout = nn.Dropout(p=0.2)
        # self.margin = ArcMarginProduct(
        #     in_feature = CFG.embed,
        #     out_feature = CFG.num_classes, 
        #     s = 30, 
        #     m = 0.5
        #     )
        # tmp = np.sqrt(1 / np.sqrt(df['id'].value_counts().sort_index().values))
        # dy_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * CFG.arcface_m_x + CFG.arcface_m_y
        # self.head = ArcMarginProduct_subcenter(CFG.embed,CFG.num_classes)
        # self.margin = ArcFaceLossAdaptiveMargin(dy_margins,CFG.num_classes,CFG.arcface_s)
            
        self._init_params()
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        # x = self.head(x)
        # x = self.margin(x,label)
        return x

    def feature_extractor(self,x_):
        x_ = self.backbone.conv1(x_.cuda())  # shape = [*, width, grid, grid]
        x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
        x_ = self.backbone.ln_pre(x_)

        x_ = x_.permute(1, 0, 2)  # NLD -> LND
        x_ = self.backbone.transformer.resblocks(x_)
        x_ = x_.permute(1, 0, 2)  # LND -> NLD
        x_ = self.backbone.ln_post(x_[:, 0, :])
        if self.backbone.proj is not None:
            x_ = x_ @ self.backbone.proj

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b7_ns',pretrained=True)
        self.backbone.classifier = nn.Identity()
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
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.backbone.global_pool(x)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        return x
from transformers import FlavaFeatureExtractor, FlavaImageModel
class CreateFlavaModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = FlavaImageModel.from_pretrained("facebook/flava-full")
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
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        x = self.margin(x,label)
        return x

    def feature_extractor(self,x_):
        with torch.no_grad():
            x_ = self.backbone.embeddings(x_)
            x_ = self.backbone.encoder.layer[0](x_)[0]
            x_ = self.backbone.encoder.layer[1](x_)[0]
            x_ = self.backbone.encoder.layer[2](x_)[0]
            x_ = self.backbone.encoder.layer[3](x_)[0]
            x_ = self.backbone.encoder.layer[4](x_)[0]
            x_ = self.backbone.encoder.layer[5](x_)[0]
            x_ = self.backbone.encoder.layer[6](x_)[0]
            x_ = self.backbone.encoder.layer[7](x_)[0]
        x_ = self.backbone.encoder.layer[8](x_)[0]
        x_ = self.backbone.encoder.layer[9](x_)[0]
        x_ = self.backbone.encoder.layer[10](x_)[0]
        x_ = self.backbone.encoder.layer[11](x_)[0]
        x_ = self.backbone.layernorm(x_)
        x_ = self.backbone.pooler(x_)

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateClipRNModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = clip.clip.load('RN50x16',jit=True)[0].visual
        # self.backbone = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')[0].visual
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
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
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
        x = self.linear(x.float())
        x = self.bn(x)
        return x

class CreateClipOpenAiModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',cache_dir='/root/autodl-tmp/OpenAI',jit=True)[0].visual
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
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        x = self.margin(x,label)
        return x

    def feature_extractor(self,x_):
        x_ = self.backbone.conv1(x_.cuda())  # shape = [*, width, grid, grid]
        x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
        x_ = self.backbone.ln_pre(x_)
        x_ = x_.permute(1, 0, 2)  # NLD -> LND
        x_ = self.backbone.transformer(x_)
        x_ = x_.permute(1, 0, 2)  # LND -> NLD
        x_ = self.backbone.ln_post(x_[:, 0, :])
        if self.backbone.proj is not None:
            x_ = x_ @ self.backbone.proj

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

# train_df = get_dataset(dataType='merge_train')[0]
# model = CreateClipModel(in_features=768)
model = CreateClipOpenAiModel(in_features=1024)
# model = CreateModel(2560)
# model.bn = nn.Sequential(nn.BatchNorm1d(CFG.embed),
#                                  nn.Linear(CFG.embed,64),
#                                  nn.BatchNorm1d(64))
model.load_state_dict(torch.load('/root/autodl-tmp/trained_models/clip_vitH_14_embed64_state_dict.pt'))
# model = nn.DataParallel(model)
jitmodel = torch.jit.script(model)
jitmodel.save('/root/autodl-tmp/trained_models/clip_vitH_14_embed64_jit.pt')

