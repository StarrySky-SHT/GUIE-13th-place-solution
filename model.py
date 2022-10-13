from cgitb import reset
import torch.nn as nn
import torch
import torch.nn.functional as F
from losses import ArcMarginProduct,ArcMarginProduct_subcenter,ArcFaceLossAdaptiveMargin
from config import CFG
from torchvision import models
import timm
# creat model
from transformers import FlavaFeatureExtractor, FlavaImageModel
import clip
import open_clip
import numpy as np
# clip_vit = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
# backbone = models.resnet50(pretrained=True)
from torchvision import transforms


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=dilations[0], padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=dilations[1], padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=dilations[2], padding='same')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.o1 = nn.Conv2d(in_chans,512,kernel_size=1)

        self.conv1 = nn.Conv2d(512 * 4, out_chans, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self,x):
        
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = F.interpolate(self.relu1(self.o1(self.pooling(x))),x.shape[2:]) 
        x = torch.cat([x0,x1,x2,x3],dim=1)
        x = self.conv1(x)
        x = self.relu2(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score   

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  

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
        with torch.no_grad():  
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            x = self.backbone.blocks[0:3](x)
        x = self.backbone.blocks[3:](x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.backbone.global_pool(x)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        return x

class CreateDolgModel(nn.Module):
    def __init__(self,in_features=1000,dataset=None):

        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4',pretrained=True)
        self.embedding = nn.Sequential(self.backbone.conv_stem,self.backbone.bn1,self.backbone.act1)
        self.feature_extractor_stage1_3 = nn.Sequential(self.backbone.blocks[0:5])
        self.feature_extractor_stage4 = nn.Sequential(self.backbone.blocks[5:])
        self.head = nn.Sequential(self.backbone.conv_head,self.backbone.bn2,self.backbone.act2,GeM(p_trainable=True))

        self.freeze_weight()
        self.unfreeze_weight(self.head)
        self.margin = ArcMarginProduct(
            in_feature = CFG.embed,
            out_feature = CFG.num_classes, 
            s = 30, 
            m = 0.5
            )

        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g
        
        self.linear = nn.Linear(backbone_out,feature_dim_l_g)
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)

        # local
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, CFG.dilations)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

        self.linear_out = nn.Linear(in_features, feature_dim_l_g, bias=True)
        self.neck = nn.Sequential(
                nn.Linear(fusion_out, CFG.embed, bias=True),
                nn.BatchNorm1d(CFG.embed)
            )
        # self.head = ArcMarginProduct_subcenter(CFG.embed, CFG.num_classes)
        # self.margin = ArcFaceLossAdaptiveMargin(dataset.margins,CFG.num_classes,CFG.arcface_s)
        del self.backbone

    def forward(self, x,label):
        x = self.feature_extractor(x)
        loss = self.margin(x,label)
        return loss

    def feature_extractor(self,x):

        x_l = self.feature_extractor_stage1_3(self.embedding(x))

        x_g = torch.nn.Flatten()(self.head(self.feature_extractor_stage4(x_l)))
        x_g = self.linear_out(x_g)

        x_l = self.mam(x_l)
        x_l, _ = self.attention2d(x_l)

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]  

        return self.neck(x_fused)
    
    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weight(self,Seq):
        for item in Seq:
            for param in item.parameters():
                param.requires_grad = True

class CreateClipOpenAiModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',cache_dir='/root/autodl-tmp/OpenAI')[0].visual
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
            x_ = self.backbone.conv1(x_.cuda())  # shape = [*, width, grid, grid]
            x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
            x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
            x_ = self.backbone.ln_pre(x_)
            x_ = x_.permute(1, 0, 2)  # NLD -> LND
            for i in range(len(self.backbone.transformer.resblocks[0:24])):
                x_ = self.backbone.transformer.resblocks[i](x_)
        for i in range(len(self.backbone.transformer.resblocks[24:])):
            x_ = self.backbone.transformer.resblocks[24+i](x_)
        x_ = x_.permute(1, 0, 2)  # LND -> NLD
        x_ = self.backbone.ln_post(x_[:, 0, :])
        if self.backbone.proj is not None:
            x_ = x_ @ self.backbone.proj

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateClipVitModel(nn.Module):
    def __init__(self,in_features=1000,df=None,backbone='ViT-L/14'):
        super().__init__()
        self.backbone = clip.clip.load(backbone)[0].visual
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
        with torch.no_grad():
            x_ = self.backbone.conv1(x_.cuda())  # shape = [*, width, grid, grid]
            x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
            x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
            x_ = self.backbone.ln_pre(x_)

            x_ = x_.permute(1, 0, 2)  # NLD -> LND
            x_ = self.backbone.transformer.resblocks[0:16](x_)
        x_ = self.backbone.transformer.resblocks[16:](x_)
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
        self.backbone = clip.clip.load('ViT-L/14')[0].visual
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
        tmp = np.sqrt(1 / np.sqrt(df['id'].value_counts().sort_index().values))
        dy_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * CFG.arcface_m_x + CFG.arcface_m_y
        self.head = ArcMarginProduct_subcenter(CFG.embed,CFG.num_classes)
        self.margin = ArcFaceLossAdaptiveMargin(dy_margins,CFG.num_classes,CFG.arcface_s)
            
        self._init_params()
        self.avg = nn.AdaptiveAvgPool1d(CFG.embed)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x,label):
        x = self.feature_extractor(x)
        x = self.head(x)
        x = self.margin(x,label)
        return x

    def feature_extractor(self,x_):
        with torch.no_grad():
            x_ = self.backbone.conv1(x_.cuda())  # shape = [*, width, grid, grid]
            x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # shape = [*, width, grid ** 2]
            x_ = x_.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x_ = torch.cat([self.backbone.class_embedding.to(x_.dtype) + torch.zeros(x_.shape[0], 1, x_.shape[-1], dtype=x_.dtype, device=x_.device), x_], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x_ = x_ + self.backbone.positional_embedding.to(x_.dtype)
            x_ = self.backbone.ln_pre(x_)

            x_ = x_.permute(1, 0, 2)  # NLD -> LND
            x_ = self.backbone.transformer.resblocks[0:18](x_)
        x_ = self.backbone.transformer.resblocks[18:](x_)
        x_ = x_.permute(1, 0, 2)  # LND -> NLD
        x_ = self.backbone.ln_post(x_[:, 0, :])
        if self.backbone.proj is not None:
            x_ = x_ @ self.backbone.proj

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateClipRNModel(nn.Module):
    def __init__(self,in_features=1000):
        super().__init__()
        self.backbone = clip.clip.load('RN50x16')[0].visual
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
        def stem(x):
            x = self.backbone.relu1(self.backbone.bn1(self.backbone.conv1(x)))
            x = self.backbone.relu2(self.backbone.bn2(self.backbone.conv2(x)))
            x = self.backbone.relu3(self.backbone.bn3(self.backbone.conv3(x)))
            x = self.backbone.avgpool(x)
            return x
        with torch.no_grad():
            x = x.type(self.backbone.conv1.weight.dtype)
            x = stem(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.attnpool(x)

        x = self.dropout(x)
        x = self.linear(x.float())
        x = self.bn(x)
        return x

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
        x_ = self.backbone(x_).pooler_output

        x_ = self.dropout(x_)
        x_ = self.linear(x_.float())
        x_ = self.bn(x_)
        return x_

class CreateClsModel(nn.Module):
    def __init__(self,in_features=1000,num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4',pretrained=True)
        self.backbone.classifier = nn.Linear(1792,num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x