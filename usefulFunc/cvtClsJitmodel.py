import numpy as np
import torch
import timm
import torch.nn as nn

class CreateClsModel(nn.Module):
    def __init__(self,in_features=1000,num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4',pretrained=True)
        self.backbone.classifier = nn.Linear(1792,num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
model = CreateClsModel()
model.load_state_dict(torch.load('/root/autodl-tmp/trained_models/cls_state_dict.pt'))
jitmodel = torch.jit.script(model)
jitmodel.save('/root/autodl-tmp/trained_models/cls_jit.pt')