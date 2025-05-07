import torch
from torch import nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, model_type='resnet50', output_shape=512, pretrained=False):
        super().__init__()
        if model_type == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_type == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_type == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError 
        
        self.output_shape = output_shape

        self.model.fc = nn.Linear(self.model.fc.in_features, self.output_shape)

    def forward(self, x):
        return self.model(x)
    
