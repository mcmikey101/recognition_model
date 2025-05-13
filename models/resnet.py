from torch import nn
import torch
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, output_classes, output_embedding=512, model_type='resnet50', weights=None):
        super().__init__()
        if model_type == "resnet18":
            self.model = models.resnet18(weights=weights)
        elif model_type == "resnet34":
            self.model = models.resnet34(weights=weights)
        elif model_type == "resnet50":
            self.model = models.resnet50(weights=weights)
        else:
            raise NotImplementedError 
        
        self.output_embedding = output_embedding
        
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.model.fc.in_features, output_classes),
                nn.Linear(self.model.fc.in_features, output_embedding),
            ]
        )
        self.model.fc = nn.Identity()

    def forward(self, x):
        result = self.model(x)
        return self.heads[0](result), self.heads[1](result)
    
