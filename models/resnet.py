from torch import nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, output_classes, model_type='resnet50', output_embedding=512, weights=None):
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

        self.model.fc = nn.Linear(self.model.fc.in_features, self.output_embedding)

    def forward(self, x):
        return self.model(x)
    
