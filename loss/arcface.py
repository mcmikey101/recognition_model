import torch
from torch import nn
import math
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, embed_size, num_classes, scale, margin):
        super(ArcFaceLoss, self).__init__()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cos_theta = torch.matmul(embeddings, weights.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(0.0, 1.0))

        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits *= self.scale

        loss = self.ce(logits, labels)
        return loss
    
    