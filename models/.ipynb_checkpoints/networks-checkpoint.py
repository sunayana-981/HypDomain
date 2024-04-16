# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import copy



class clip(nn.Module):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.n_outputs = 512
        self.freeze()
    def freeze(self):        
        for names, params in model.named_parameters():
            params.requires_grad = False
    def forward(self,x):
        inputs = processor(images=x, return_tensors="pt")
        return self.model.get_image_features(**inputs)


def Featurizer(input_shape):
    """Auto-select an appropriate featurizer for the given input shape."""
    return clip()
    


def Classifier(in_features, out_features):
    
    return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU()) , in_features // 4

def final(in_features, out_features):
    return torch.nn.Linear(in_features, out_features)


