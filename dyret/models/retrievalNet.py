import timm
from torch import nn
import math
import torch
import torch.nn.functional as F
from tools.torch_utils import weights_init_classifier, weights_init_kaiming


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=10, m=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight)).float()
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

class retrievalNet(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, bnn_neck):
        super(retrievalNet, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        self.bnn_neck = bnn_neck
        self.num_classes = num_classes

        if 'resnet' in model_name:
            self.in_planes = self.model.head.fc.in_channels
            self.model.head.fc = nn.Identity()
        elif 'efficientnet' in model_name:
            self.in_planes = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'se' in model_name:
            self.in_planes = self.model.last_linear.in_features
            self.model.last_linear = nn.Identity()

        # self.margin = ArcModule(in_features=self.in_planes, out_features=num_classes)

        if self.bnn_neck is True:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)

    def forward(self, imgs):
        global_features = self.model(imgs)
        global_features = F.normalize(global_features)

        if self.bnn_neck is True:
            features = self.bottleneck(global_features)
        else:
            features = global_features

        if self.training:
            cls_score = self.classifier(features)
            return cls_score, global_features
        else:
            # print("Test with feature after BN")
            return features

class retrievalNet_margin(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, bnn_neck):
        super(retrievalNet_margin, self).__init__()

        self.num_classes = num_classes
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=self.num_classes)


        if 'resnet' in model_name:
            self.in_planes = self.model.head.fc.in_channels
            self.model.head.fc = nn.Identity()
        elif 'efficientnet' in model_name:
            self.in_planes = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'se' in model_name:
            self.in_planes = self.model.last_linear.in_features
            self.model.last_linear = nn.Identity()

        self.margin = ArcModule(in_features=self.in_planes, out_features=self.num_classes)


    def forward(self, imgs, labels=None):
        global_features = self.model(imgs)
        global_features = F.normalize(global_features)
        if self.training:
            cls_score = self.margin(global_features, labels)
            return cls_score, global_features
        else:
            return global_features

