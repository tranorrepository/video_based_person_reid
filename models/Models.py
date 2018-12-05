# -*- coding: utf-8 -*
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision






__all__ = ['Base', 'Base_spatial', 'Base_temporal','Ours']


class Base(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Base, self).__init__()
        self.loss = loss
        models = torchvision.models.resnet50(pretrained=True)
        models = list(models.children())[:-2]
        models.append(nn.AdaptiveAvgPool2d(1))
        models.append(nn.Dropout())
        self.model = nn.Sequential(*models)
        self.feat_dim = 2048
        self.emb_dim=1024
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.embedding = nn.Sequential(nn.Linear(self.feat_dim, self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        self.logits = nn.Sequential(nn.Dropout(),
                                nn.ReLU(),
                                nn.Linear(self.emb_dim, num_classes))

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.model(x).view(x.size(0),-1)
        x = self.embedding(x)
        x = x.view(b,t,-1)

        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, -1)
        if not self.training:
            return f
        y = self.logits(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))



class Base_spatial(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Base_spatial, self).__init__()
        self.loss = loss
        self.attn0 = Self_Attn(256, 'relu')
        self.attn1 = Self_Attn(512, 'relu')
        self.attn2 = Self_Attn(1024, 'relu')
        self.attn3 = Self_Attn(2048, 'relu')
        models = torchvision.models.resnet50(pretrained=True)
        models = list(models.children())[:-2]
        models.insert(-1, self.attn2)
        models.append(self.attn3)
        models.append(nn.AdaptiveAvgPool2d(1))
        models.append(nn.Dropout())
        self.model = nn.Sequential(*models)
        self.feat_dim = 2048
        self.emb_dim=1024
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.embedding = nn.Sequential(nn.Linear(self.feat_dim, self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        self.logits = nn.Sequential(nn.Dropout(),
                                nn.ReLU(),
                                nn.Linear(self.emb_dim, num_classes))

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.model(x)
        x = self.embedding(x)
        x = x.view(b,t,-1)

        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, -1)
        if not self.training:
            return f
        y = self.logits(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class Base_temporal(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Base_temporal, self).__init__()
        self.loss = loss
        models = torchvision.models.resnet50(pretrained=True)
        models = list(models.children())[:-2]
        models.append(nn.AdaptiveAvgPool2d(1))
        models.append(nn.Dropout())
        self.model = nn.Sequential(*models)
        self.feat_dim = 2048
        self.emb_dim=1024
        self.embedding = nn.Sequential(nn.Linear(self.feat_dim, self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        self.quality = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

        self.logits = nn.Sequential(nn.Dropout(),
                                nn.ReLU(),
                                nn.Linear(self.emb_dim, num_classes))
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.model(x).view(x.size(0),-1)
        emb = self.embedding(x)

        qual = self.quality(x)
        qual = qual.view(b, t, 1)


        a = (qual.sum(dim=1) + 1e-12)
        v_emb = (qual * emb.view(b, t, -1)).sum(dim=1) / a
        if not self.training:
            return v_emb
        logit = self.logits(v_emb)
        return  v_emb, logit






class Ours(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Ours, self).__init__()
        self.attn0 = Self_Attn(256, 'relu')
        self.attn1 = Self_Attn(512, 'relu')
        self.attn2 = Self_Attn(1024, 'relu')
        self.attn3 = Self_Attn(2048, 'relu')
        models = torchvision.models.resnet50(pretrained=True)
        models = list(models.children())[:-2]
        models.insert(-1, self.attn2)
        models.append(self.attn3)
        models.append(nn.AdaptiveAvgPool2d(1))
        models.append(nn.Dropout())
        self.model = nn.Sequential(*models)
        self.feat_dim = 2048
        self.emb_dim = 1024
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.embedding = nn.Sequential(nn.Linear(2048, self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        self.quality = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
        self.num_calsses = num_classes
        if num_classes is not None:
            self.logits = nn.Sequential(nn.Dropout(),
                                        nn.ReLU(),
                                        nn.Linear(self.emb_dim, num_classes))

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.model(x).view(x.size(0),-1)
        qual = self.quality(x)
        qual = qual.view(b, t, 1)
        emb = self.embedding(x)
        a = (qual.sum(dim=1) + 1e-12)
        v_emb = (qual * emb.view(b, t, -1)).sum(dim=1) / a
        if not self.training:
            return v_emb
        logit = self.logits(v_emb)
        return  v_emb, logit




class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        # return out, attention

        return out