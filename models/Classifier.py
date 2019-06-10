import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TE_Module import TE_Module
class Classifier(nn.Module):
    def __init__(self, _num_stages=3, _use_avg_on_conv3=True, indim=128, num_classes=10):
        """
        :param _num_stages: block combination
        :param _use_avg_on_conv3: finally use avg or not
        :param indim:
        :param num_classes: transformation matrix
        """
        #nChannels = 192
        super(Classifier, self).__init__()
        self.clf = TE_Module(_num_stages=_num_stages, _use_avg_on_conv3=_use_avg_on_conv3)
        self.fc = nn.Linear(indim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
        #self.attention=Self_Attn(nChannels, 'relu')
    def forward(self, x1):
        x1,_ = self.clf(x1)
        return self.fc(x1)