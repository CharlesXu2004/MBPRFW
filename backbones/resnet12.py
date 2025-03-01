import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import time

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1,layer_=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.layer_=layer_


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.layer_==1:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            out = self.maxpool(out)
            return out
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet12(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, flatten = True):
        self.inplanes = 3
        super(ResNet12, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                      block_size=dropblock_size,layer_=0)




        if flatten:
            self.flatten = True
            self.avgpool = nn.AvgPool2d(5, stride=1)
            self.flatten_module = Flatten()
            self.final_feat_dim = 640
        else:
            self.flatten = False
            self.final_feat_dim = [640, 5, 5]

        self.relu = nn.ReLU()
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        # self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        # self.drop_rate = drop_rate


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,layer_=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,layer_=layer_))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def set_flatten(self, flatten):
        self.flatten = flatten
        if self.flatten:
            self.avgpool = nn.AvgPool2d(5, stride=1)
            self.flatten_module = Flatten()
            self.final_feat_dim = 640
        else:
            self.final_feat_dim = [640, 5, 5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)



        # x = self.relu(x)   # power transform

        # if self.flatten:
        #     out = self.avgpool(x)
        #     out = self.avgpool(out)
        #     out = F.avg_pool2d(out, 2)
        #     out = self.flatten_module(out)

        x = self.layer4(x)
        # out = self.conv3(out)
        # out = self.bn3(out)
        #
        # if self.downsample is not None:
        #     residual = self.downsample(x)
        # out += residual
        #
        # out = self.relu(out)
        # out = self.maxpool(out)
        #
        # if self.drop_rate > 0:
        #     out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        if self.flatten:
            x = self.avgpool(x)
            #x = F.avg_pool2d(x, 2)
            x = self.flatten_module(x)
        x = self.relu(x)


        return x
class overall_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet12()
        self.final_dim = self.backbone.final_feat_dim
        a = np.eye(640, dtype=float)
        a = torch.from_numpy(np.float32(a))

        self.train_w = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            #nn.BatchNorm2d(self.final_dim)
            #nn.BatchNorm1d(self.final_dim, eps=0.001, momentum=0.1, affine=True)
        )
        self.train_w[0].weight.data = a


    def forward(self,x1,x2):
        #feature1 = self.backbone(x1)
        feature1_w = self.train_w(x1)
        #feature2 = self.backbone(x2)
        feature2_w = self.train_w(x2)

        return feature1_w,feature2_w
    def evaluate(self, data_query, data_shot, label,n,BN):


        if n == 5:
            data_shot = data_shot.reshape(5, 5, 640)
            data_shot = torch.mean(data_shot, dim=0)
        if(BN==False):
            data_query = F.normalize(data_query, dim=1)
            data_shot = F.normalize(data_shot, dim=1)
        score = torch.matmul(data_query, data_shot.T)
        #print("score:",score)
        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc

class overall_model_som(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet12()
        self.final_dim = self.backbone.final_feat_dim
        a = np.eye(640, dtype=float)
        a = torch.from_numpy(np.float32(a))

        self.train_w = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            # nn.BatchNorm2d(self.final_dim)
            # nn.BatchNorm1d(self.final_dim, eps=0.001, momentum=0.1, affine=True)
        )
        self.train_w[0].weight.data = a

    def forward(self, x1, x2):
        # feature1 = self.backbone(x1)
        feature1_w = self.train_w(x1)
        # feature2 = self.backbone(x2)
        feature2_w = self.train_w(x2)

        return feature1_w, feature2_w



if __name__ == '__main__':
    net = ResNet12()
    print(net)

