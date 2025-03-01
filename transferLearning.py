import argparse
import os.path as osp

import math
import minisom
from torchvision.models import ResNet
import numpy as np
import os
import torch
from transferUtils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import load_model
from CIFAR import CIFAR
from FC100 import FC100
from cub import CUB
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12
from basefinetine import BaseFinetine
from torchvision import transforms
from utils import TwoCropTransform

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}

if __name__ == '__main__':
    args = arg_parameter()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    episode = 0

    test_loader, base_class = data_load(args)
    base_model,model_name = basemodel_load(args, base_class)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet12.overall_model().to(device)
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)



    print_epoch = 21
    accAll=[]
    for i in range(print_epoch):
        accAll.append([])


    for j, batch in enumerate(test_loader):
        episode = j
        print('episode:',j,'-------------------------------------------------')
        a = np.eye(640, dtype=float)
        a = torch.from_numpy(np.float32(a)).cuda()
        net.train_w[0].weight.data = a

        data, label = [_.cuda() for _ in batch]
        data = base_model(data)
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]
        img1_1 = data_shot[0].unsqueeze(0)
        if (args.shot == 1):
            img1_2 = data_shot[0].unsqueeze(0)
        else:
            img1_2 = data_shot[5].unsqueeze(0)
        img0_1 = data_shot[0].unsqueeze(0)
        img0_2 = data_shot[1].unsqueeze(0)
        for i in range(p):
            for j in range(i + 1, p):
                if (label[i] == label[j]):
                    img1_1 = torch.cat((img1_1, data_shot[i].unsqueeze(0)), 0)
                    img1_2 = torch.cat((img1_2, data_shot[j].unsqueeze(0)), 0)
                else:
                    img0_1 = torch.cat((img0_1, data_shot[i].unsqueeze(0)), 0)
                    img0_2 = torch.cat((img0_2, data_shot[j].unsqueeze(0)), 0)

        for epoch in range(0, args.max_epoch + 1):

            if epoch==0:
                pass
            else:
                net.train()
                feature1_1,feature1_2 = net(img1_1,img1_2)
                feature0_1, feature0_2 = net(img0_1, img0_2)
                output0 = torch.cosine_similarity(feature0_1,feature0_2)
                output1 = torch.cosine_similarity(feature1_1, feature1_2)
                output0 = ((output0+1)/2).view(-1,1)
                output1 = ((output1+1)/2).view(-1,1)
                label0 = torch.zeros_like(output0)
                label1 = torch.ones_like(output1)
                preds0 = output0>0.5
                preds1 = output1>0.5
                loss0 = nn.BCELoss()(output0,label0)
                if(args.shot==1):
                    loss1 = 0
                else:
                    loss1 = nn.BCELoss()(output1, label1)
                loss = loss1 + loss0
                running_corrects = (preds0 == label0).type(torch.cuda.FloatTensor).mean().item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                print('epoch {}, , loss={:.4f}'
                      .format(epoch, loss.item()))
                print('running_corrects:', running_corrects)
                net.eval()

                    # loss.backward(retain_graph=True)
                    # optimizer.step()
                    # print('epoch {}, , loss={:.4f}'
                    #       .format(epoch, loss.item()))
                    # print('running_corrects:', running_corrects)
                    # net.eval()


            #net.eval()
            with torch.no_grad():
                data_shot,data_query = net(data_shot,data_query)


                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                acc = evaluate(data_query, data_shot, label, args.shot)

                # for i in range(print_epoch):
                #     if(epoch==i*10):
                accAll[epoch].append(acc)


                print("acc={:.4f}".format(acc))



    val = np.asarray(accAll[0])
    print("np.std(val):", np.std(val))
    print("model_name={}".format(model_name))
    print("shot={}______lr={}_____episode={}".format(args.shot,args.lr,episode+1))
    for i in range(print_epoch):
        print("acc"+str(i*10)+"={:.4f} +- {:.4f}".format(np.mean(np.asarray(accAll[i]))
                                                         , 1.96 * (np.std(np.asarray(accAll[0])) / np.sqrt(len(val)))))





