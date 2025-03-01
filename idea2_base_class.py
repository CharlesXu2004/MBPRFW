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

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[350, 400, 440, 460, 480], help='Decrease learning rate at these epochs.')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='CUB')
    parser.add_argument('--model', default='basefinetine')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--projection', type=bool, default=True)
    # parser.add_argument('--aug', default='protonet')
    parser.add_argument('--save-epoch', type=int, default=2)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    image_size = 84
    batch_size_ = 64
    if args.dataset =='mini':
        trainset = MiniImageNet('train', image_size, False)
        valset = MiniImageNet('val', image_size, False)
        base_class = 64

    elif args.dataset == 'CUB':
        trainset = CUB('train',image_size,False)
        valset = CUB('val',image_size,False)
        base_class = 100
    elif args.dataset == 'FC100':
        trainset = FC100('train', image_size, False)
        valset = FC100('val', image_size, False)
        print("CLASS: ", trainset.num_classes)
        base_class = 60
    elif args.dataset == 'CIFAR':
        trainset = CIFAR('train', image_size, False)
        valset = CIFAR('val', image_size, False)
        base_class = 64
        print("CLASS: ", trainset.num_classes,image_size)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size_, shuffle=True, num_workers=1, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=1, pin_memory=True)

    backbone = resnet12.ResNet12()
    path = '/home/user2/feifan/save/model/'
    base_model = BaseFinetine(backbone, base_class, args.test_way)
    model_name = 'cub/rotation/62_ok.pth'
    model_path = path + model_name
    if os.path.exists(path):
        load_model(base_model, model_path)
        print("load pretrain model successfully")
    base_model.cuda()
    base_model.eval()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet12.overall_model().to(device)
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    result = {}
    result['acc'] = 0
    for epoch in range(1, args.max_epoch + 1):
        net.train()
        for j, batch in enumerate(train_loader):
            data,label = [_.cuda() for _ in batch]
            data = base_model(data)
            n = len(data)
            img0 = data[0].unsqueeze(0)
            img1 = data[1].unsqueeze(0)
            if label[0]==label[1]:
                label_train = torch.tensor(1).unsqueeze(0).cuda()
            else:
                label_train = torch.tensor(0).unsqueeze(0).cuda()

            for i in range(n):
                for j in range(i + 1, n):
                    img0 = torch.cat((img0, data[i].unsqueeze(0)), 0)
                    img1 = torch.cat((img1, data[j].unsqueeze(0)), 0)
                    if (label[i] == label[j]):
                        label_train = torch.cat((label_train, torch.tensor(1).unsqueeze(0).cuda()), 0)
                    else:
                        label_train = torch.cat((label_train, torch.tensor(0).unsqueeze(0).cuda()), 0)
            feature0, feature1 = net(img0, img1)
            output = torch.cosine_similarity(feature0, feature1)
            output = ((output + 1) / 2).view(-1, 1).to(torch.float32)
            label_train = label_train.reshape(-1,1).to(torch.float32)



            preds = output > 0.5

            loss = nn.BCELoss()(output, label_train)
            running_corrects = (preds == label_train).type(torch.cuda.FloatTensor).mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch {}, , loss={:.4f}'
                  .format(epoch, loss.item()))
            print('running_corrects:', running_corrects)

        net.eval()
        val = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                data_shot = base_model(data_shot)

                data_query = base_model(data_query)
                data_query,data_shot = net(data_query,data_shot)

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                acc =evaluate(data_query, data_shot, label, args.shot)
                val += acc

        print("epoch {}, val_acc={:.4f}".format(epoch, val / len(val_loader)))
        # with open("ACC_VAL.txt", "a")as f:
        #     f.write("epoch:" + str(epoch) + "  " + ''.join(str(float(val / len(val_loader)))) + '\r\n')

        if (val > result['acc']):
            result['acc'] = val
            path = args.save_path + args.model + args.backbone + 'cub' + '/'
            if not os.path.exists(path):
                os.mkdir(path)
            print('-----------------------------------maxacc-----------------------------------------')
            torch.save(net.state_dict(), osp.join(path, 'maxacc_net' + '.pth'))

        if True:
            path = args.save_path + args.model + args.backbone + 'cub' + '/'
            torch.save(net.state_dict(), osp.join(path, str(epoch) +'_net'+'.pth'))


































