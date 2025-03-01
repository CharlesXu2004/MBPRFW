import argparse
import os.path as osp
from torchvision import transforms
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CIFAR import CIFAR
from utils import load_model

from mini_imagenet import MiniImageNet
from sklearn.decomposition import PCA
from samplers import CategoriesSampler
from cub import CUB
from backbones import backbone
from backbones import resnet12

from basefinetine import BaseFinetine
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=300)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='CIFAR')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--model', default='basefinetine')
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    print(vars(args))



    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if 'Conv' in args.backbone or 'ResNet12' in args.backbone:
        image_size = 84

    else:
        image_size = 224

    if args.dataset == 'mini':
        testset = MiniImageNet('test', image_size, False)
        trainset = MiniImageNet('train', image_size, False)   #LiFeiFan
        base_class = 64
    elif args.dataset == 'CIFAR':
        testset = CIFAR('test', image_size, False)
        base_class = 64

    test_sampler = CategoriesSampler(testset.label, 2000,
                                    args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=1, pin_memory=True)

    if args.model == 'relationnet':
        if args.backbone == 'Conv4':
            backbone = backbone.Conv4NP()
        else:
            backbone = model_dict[args.backbone]()
    elif args.model == 'protonet':
        backbone = model_dict[args.backbone]()

    else:
        backbone = model_dict[args.backbone]()
        model = BaseFinetine(backbone, base_class, args.test_way)


    if args.pretrain:
        #path = 'save/' + args.model + args.backbone + '/'
        #path = '/home/rugu/桌面/model/mini/base/'
        #path = '/home/rugu/桌面/model/mini/rotation/'
        #path ='/home/rugu/桌面/model/cifar/base/'
        #path = '/home/rugu/桌面/model/cifar/rotation/'
        #path = '/home/rugu/桌面/cifar_model/'
        path = '/home/rugu/桌面/cifar/'
        model_path = path +'maxaccGen0_CE.pth'
        if os.path.exists(path):
            load_model(model, model_path)
            print("load pretrain model successfully")
    else:
        path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' +  '/'
        #model_path = path + 'maxacc.pth'
        model_path = path +'98.pth'
        model.load_state_dict(torch.load(model_path))
        print("load model successfully")
    model.cuda()

    # for epoch in range(1, args.max_epoch + 1):
    model.eval()
    val = []


    with torch.no_grad():


        for i, batch in enumerate(test_loader):
            print("batch:" + str(i))
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]
            data_query = model(data_query).cpu().numpy()



            tsne = TSNE(n_components=2, init='pca', random_state=0)

            result = tsne.fit_transform(data_query)

            x_min, x_max = np.min(result, 0), np.max(result, 0)
            result = (result - x_min) / (x_max - x_min)
            print("sult:",result[0])


            fig = plt.figure(dpi=100)
            plt.scatter(result[0::5, 0], result[0::5, 1], marker="o", s=15,
                        c=plt.cm.Set1(1.0 / 8.0), alpha=1)#blue

            plt.scatter(result[1::5, 0], result[1::5, 1], marker="o", s=15,
                        c=plt.cm.Set1(2.0 / 8.0), alpha=1)#green

            plt.scatter(result[2::5, 0], result[2::5, 1], marker="o", s=15,
                        c=plt.cm.Set1(3.0 / 8.0), alpha=1)#purple
            plt.scatter(result[3::5, 0], result[3::5, 1], marker="o", s=15,
                        c=plt.cm.Set1(4.0 / 8.0), alpha=1)#yellow
            plt.scatter(result[4::5, 0], result[4::5, 1], marker="o", s=15,
                        c='r', alpha=1)

            # plt.scatter(result[0::5, 0], result[0::5, 1], marker="^", s=20,
            #             c='none', alpha=1, edgecolors='k')
            #
            # plt.scatter(result[1::5, 0], result[1::5, 1], marker="*", s=20,
            #             c='none', alpha=1, edgecolors='k')
            #
            # plt.scatter(result[2::5, 0], result[2::5, 1], marker="o", s=20,
            #             c='none', alpha=1, edgecolors='k')
            plt.xlabel("")
            plt.ylabel("")

            #plt.legend(loc='upper right')
            plt.title('Gen0_CE',y=-0.13)
            #plt.title("HL",y=-0.1)
            plt.show()
            break







