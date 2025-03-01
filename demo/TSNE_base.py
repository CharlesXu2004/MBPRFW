import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from CIFAR import CIFAR
from FC100 import FC100
from cub import CUB
from utils import load_model
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12
from basefinetine import BaseFinetine
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    parser.add_argument('--query', type=int, default=200)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='mini')
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
        base_class = 64
    elif args.dataset == 'CUB':
        testset = CUB('test', image_size, False)
        base_class = 100
    elif args.dataset == 'FC100':
        testset = FC100('test', image_size, False)
        base_class = 60
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
    # path = '/home/rugu/桌面/'
    if args.pretrain:
        #path = 'save/' + args.model + args.backbone + '/'
        #path = 'save/' + args.model + args.backbone + 'FC100_0.8' + '/'
        #path = '/media/rugu/新加卷/model/CIFAR_base/'
        path = '/home/rugu/桌面/10.19_eqloss_0.8单层rotation/'
        #path = '/home/rugu/桌面/FC100_0.8eq_多层/'
        #path = '/home/rugu/桌面/CUB_rotation_多层_0.8eq/'
        #path = '/home/rugu/桌面/'
        # path = '/home/rugu/桌面/model/'
        # path = '/media/rugu/新加卷/'
        model_path = path +'maxacc.pth'
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


    model.eval()
    val = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            print("batch:" + str(i))
            data, labelori = [_.cuda() for _ in batch]
            # p = args.shot * args.test_way
            # data_shot, data_query = data[:p], data[p:]



            data = model(data).cpu().numpy()
            label = np.array([0,1,2,3,4]*201)


            tsne = TSNE(n_components=2,init='pca',random_state=0)
            result = tsne.fit_transform(data)
            x_min, x_max = np.min(result,0)-1,np.max(result,0)+1
            result = (result - x_min)/(x_max - x_min)
            fig = plt.figure(dpi=200)
            for i in range(result.shape[0]):
                if i<5:
                    # plt.text(result[i,0],result[i,1],'o',color=plt.cm.Set1(label[i]/8.),fontdict={'weight': 'bold', 'size' : 15})
                    plt.scatter(result[i, 0], result[i, 1], marker='<', s=50,
                                c=plt.cm.Set1(label[i] / 8.))
                else:
                    #plt.text(result[i,0],result[i,1],'o',color=plt.cm.Set1(label[i]/8.),fontdict={'weight': 'bold', 'size' : 15})
                    plt.scatter(result[i, 0], result[i, 1], marker='o', s=6,
                                c=plt.cm.Set1(label[i] / 8.))
            plt.xticks([])
            plt.yticks([])
            plt.title('Baseline',y=-0.1)
            plt.show()
            break









