import argparse
import os.path as osp
from torchvision import transforms
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}
def distribution_calibration(query, base_means,k):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    return calibrated_mean

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t
def maxnorm(array,mincol):
    mincols=mincol
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=300)
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
        trainset = MiniImageNet('train', image_size, False)   #LiFeiFan
        base_class = 64
    else:
        testset = CUB('test', image_size, False)
        base_class = 100
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)    #LiFeiFan

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
        path = '/home/rugu/桌面/10.19_eqloss_0.8单层rotation/'
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

    # for epoch in range(1, args.max_epoch + 1):
    model.eval()
    val = []

    #base_mean
    classes_count = np.zeros([64, ])
    feature_matrix = np.zeros([64, 640])
    #cov_matrix = np.zeros([64,640])
    #all_matrix = np.zeros([len(train_loader)*32,640])
    with torch.no_grad():
        # for i, batch in enumerate(train_loader):
        #     print("batch:"+str(i))
        #     data, train_label = [_ for _ in batch]
        #     data = data.type(torch.cuda.FloatTensor)
        #     feature = model(data)
        #     feature_copy = feature.cpu().numpy()
        #     #feature_copy = np.power(feature_copy[:, ], 0.5)
        #     for index in range(32):
        #         all_matrix[i * 32 + index] = feature_copy[index]
        # pca = PCA(n_components=640)
        # pca.fit(all_matrix)
        #mincol = pca.transform(all_matrix).min(axis=0)
        # print("pca is ok !!!")
        # print(pca.explained_variance_ratio_)
        for i, batch in enumerate(train_loader):
            print("batch:" + str(i))
            data, train_label = [_ for _ in batch]
            data = data.type(torch.cuda.FloatTensor)
            feature = model(data)
            #feature = F.normalize(feature, dim=1)
            feature_copy = feature.cpu().numpy()

            # feature_copy = pca.transform(feature_copy)
            # feature_copy = maxnorm(feature_copy,mincol)


            #feature_copy = np.where(feature_copy<0,np.power(feature_copy*(-1),0.5)*(-1),np.power(feature_copy,0.5))

            # feature_copy = maxminnorm(feature_copy)
            feature_copy = np.power(feature_copy[:, ], 0.5)
            # feature_copy = np.where(feature_copy<0,0,feature_copy)
            # feature_copy = np.power(feature_copy[:,],2)

            for index in range(32):
                feature_matrix[train_label[index]] += feature_copy[index]
                classes_count[train_label[index]] += 1

        for i in range(64):
            feature_matrix[i] /= classes_count[i]
            #cov_matrix[i] = np.cov(feature_matrix[i].T)
        #
        # print(classes_count)
        #
        # #features = np.mean(feature_matrix, axis=0)




        for i, batch in enumerate(test_loader):
            print("batch:" + str(i))
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]
            data_query = model(data_query).cpu().numpy()
            data_shot = model(data_shot).cpu().numpy()
            shot = np.zeros_like(data_shot)
            shot = data_shot+shot



            for i in range(len(data_shot)):
                data_shot[i]= distribution_calibration(data_shot[i], feature_matrix, k=5)


            if args.shot == 5:
                data_shot = data_shot.reshape(5, 5, 640)
                shot = shot.reshape(5,5,640)
                shot = np.mean(shot,axis=0)
                data_shot = np.mean(data_shot, axis=0)

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            temp = np.concatenate((data_shot,shot,data_query), axis=0)


            result = tsne.fit_transform(temp)

            x_min, x_max = np.min(result, 0), np.max(result, 0)
            print("x_min",x_min,x_max)
            result = result[::5,:]
            real = np.mean(result,axis=0)
            real = np.expand_dims(real,axis=0)
            result = np.concatenate((real,result),axis=0)
            result = (result - x_min) / (x_max - x_min)


            fig = plt.figure(dpi=200)
            # for i in range(shot_aug.shape[0]):
            #     plt.scatter(shot_aug[i, 0], shot_aug[i, 1],marker='1', s=20,
            #                 c=plt.cm.Set1(label_shot[i] / 8.))
            for i in range(result.shape[0]):
                    if i==0:
                        plt.scatter(result[i, 0], result[i, 1], marker="p", s=200,
                                    c='b',alpha=0.5,label='real')
                    elif i==1:
                        plt.scatter(result[i, 0], result[i, 1], marker="^", s=200,
                                    c='r', alpha=1,label='GPC')
                    elif i==2:
                        plt.scatter(result[i, 0], result[i, 1], marker="s", s=200,
                                                    c='c',alpha=0.5,label='PC')
                    else:
                        plt.scatter(result[i, 0], result[i, 1], marker='o', s=10,
                                    c='g',alpha=0.5)
                        # elif i>=5 and i<10:
                    #     plt.scatter(result[i, 0], result[i, 1], marker="s", s=50,
                    #                 c=plt.cm.Set1(label[i] / 8.),alpha=0.5)
                    # else:
                    #     plt.scatter(result[i, 0], result[i, 1], marker='o', s=10,
                    #                 c=plt.cm.Set1(label[i] / 8.),alpha=0.5)
            # plt.xticks([])
            # plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
            plt.legend(loc='upper right')
            plt.title('mini',y=-0.1)
            plt.show()
            break







