import argparse
import os.path as osp

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_model

from mini_imagenet import MiniImageNet
from sklearn.decomposition import PCA
from samplers import CategoriesSampler

from backbones import backbone
from backbones import resnet12

from basefinetine import BaseFinetine

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}
def distribution_calibration(query, base_means,base_cov,k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index],axis=0)+alpha
    print("cov:",calibrated_cov.shape)
    return calibrated_mean,calibrated_cov

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
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
        #testset = CUB('test', image_size, False)
        base_class = 50
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)    #LiFeiFan

    test_sampler = CategoriesSampler(testset.label, 5000,
                                    args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=8, pin_memory=True)

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
        path = 'save/' + args.model + args.backbone + '/'
        model_path = path + 'maxacc.pth'
        if os.path.exists(path):
            load_model(model, model_path)
            print("load pretrain model successfully")
    else:
        path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' +  '/'
        model_path = path + 'maxacc.pth'
        #model_path = path +'90.pth'
        model.load_state_dict(torch.load(model_path))
        print("load model successfully")
    model.cuda()

    # for epoch in range(1, args.max_epoch + 1):
    model.eval()
    val = []

    #base_mean
    classes_count = np.zeros([64, ])
    feature_matrix = np.zeros([64, 640])
    cov_matrix = np.zeros([64,640,640])
    all_matrix = np.zeros([len(train_loader)*32,640])
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            print("batch:" + str(i))
            data, train_label = [_ for _ in batch]
            data = data.type(torch.cuda.FloatTensor)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature_copy = feature.cpu().numpy()

            #feature_copy = np.where(feature_copy<0,np.power(feature_copy*(-1),0.5)*(-1),np.power(feature_copy,0.5))
            feature_copy = np.power(feature_copy[:, ], 0.5)

            for index in range(32):
                feature_matrix[train_label[index]] += feature_copy[index]
                classes_count[train_label[index]] += 1

        for i in range(64):
            feature_matrix[i] /= classes_count[i]
            cov_matrix[i] = np.cov(feature_matrix[i].T)

        print(classes_count)

        #features = np.mean(feature_matrix, axis=0)




        for i, batch in enumerate(test_loader):
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            data_query = model(data_query).cpu().numpy()
            data_shot = model(data_shot).cpu().numpy()
            #data_shot = F.normalize(data_shot,dim=1).cpu().numpy()
            #data_query = F.normalize(data_query,dim=1).cpu().numpy()
            #data_shot = maxminnorm(data_shot)
            # data_shot = np.where(data_shot < 0, np.power(data_shot * (-1), 0.5)*(-1),
            #                          np.power(data_shot, 0.5))
            #data_shot = maxnorm(data_shot,mincol)
            #data_query = pca.transform(data_query)

            #data_query = maxminnorm(data_query)
            # ---- Tukey's transform
            # beta = 0.5
            # data_shot = np.power(data_shot[:, ], beta)
            #data_query = maxnorm(data_query,mincol)
            #data_query = np.power(data_query[:, ], beta)
            samplers_data = []
            samplers_label = []
            num_samples = 640
            for i in range(len(data_shot)):
                mean ,cov= distribution_calibration(data_shot[i],feature_matrix,cov_matrix,k=5)
                samplers_data.append([data_shot[i]])
                samplers_data.append(np.random.multivariate_normal(mean=mean,cov=cov,size=num_samples))
                #samplers_label.extend([label[i]]*(num_samples+1))
            samplers_data = np.concatenate(samplers_data[:]).reshape(args.shot * args.test_way * (num_samples+1),-1)
            X_aug = samplers_data
            #Y_aug = np.concatenate([samplers_label])

            kappa = 0.1
            ppp = np.dot(np.linalg.inv(np.dot(X_aug,X_aug.T) + kappa * np.eye(len(X_aug))),X_aug)
                #acc = model.evaluate(data_query, data_shot, label)
            #acc = model.evaluate_eulidean_free_lunch(data_query, data_shot, label)
                #acc = model.evaluate_Euclidean(data_query, data_shot, label)
            acc = model.test_crc(X_aug,ppp,data_query,label)
            val.append(acc)
            print(acc)

            # print(val)
    val = np.asarray(val)
    print("acc={:.4f} +- {:.4f}".format(np.mean(val), 1.96 * (np.std(val) / np.sqrt(len(val)))))