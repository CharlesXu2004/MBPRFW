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
        path = '/home/user2/feifan/save/model/mini/rotation/'

        model_path = path + '62.pth'
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
            #print("batch:" + str(i))
            data, train_label = [_ for _ in batch]
            data = data.type(torch.cuda.FloatTensor)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature_copy = feature.cpu().numpy()

            # feature_copy = pca.transform(feature_copy)
            # feature_copy = maxnorm(feature_copy,mincol)


            #feature_copy = np.where(feature_copy<0,np.power(feature_copy*(-1),0.5)*(-1),np.power(feature_copy,0.5))

            # feature_copy = maxminnorm(feature_copy)
            #feature_copy = np.power(feature_copy[:, ], 0.9)
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


            #print("data_shot",data_shot.shape)

                      #aug__________data_shot
            aug = None
            if aug:
                data_shot = data_shot.cpu()
                data_query = data_query.cpu()
                data_shot_aug = None
                for i in range(p):
                    for j in range(5):
                        rand_b = random.uniform(-0.2, 0.2)
                        rand_c = random.uniform(-0.2, 0.2)
                        rand_s = random.uniform(-0.2, 0.2)
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            #transforms.CenterCrop(84),
                            transforms.ColorJitter(brightness=0.4+rand_b, contrast=0.4+rand_c, saturation=0.4+rand_s),
                            #transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                        ])
                        if data_shot_aug == None:
                            data_shot_aug = transform(data_shot[i]).unsqueeze(0)
                        else:
                            data_shot_aug = torch.cat([data_shot_aug,transform(data_shot[i]).unsqueeze(0)],dim=0)


                data_query_aug = None
                for i in range(len(data_query)):
                    transform = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    if i == 0:
                        data_query_aug = transform(data_query[i]).unsqueeze(0)
                    else:
                        data_query_aug = torch.cat([data_query_aug, transform(data_query[i]).unsqueeze(0)], dim=0)

                data_shot = data_shot_aug.type(torch.cuda.FloatTensor)
                data_query = data_query_aug.type(torch.cuda.FloatTensor)

            label = torch.arange(args.test_way).repeat(args.query).cuda()
            #label = np.array([0,1,2,3,4]*15)
           # label = label.type(torch.cuda.LongTensor)
            #label_sup = torch.arange(args.test_way).repeat(args.shot)
            #label_sup = np.array([0,1,2,3,4] * args.shot)
           # label_sup = label_sup.type(torch.cuda.LongTensor)
            data_query = model(data_query).cpu().numpy()
           #
            data_shot = model(data_shot).cpu().numpy()
            #data_query = F.normalize(data_query,dim=1).cpu().numpy()
            #data_shot = F.normalize(data_shot, dim=1).cpu().numpy()
            # print(data_query.shape)
            # print(data_shot.shape)
            if aug:
                data_shot = data_shot.reshape(-1,5,640)
                data_shot = np.mean(data_shot,axis=1).reshape(-1,640)


            #data_shot = F.normalize(data_shot,dim=1).cpu().numpy()
            #data_query = F.normalize(data_query,dim=1).cpu().numpy()
            #data_shot = maxminnorm(data_shot)

            #data_shot = maxnorm(data_shot,mincol)
            # data_shot = pca.transform(data_shot)
            # data_query = pca.transform(data_query)

            # data_shot = np.where(data_shot < 0, np.power(data_shot * (-1), 0.6)*(-1),
            #                          np.power(data_shot, 0.6))
            # data_query = np.where(data_query < 0, np.power(data_query * (-1), 0.6)*(-1),
            #                          np.power(data_query, 0.6))

            #data_query = maxminnorm(data_query)
            #---- Tukey's transform
            # beta = 0.5
            # data_shot = np.power(data_shot[:, ], beta)
            #data_query = np.power(data_query[:, ], beta)
            # data_query = maxnorm(data_query,mincol)

            for i in range(len(data_shot)):
                data_shot[i]= distribution_calibration(data_shot[i], feature_matrix, k=4)
            #print(data_shot.shape)


            if args.shot == 5:
                data_shot = data_shot.reshape(5, 5, 640)
                data_shot = np.mean(data_shot, axis=0)
            # base_feature = np.load('base_feature.npy')
            # base_feature = np.mean(base_feature, axis=0)
            # acc = model.evaluate_test_gen(data_query, data_shot, label, base_feature)
            #acc = model.evaluate_test(data_query, data_shot, label, features)
            #acc = model.evaluate(data_query, data_shot, label)
            acc = model.evaluate_eulidean_free_lunch(data_query, data_shot, label)
            #acc = model.train_LR(data_shot,label_sup,data_query,label)
            #acc = model.train_svm(data_shot,label_sup,data_query,label)

                #classifer= LogisticRegression(max_iter=1000).fit(X=data_shot, y=label_sup)
                #predicts=classifer.predict(data_query)
                #acc=np.mean(predicts==label)
                #acc = model.evaluate_Euclidean(data_query, data_shot, label)
            # acc = model.finetine_loop(data_query, data_shot, base_feature)

            val.append(acc)
            print(acc)

            # print(val)
    val = np.asarray(val)
    print("np.std(val):",np.std(val))
    print(len(val))
    print("acc={:.4f} +- {:.4f}".format(np.mean(val), 1.96 * (np.std(val) / np.sqrt(len(val)))))





