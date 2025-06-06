import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from args import get_parser_test
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CIFAR, FC100, CUB, MiniImageNet
from utils import load_model
from sklearn.decomposition import PCA
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12
from BaseModel import BaseModel
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
aug = True

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12
}


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


def maxnorm(array, mincol):
    mincols=mincol
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])
    return t


if __name__ == '__main__':
    parser = get_parser_test()
    args = parser.parse_args()

    args.max_epoch = 600
    image_size = 84
    testset = FC100('test', image_size, False)
    trainset = FC100('train', image_size, False)
    base_class = 60

    test_sampler = CategoriesSampler(testset.label, 2000, args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=1, pin_memory=True)
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)

    backbone = model_dict[args.backbone]()
    model = BaseModel(backbone, base_class, args.test_way)


    if args.pretrain:
        if os.path.exists(args.model_path):
            load_model(model, args.model_path)
            print("load pretrain model successfully")
    else:
        path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' +  '/'
        #model_path = path + 'maxacc.pth'
        model_path = path +'100.pth'
        model.load_state_dict(torch.load(model_path))
        print("load model successfully")

    model.cuda()
    model.eval()

    accuracy = []
    classes_count = np.zeros([base_class,])
    feature_matrix = np.zeros([base_class, 640])
    #all_matrix = np.zeros([len(train_loader)*32,640])
    with torch.no_grad():
        # for i, batch in enumerate(train_loader):
        #     data, train_label = [_ for _ in batch]
        #     data = data.type(torch.cuda.FloatTensor)
        #     feature = model(data)
        #     feature_copy = feature.cpu().numpy()
        #     # feature_copy = np.power(feature_copy[:, ], 0.5)
        #     for index in range(32):
        #         all_matrix[i * 32 + index] = feature_copy[index]
        # pca = PCA(n_components=640)
        # pca.fit(all_matrix)
        # mincol = pca.transform(all_matrix).min(axis=0)
        for i, batch in enumerate(train_loader):
            data, train_label = [_ for _ in batch]
            data = data.type(torch.cuda.FloatTensor)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature_copy = feature.cpu().numpy()

            n = len(train_label)
            for index in range(n):
                feature_matrix[train_label[index]] += feature_copy[index]
                classes_count[train_label[index]] += 1

        for i in range(base_class):
            feature_matrix[i] /= classes_count[i]

        for i, batch in enumerate(test_loader):
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            if aug ==True:
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
                        #transforms.ToPILImage(),
                        #transforms.CenterCrop(84),
                        #transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    if i == 0:
                        data_query_aug = transform(data_query[i]).unsqueeze(0)
                    else:
                        data_query_aug = torch.cat([data_query_aug, transform(data_query[i]).unsqueeze(0)], dim=0)

                data_shot = data_shot_aug.type(torch.cuda.FloatTensor)
                data_query = data_query_aug.type(torch.cuda.FloatTensor)
            
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            data_query = model(data_query).cpu().numpy()
            data_shot = model(data_shot).cpu().numpy()
            if aug:
                data_shot = data_shot.reshape(-1,5,640)
                data_shot = np.mean(data_shot,axis=1).reshape(-1,640)

            #---- Tukey's transform
            # beta = 0.5
            # data_shot = np.power(data_shot[:, ], beta)
            # data_query = np.power(data_query[:, ], beta)
            # data_query = maxnorm(data_query, mincol)

            for i in range(len(data_shot)):
                data_shot[i]= distribution_calibration(data_shot[i], feature_matrix, k=4)

            if args.shot == 5:
                data_shot = data_shot.reshape(5, 5, 640)
                data_shot = np.mean(data_shot, axis=0)

            # base_feature = np.load('base_feature.npy')
            # base_feature = np.mean(base_feature, axis=0)
            # acc = model.evaluate_test_gen(data_query, data_shot, label, base_feature)
            # acc = model.evaluate_test(data_query, data_shot, label, features)
            # acc = model.evaluate(data_query, data_shot, label)
            acc = model.evaluate_eulidean_free_lunch(data_query, data_shot, label)
            # acc = model.evaluate_Euclidean(data_query, data_shot, label)
            # acc = model.finetune_loop(data_query, data_shot, base_feature)

            accuracy.append(acc)

    accuracy = np.asarray(accuracy)
    print("acc={:.4f} +- {:.4f}".format(np.mean(accuracy), 1.96 * (np.std(accuracy) / np.sqrt(len(accuracy)))))