import argparse
import os

import torch
from torch.utils.data import DataLoader
#from backbones import backbone
from backbones import resnet12
from basefinetine import BaseFinetine
from CIFAR import CIFAR
from FC100 import FC100
from cub import CUB
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from utils import load_model
import torch.nn.functional as F

def arg_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[350, 400, 440, 460, 480], help='Decrease learning rate at these epochs.')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='mini')
    parser.add_argument('--model', default='basefinetine')
    parser.add_argument('--lr', type=float, default=0.002)  # 0.025
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--projection', type=bool, default=True)
    # parser.add_argument('--aug', default='protonet')
    parser.add_argument('--save-epoch', type=int, default=2)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--gpu', default='1')
    return parser.parse_args()
def data_load(args):
    image_size = 84
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

    test_sampler = CategoriesSampler(testset.label, 200,
                                     args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                             num_workers=1, pin_memory=True)
    return test_loader,base_class
def data_load_train(args):
    image_size = 84
    if args.dataset == 'mini':
        testset = MiniImageNet('test', image_size, False)
        base_class = 64
        trainset = MiniImageNet('train', image_size, False)
        valset = MiniImageNet('val', image_size, False)
    elif args.dataset == 'CUB':
        testset = CUB('test', image_size, False)
        base_class = 100
        trainset = CUB('train', image_size, False)
        valset = CUB('val', image_size, False)

    elif args.dataset == 'FC100':
        testset = FC100('test', image_size, False)
        trainset = FC100('train', image_size, False)
        valset = FC100('val', image_size, False)
        base_class = 60
    elif args.dataset == 'CIFAR':
        testset = CIFAR('test', image_size, False)
        trainset = CIFAR('train', image_size, False)
        valset = CIFAR('val', image_size, False)
        base_class = 64

    test_sampler = CategoriesSampler(testset.label, 200,
                                     args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                             num_workers=1, pin_memory=True)
    train_loader = DataLoader(dataset=trainset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=1, pin_memory=True)
    return test_loader,base_class,train_loader,val_loader
def basemodel_load(args,base_class):
    backbone = resnet12.ResNet12()
    path = '/home/user2/feifan/save/model/'
    base_model = BaseFinetine(backbone, base_class, args.test_way)
    model_name = 'mini/rotation/maxacc_ok.pth'
    model_path = path + model_name
    if os.path.exists(path):
        load_model(base_model, model_path)
        print("load pretrain model successfully")
    base_model.cuda()
    base_model.eval()
    return base_model,model_name
def evaluate(data_query, data_shot, label,n):
    if n == 5:
        data_shot = data_shot.reshape(5, 5, 640)
        data_shot = torch.mean(data_shot, dim=0)

    data_query = F.normalize(data_query, dim=1)
    data_shot = F.normalize(data_shot, dim=1)
    score = torch.matmul(data_query, data_shot.T)
    #print("score:",score)
    pred = torch.argmax(score, dim=1)

    acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return acc
