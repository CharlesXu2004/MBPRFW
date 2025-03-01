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

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='CUB')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--model', default='basefinetine')
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    print(vars(args))



    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

    test_sampler = CategoriesSampler(testset.label, 200,
                                    args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=1, pin_memory=True)


    backbone = model_dict[args.backbone]()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseFinetine(backbone, base_class, args.test_way)
    net = resnet12.overall_model().to(device)



    #path = '/home/user2/feifan/save/model/mini/base/'
    path = '/home/user2/feifan/save/model/cub/rotation/'
    path_net = '/home/user2/feifan-som/save/basefinetineResNet12cub/'
    model_net = path_net + 'maxacc_net_base.pth'
    model_path = path +'62_ok.pth'
    if os.path.exists(path):
        load_model(model, model_path)
        load_model(net,model_net)
        print("load pretrain model successfully")

    model.cuda()


    model.eval()
    val = []
    acc = 0
    val_base = []
    acc_base= 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            print("batch:" + str(i))
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]


            label = torch.arange(args.test_way).repeat(args.query).cuda()


            #acc = model.evaluate_Euclidean(data_query, data_shot, label)
            acc,acc_base = model.evaluate_net(data_query,data_shot,label,args.shot,net)
            #acc = model.cluster_evaluate(data,data_query, data_shot, label, args.shot)
            #acc = model.cluster5_evaluate(data, data_query, data_shot, label, args.shot)

            val.append(acc)
            print(acc)
            val_base.append(acc_base)
            print(acc_base)

    val = np.asarray(val)
    print("np.std(val):",np.std(val))
    print(len(val))
    print("acc={:.4f} +- {:.4f}".format(np.mean(val), 1.96 * (np.std(val) / np.sqrt(len(val)))))

    val = np.asarray(val_base)
    print("np.std(val):", np.std(val))
    print(len(val))
    print("acc_base={:.4f} +- {:.4f}".format(np.mean(val), 1.96 * (np.std(val) / np.sqrt(len(val)))))





