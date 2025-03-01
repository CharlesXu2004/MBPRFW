import argparse
import os.path as osp

from torchvision.models import ResNet
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_model

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
from methods.basefinetine import BaseFinetine

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=120)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[350, 400, 440, 460, 480], help='Decrease learning rate at these epochs.')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--model', default='basefinetine')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--gamma', type=float, default=0.2)
    # parser.add_argument('--aug', default='protonet')
    parser.add_argument('--save-epoch', type=int, default=10)
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

    trainset = MiniImageNet('train', image_size, False)
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    num_classes = trainset.num_classes

    if args.model == 'relationnet':
        if args.backbone == 'Conv4':
            backbone = backbone.Conv4NP()
        else:
            backbone = model_dict[args.backbone]()
        model = RelationNet(backbone, args.train_way, args.test_way, args.shot, args.hidden_size, 'local')
    elif args.model == 'protonet':
        backbone = model_dict[args.backbone]()
        model = ProtoNet(backbone, args.train_way, args.test_way, args.shot, args.temperature)
    else:
        backbone = model_dict[args.backbone]()
        model = BaseFinetine(backbone, num_classes, args.test_way)

    # print(model)
    # valset = MiniImageNet('val', image_size, False)
    # # val_sampler = CategoriesSampler(valset.label, 400,
    # #                                 args.test_way, args.shot + args.query)
    # val_sampler = CategoriesSampler(valset.label, 200,
    #                                 args.test_way, args.shot + args.query)
    # val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
    #                         num_workers=8, pin_memory=True)
    # if args.model == 'protonet':
    #     model.set_pretrain(trainset.num_classes)
    # path = 'pretrain/' + 'protonet' + args.backbone + '/'
    # model_path = path + 'maxacc.pth'
    path = 'contrastive/' + args.model + args.backbone + '/'
    model_path = path + 'maxacc.pth'
    if os.path.exists(path):
        load_model(model, model_path)
        print("load pretrain model successfully")

    # path = args.save_path + args.model + args.backbone + str(args.test_way) + 'way' + str(args.shot) + 'shot' + '/'
    # model_path = path + 'maxacc62.9.pth'
    # model.load_state_dict(torch.load(model_path))
    # print("load model successfully")
    # print(trainset.num_classes)
    model.cuda()
    classes_count = np.zeros([num_classes,])
    # print(classes_count)
    feature_matrix = np.zeros([num_classes, model.model.final_feat_dim])
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            print("batch:" + str(i))
            data, train_label = [_ for _ in batch]
            # print(train_label)
            feature = model(data.cuda())
            feature_copy = feature.cpu().numpy()
            for index in range(32):
                feature_matrix[train_label[index]] += feature_copy[index]
                classes_count[train_label[index]] += 1

    for i in range(num_classes):
        feature_matrix[i] /= classes_count[i]

    print(feature_matrix.shape)
    print(classes_count)
    np.save('contrastive_base_feature', feature_matrix)
    # feature_matrix = []
    # with torch.no_grad():
    #     for i, batch in enumerate(train_loader):
    #         print("batch:" + str(i))
    #         data, train_label = [_ for _ in batch]
    #         # print(train_label)
    #         feature = model(data.cuda())
    #         feature_copy = feature.cpu().numpy()
    #         for index in range(32):
    #             feature_matrix.append(feature_copy[index])
    #
    # feature_matrix = np.array(feature_matrix)
    # print(feature_matrix.shape)
    # np.save('all_base_feature', feature_matrix)

