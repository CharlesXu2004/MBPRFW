import argparse
import os.path as osp

from torchvision.models import ResNet
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

from CIFAR import CIFAR
from FC100 import FC100
from cub import CUB
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12
from basefinetine import BaseFinetine
#from methods.protonet import ProtoNet
#from methods.relationnet import RelationNet
#from methods.basefinetine import BaseFinetine
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
    parser.add_argument('--max-epoch', type=int, default=150)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[350, 400, 440, 460, 480], help='Decrease learning rate at these epochs.')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset',default='CIFAR')
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
    # print(model)
    if 'Conv' in args.backbone or 'ResNet12' in args.backbone:
        image_size = 84
    else:
        image_size = 224

    batch_size_ = 16
    if args.dataset =='mini':
        trainset = MiniImageNet('train', image_size, True)
        valset = MiniImageNet('val', image_size, False)

    elif args.dataset == 'CUB':
        trainset = CUB('train',image_size,True)
        valset = CUB('val',image_size,False)
    elif args.dataset == 'FC100':
        trainset = FC100('train', image_size, True)
        valset = FC100('val', image_size, False)
        print("CLASS: ", trainset.num_classes)
    elif args.dataset == 'CIFAR':
        trainset = CIFAR('train', image_size, True)
        valset = CIFAR('val', image_size, False)
        print("CLASS: ", trainset.num_classes,image_size)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size_, shuffle=True, num_workers=1, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=1, pin_memory=True)


    backbone = model_dict[args.backbone]()
    model = BaseFinetine(backbone, trainset.num_classes, args.test_way)

    model.cuda()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    result = {}
    result['acc'] = 0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        for j, batch in enumerate(train_loader):

            data, train_label = [_.cuda() for _ in batch]
            batch_size_ = len(train_label)


            #----------rotation--------
            x_90 = data.transpose(2,3).flip(2)

            x_180 = data.flip(2).flip(3)

            x_270 = data.flip(2).transpose(2,3)


            data_rotation = torch.cat((data,x_90,x_180,x_270),0)
            label_aug = torch.arange(4).repeat(batch_size_).reshape(batch_size_,-1).t().reshape(-1).cuda()


            label = train_label.repeat(4)
            loss, acc = model.train_rotation_crop(data_rotation,label,label_aug)

            #loss, acc = model.train_loop(data, train_label)
            #loss ,acc,acc_eq ,en_loss,eq_loss= model.train_loop_eq(data,train_label)

            #loss, acc = model.train_loop_trans(data, train_label,data_trans,outs,label_outs)
            #loss = model.train_contrastive(data, train_label, args.temperature)

                #print("rotation:",acc_eq)


            print('epoch {}, train {}/{}, loss={:.4f}'
                  .format(epoch, j, len(train_loader), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #n = len(train_loader)
        # with open("loss.txt", "a")as f:
        #     f.write("epoch" + str(epoch) + " :"+" loss: "+''.join(str(float(loss_txt/n))) +"  en_loss:  " +''.join(str(float(en_loss_txt/n))) + "  eq_loss:   "+ ''.join(str(float(eq_loss_txt/n)))+ '\r\n')
        # with open("ACC.txt", "a")as f:
        #     f.write("epoch:" + str(epoch) + " acc: " + ''.join(str(float(acc_train_txt / n))) + " acc_eq: " + ''.join(str(float(acc_eq_txt / n)))  + '\r\n')

        model.eval()

        val = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                # if args.model == 'protonet' or args.model == 'basefinetine':7
                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                # else:
                #     y = torch.from_numpy(np.tile(range(args.test_way), args.query))
                #     label = torch.zeros((len(y), args.test_way)).scatter_(1, y.unsqueeze(1), 1).cuda()
                if args.model == 'protonet':
                    loss, acc = model.set_forward_loss(data_query, data_shot, label)
                else:
                    acc = model.evaluate(data_query, data_shot, label,1)
                val += acc

        print("epoch {}, acc={:.4f}".format(epoch, val / len(val_loader)))
        # with open("ACC_VAL.txt", "a")as f:
        #     f.write("epoch:" + str(epoch) + "  " + ''.join(str(float(val / len(val_loader)))) + '\r\n')

        if (val > result['acc']):
            result['acc'] = val
            path = args.save_path + args.model + args.backbone + 'CIFAR_Rotation_4_0.8' + '/'
            if not os.path.exists(path):
                os.mkdir(path)
            print('-----------------------------------maxacc-----------------------------------------')
            torch.save(model.state_dict(), osp.join(path, 'maxacc' + '.pth'))

        if epoch>55:
            path = args.save_path + args.model + args.backbone +'CIFAR_Rotation_4_0.8' + '/'
            torch.save(model.state_dict(), osp.join(path, str(epoch) + '.pth'))

        lr_scheduler.step()



