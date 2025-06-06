from args import get_parser
import os.path as osp

from torchvision.models import ResNet
import numpy as np
import os
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import CIFAR, FC100, CUB, MiniImageNet
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12
from BaseModel import BaseModel
from torchvision import transforms
from tqdm import tqdm

# import wandb
# wandb.init(project="MBPRFW", 
#             name="pretrain_0", 
#             config={
#                 "max_epoch": 150,
#                 "train_way": 5,
#                 "test_way": 5,
#                 "query": 15,
#                 "shot": 5,
#                 "backbone": "ResNet12",
#                 "dataset": "FC100",
#                 "lr_start": 0.025,
#                 "step_size": 30,
#                 "gamma": 0.2
#             }
#     )

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size_ = 64

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12
}

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if 'Conv' in args.backbone or 'ResNet12' in args.backbone:
        image_size = 84
    else:
        image_size = 224

    if args.dataset == 'mini':
        trainset = MiniImageNet('train', image_size, True)
        valset = MiniImageNet('val', image_size, False)
    elif args.dataset == 'CUB':
        trainset = CUB('train', image_size, True)
        valset = CUB('val', image_size, False)
    elif args.dataset == 'FC100':
        trainset = FC100('train', image_size, True)
        valset = FC100('val', image_size, False)
    elif args.dataset == 'CIFAR':
        trainset = CIFAR('train', image_size, True)
        valset = CIFAR('val', image_size, False)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size_, shuffle=True, num_workers=1, pin_memory=True)
    val_sampler = CategoriesSampler(valset.label, 400, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=1, pin_memory=True)

    backbone = model_dict[args.backbone]()
    model = BaseModel(backbone, trainset.num_classes, args.test_way)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    path = osp.join(args.save_path, "_".join([args.backbone, args.dataset, f"{str(args.test_way)}way", f"{str(args.shot)}shot"]))
    if not os.path.exists(path):
        os.makedirs(path)

    max_acc = 0
    for epoch in range(1, args.max_epoch + 1):

        model.train()
        for j, batch in enumerate(tqdm(train_loader, desc="Epoch {} Training".format(epoch))):

            data_0, data_1, train_label = [_.cuda() for _ in batch]
            data = torch.cat((data_0,data_1),0)
            batch_size_ = len(train_label)

            #----------rotation--------#
            x_90 = data_0.transpose(2,3).flip(2)
            x1_90 = data_1.transpose(2,3).flip(2)
            x_180 = data_0.flip(2).flip(3)
            x1_180 = data_1.flip(2).flip(3)
            x_270 = data_0.flip(2).transpose(2,3)
            x1_270 = data_1.flip(2).transpose(2, 3)
            data_rotation = torch.cat((data_0, x_90, x_180, x_270, data_1, x1_90, x1_180, x1_270), 0)
            label_aug = torch.arange(8).repeat(batch_size_).reshape(batch_size_,-1).t().reshape(-1).cuda()
            label = train_label.repeat(8)

            loss, acc = model.train_rotation_crop(data_rotation,label,label_aug)
            #loss, acc = model.train_loop_trans(data, train_label,data_trans,outs,label_outs)
            #loss = model.train_contrastive(data, train_label, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_acc = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Epoch {} Validation".format(epoch))):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                acc = model.evaluate(data_query, data_shot, label, args.shot)
                val_acc += acc

        val_acc /= len(val_loader)
        print("epoch {}, acc={:.4f}".format(epoch, val_acc))

        if (val_acc > max_acc):
            max_acc = val_acc
            print('-----------------------------------model_best-----------------------------------------')
            torch.save(model.state_dict(), osp.join(path, 'model_best.pth'))

        if epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), osp.join(path, str(epoch) + '.pth'))

        lr_scheduler.step()
        # lr_last = lr_scheduler.get_last_lr()[0]
        # wandb.log({
        #     "epoch": epoch,
        #     "lr": lr_last,
        #     "acc": val_acc,
        #     "loss": loss.item(),
        # })