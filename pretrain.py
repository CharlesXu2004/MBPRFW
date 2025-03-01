import argparse
import os.path as osp

from torchvision.models import ResNet
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
from backbones import backbone
from backbones import resnet12
from basefinetine import BaseFinetine
#from methods.protonet import ProtoNet
#from methods.relationnet import RelationNet
#from methods.basefinetine import BaseFinetine
import torchvision.transforms as transforms
from PIL import Image
from utils import TwoCropTransform

def rotrate_concat(inputs):
    outs = None
    flag = 1
    inputs = inputs.cpu()
    for x in inputs:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        x1 = transforms.ToPILImage()(x)
        x2 = transforms.functional.resized_crop(x1, np.random.randint(28), 0, 56, 84, (84, 84))
        x2 = transform(x2)
        x3 = transforms.functional.resized_crop(x1, 0, np.random.randint(28), 56, 84, (84, 84))
        x3 = transform(x3)
        x4 = transforms.functional.resized_crop(x1, np.random.randint(28), np.random.randint(28), 56, 56, (84, 84))
        x4 = transform(x4)
        x1 = transform(x1)
        xr=torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)), 0)
        for xx in xr:
            x_90 = xx.transpose(1, 2).flip(1).unsqueeze(0)
            x_180 = xx.flip(1).flip(2).unsqueeze(0)
            x_270 = xx.flip(1).transpose(1, 2).unsqueeze(0)
            if flag==1:
               outs = torch.cat((xx.unsqueeze(0), x_90, x_180, x_270),0)
               flag = 0
            else:
                outs = torch.cat((outs, xx.unsqueeze(0), x_90, x_180, x_270), 0)

    return outs

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


    trainset = MiniImageNet('train', image_size, True)

    train_loader = DataLoader(dataset=trainset, batch_size=3, shuffle=True, num_workers=8, pin_memory=True)
    valset = MiniImageNet('val', image_size, False)
    # val_sampler = CategoriesSampler(valset.label, 400,
    #                                 args.test_way, args.shot + args.query)
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    if args.model == 'relationnet':
        if args.backbone == 'Conv4':
            backbone = backbone.Conv4NP()
        else:
            backbone = model_dict[args.backbone]()
        #model = RelationNet(backbone, args.train_way, args.test_way, args.shot, args.hidden_size)
    elif args.model == 'protonet':
        backbone = model_dict[args.backbone]()
        #model = ProtoNet(backbone, args.train_way, args.test_way, args.shot, args.temperature)
    else:
        backbone = model_dict[args.backbone]()
        model = BaseFinetine(backbone, trainset.num_classes, args.test_way)

    if args.model == 'protonet':
        model.set_pretrain(trainset.num_classes)
    # print(trainset.num_classes)
    model.cuda()
    # optimizer = torch.optim.SGD([{'params': model.model.parameters(), 'lr': args.lr},
    #                              {'params': model.fc.parameters(), 'lr': args.lr}
    #                              ], momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # print(optimizer.param_groups)

    result = {}
    result['acc'] = 0
    for epoch in range(1, args.max_epoch + 1):
        # if epoch in args.schedule:
        #     lr *= args.gamma
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        all_loss = 0
        all_loss_x = 0
        all_en_loss = 0
        all_eq_loss = 0
        all_acc_x = 0
        all_acc = 0
        all_acc_eq = 0
        model.train()
        for j, batch in enumerate(train_loader):
            data, train_label = [_.cuda() for _ in batch]

            if args.model == 'protonet':
                loss, acc = model.pretrain_forward_loss(data, train_label)
            else:
                #loss, acc = model.train_loop(data, train_label)

                loss,loss_x,en_loss,eq_loss,acc_x,acc,acc_eq= model.train_loop_eq(data,train_label)

                #loss, acc = model.train_loop_trans(data, train_label,data_trans,outs,label_outs)
                #loss = model.train_contrastive(data, train_label, args.temperature)

                #print("rotation:",acc_eq)
            all_acc = all_acc+acc
            all_acc_x = all_acc_x+acc_x
            all_acc_eq = all_acc_eq + acc_eq
            all_loss = all_loss + loss
            all_loss_x = all_loss_x + loss_x
            all_en_loss = all_en_loss + en_loss
            all_eq_loss = all_eq_loss + eq_loss

            print('epoch {}, train {}/{}, loss={:.4f}'
                  .format(epoch, j, len(train_loader), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        n = len(train_loader)
        with open("loss.txt", "a")as f:
            f.write("epoch:" + ''.join(str(epoch)) + " loss:" + ''.join(str(float(all_loss/n))) + "  loss_x "+ ''.join(str(float(all_loss_x/n)))+ "  en_loss "+ ''.join(str(float(all_en_loss/n)))+  "  eq_loss "+ ''.join(str(float(all_eq_loss/n)))+ '\r\n')
        with open("acc.txt", "a")as f:
            f.write("epoch:" + str(epoch) + ": acc_x: " + ''.join(str(float(all_acc_x / n))) + " acc: " + ''.join(str(float(all_acc / n)))+ " acc_eq: " + ''.join(str(float(all_acc_eq / n)))+ '\r\n')


        model.eval()

        val = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]
                # if args.model == 'protonet' or args.model == 'basefinetine':
                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                # else:
                #     y = torch.from_numpy(np.tile(range(args.test_way), args.query))
                #     label = torch.zeros((len(y), args.test_way)).scatter_(1, y.unsqueeze(1), 1).cuda()
                if args.model == 'protonet':
                    loss, acc = model.set_forward_loss(data_query, data_shot, label)
                else:
                    acc = model.evaluate(data_query, data_shot, label)
                val += acc

        print("epoch {}, acc={:.4f}".format(epoch, val / len(val_loader)))
        with open("ACC_VAL.txt", "a")as f:
            f.write("epoch:" + str(epoch) + "  " + ''.join(str(float(val / len(val_loader)))) + '\r\n')

        if (val > result['acc']):
            result['acc'] = val
            path = args.save_path + args.model + args.backbone + '/'
            if not os.path.exists(path):
                os.mkdir(path)
            print('-----------------------------------maxacc-----------------------------------------')
            torch.save(model.state_dict(), osp.join(path, 'maxacc' + '.pth'))

        if epoch % args.save_epoch == 0 or epoch>80:
            path = args.save_path + args.model + args.backbone + '/'
            torch.save(model.state_dict(), osp.join(path, str(epoch) + '.pth'))

        lr_scheduler.step()



