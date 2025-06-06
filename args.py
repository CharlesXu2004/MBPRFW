import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=150)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='FC100', choices=['mini', 'CUB', 'FC100', 'CIFAR'])
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--save-epoch', type=int, default=2)
    parser.add_argument('--save-path', default='./save/')
    return parser

def get_parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='FC100', choices=['mini', 'CUB', 'FC100', 'CIFAR'])
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--model-path', default='./save/model_best.pth')
    return parser

def get_parser_transfer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--backbone', default='ResNet12')
    parser.add_argument('--dataset', default='FC100', choices=['mini', 'CUB', 'FC100', 'CIFAR'])
    parser.add_argument('--model', default='base')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--projection', type=bool, default=True)
    parser.add_argument('--save-epoch', type=int, default=2)
    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--BN', type=bool, default=False)
    parser.add_argument('--SOM', type=bool, default=True)
    parser.add_argument('--model-path', default='./save/model_best.pth')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()