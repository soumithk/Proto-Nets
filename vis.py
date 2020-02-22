import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet, CTM_apadter
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model = CTM_apadter(model, args).cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, labels = [_.cuda() for _ in batch]

        label = torch.arange(args.test_way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        logits = model(data)
        pred = torch.argmax(logits, dim=1)

        acc = count_acc(logits, label)

        print(labels)
        print(pred)
        print(label)
        input()

        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        x = None; p = None; logits = None

