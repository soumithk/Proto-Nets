import torch.nn as nn
import torch
from utils import euclidean_metric
from torch.nn import functional as F
import math

from layers import conv_block, Bottleneck



class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x
        # return x.view(x.size(0), -1)

class ResNet(nn.Module):
    def __init__(self, block, layers, in_c):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class CTM_apadter(nn.Module):

    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        out_size = 16

        if args.base_model.startswith('resnet'):
            self.inplanes = 256 * 4
            self.concentrator = nn.Sequential(
                self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                self._make_layer(Bottleneck, out_size, 2, stride=1)
            )
            self.inplanes = args.train_way * 64
            self.projector = nn.Sequential(
                self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                self._make_layer(Bottleneck, out_size, 2, stride=1)
            )
            self.inplanes = 256 * 4
            self.reshaper = nn.Sequential(
                self._make_layer(Bottleneck, out_size*2, 3, stride=1),
                self._make_layer(Bottleneck, out_size, 2, stride=1)
            )
        else :
            self.inplanes = 64
            self.concentrator = self._make_layer(Bottleneck, out_size, 2, stride=1)
            self.inplanes = 64 * args.train_way
            self.projector = self._make_layer(Bottleneck, out_size, 2, stride=1)
            self.reshaper = None
    def forward(self, data) :

        shot = self.args.shot
        if self.training :
            ways = self.args.train_way
        else :
            ways = self.args.test_way

        nk = shot * ways
        x, data_query = data[:nk], data[nk:]

        x = self.base_model(x)
        if self.reshaper is not None :
            proto = self.reshaper(x)
        else :
            proto = x
        proto = proto.view(shot, ways, -1).mean(0)

        if self.args.base_model.startswith('resnet') :
            c, w, h = 64, 6, 6
        else:
            c, w, h = 64, 5, 5
        concentrated = self.concentrator(x)
        concentrated = concentrated.view(shot, ways, c, w, h).mean(0)
        stacked = concentrated.view(-1, w, h).unsqueeze(0)
        mask =  self.projector(stacked)
        mask = F.softmax(mask, dim = 1)
        mask = mask.view(1, -1)
        proto = torch.mul(proto, mask)


        # query = self.reshaper(self.base_model(data_query))
        query = self.base_model(data_query)
        if self.reshaper is not None :
            query = self.reshaper(query)
        query = query.view(query.size(0), -1)
        query = torch.mul(query, mask)

        # print(mask.shape, proto.shape, query.shape)

        logits = euclidean_metric(query, proto)

        return logits

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class No_apadter(nn.Module):

    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args

    def forward(self, data):
        shot = self.args.shot
        if self.training :
            ways = self.args.train_way
        else :
            ways = self.args.test_way

        nk = shot * ways
        x, data_query = data[:nk], data[nk:]

        x = self.base_model(x)
        proto = x.view(shot, ways, -1).mean(0)

        query = self.base_model(data_query)
        query = query.view(query.size(0), -1)


        logits = euclidean_metric(query, proto)

        return logits

def create_model(args):

    if args.base_model == "convnet" :
        model = Convnet().cuda()
    elif args.base_model == "resnet18" :
        model = ResNet(Bottleneck, [2, 2, 2], 3).cuda()
    else : # default
        model = Convnet().cuda()

    if args.use_CTM :
        model = CTM_apadter(model, args).cuda()

    return model

