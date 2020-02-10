import torch.nn as nn
import torch
from utils import euclidean_metric

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


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



# Currently only concentrator is implemented
class CTM_apadter(nn.Module):

    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.reshaper = nn.Conv2d(64, 64, 3)
        self.concentrator = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.Softmax2d()
        )
        self.projector = nn.Sequential(
            nn.Conv2d(64 * args.train_way, 64, 3),
            nn.Softmax2d()
        )

    def forward(self, data) :

        shot = self.args.shot
        if self.training :
            ways = self.args.train_way
        else :
            ways = self.args.test_way

        nk = shot * ways
        x, data_query = data[:nk], data[nk:]

        x = self.base_model(x)
        proto = self.reshaper(x)

        feats = x.reshape(shot, ways, x.size(1), x.size(2), x.size(3)).mean(0)
        p = self.projector(feats.reshape(-1, 5, 5).unsqueeze(0))
        p = p.view(p.size(0), -1)


        proto = proto.reshape(shot, ways, -1).mean(0)
        proto = proto.view(proto.size(0), -1)
        proto = torch.mul(proto, p)


        query = self.reshaper(self.base_model(data_query))
        query = query.view(query.size(0), -1)
        query = torch.mul(query, p)

        logits = euclidean_metric(query, proto)

        return logits

