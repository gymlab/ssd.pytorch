import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from layers.modules.gcn import MSGCN
import pickle


class WIAggregate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WIAggregate, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.agg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feature, word_attention):
        img_feature = self.bn(img_feature)
        img_feature = self.relu(word_attention * img_feature)
        return self.agg(img_feature)


class GSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(GSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.ms_gcn = MSGCN(self.num_classes - 1, 6, 6, t=0.4, p=0.2, adj_file='A.pt')
        with open('voc_glove_word2vec.pkl', 'rb') as f:
            self.word_embedding = torch.from_numpy(pickle.load(f))
        self.word_embedding = torch.autograd.Variable(self.word_embedding).float().cuda().detach()
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.agg = nn.ModuleList(
            [WIAggregate(512, 256), WIAggregate(1024, 256), WIAggregate(512, 256),
             WIAggregate(256, 256), WIAggregate(256, 256), WIAggregate(256, 256)]
        )

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        att = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # s = self.L2Norm(x)
        sources.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # Apply MSGCN
        gcn_attention = self.ms_gcn(self.word_embedding)
        att.append(gcn_attention[:512].view(1, 512, 1, 1))
        att.append(gcn_attention[(512):(512+1024)].view(1, 1024, 1, 1))
        att.append(gcn_attention[(512+1024):(512+1024+512)].view(1, 512, 1, 1))
        att.append(gcn_attention[(512+1024+512):(512+1024+512+256)].view(1, 256, 1, 1))
        att.append(gcn_attention[(512+1024+512+256):(512+1024+512+256+256)].view(1, 256, 1, 1))
        att.append(gcn_attention[(512+1024+512+256+256):].view(1, 256, 1, 1))

        # apply multibox head to source layers
        for (x, a, g, l, c) in zip(sources, att, self.agg, self.loc, self.conf):
            x = g(x, a)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(x.type())                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    det_channels = 256
    loc_layers = []
    conf_layers = []
    # vgg_source = [21, -2]
    # for k, v in enumerate(vgg_source):
    #     loc_layers += [nn.Conv2d(vgg[v].out_channels,
    #                              cfg[k] * 4, kernel_size=3, padding=1)]
    #     conf_layers += [nn.Conv2d(vgg[v].out_channels,
    #                     cfg[k] * num_classes, kernel_size=3, padding=1)]
    # for k, v in enumerate(extra_layers[1::2], 2):
    #     loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
    #                              * 4, kernel_size=3, padding=1)]
    #     conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
    #                               * num_classes, kernel_size=3, padding=1)]
    for k in range(6):
        loc_layers += [nn.Conv2d(det_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(det_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

mbox = {
    '300': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [],
}


def build_gssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return GSSD(phase, size, base_, extras_, head_, num_classes)
