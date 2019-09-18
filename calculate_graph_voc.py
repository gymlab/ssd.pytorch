from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import math
import numpy as np
from layers.box_utils import match

dataset = 'VOC'
dataset_root = VOC_ROOT
trained_model = 'weights/VOC.pth'
save_folder = 'graph/'
batch_size = 32
num_workers = 8
cuda = True

if torch.cuda.is_available():
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

cfg = voc
dataset = VOCDetection(root=dataset_root, transform=BaseTransform(cfg['min_dim'], MEANS))
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net

net = torch.nn.DataParallel(ssd_net)
cudnn.benchmark = True

ssd_net.load_weights(trained_model)

if cuda:
    net = net.cuda()

net.eval()

epoch_size = len(dataset) // batch_size
step_index = 0

data_loader = data.DataLoader(dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=False, collate_fn=detection_collate,
                              pin_memory=True, drop_last=False)

occurrence_matrix = torch.zeros(20, 20)

# create batch iterator
for iteration, batch_data in enumerate(data_loader):
    # load train data
    images, targets = batch_data

    if cuda:
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]
    else:
        images = images
        targets = [ann for ann in targets]

    # batch iterator
    for _, b_target in enumerate(targets):
        co_labels = []
        for n in range(b_target.size(0)):
            # radius = torch.sqrt((b_target[n, 2] - b_target[n, 0]) * (b_target[n, 3] - b_target[n, 1]))
            cls = b_target[n, 4:].item()
            # if radius < 0.1:
            #     idx = 0
            # elif radius < 0.3:
            #     idx = 1
            # elif radius < 0.5:
            #     idx = 2
            # elif radius < 0.7:
            #     idx = 3
            # elif radius < 0.9:
            #     idx = 4
            # else:
            #     idx = 5

            co_labels += [cls - 1]
        idx = torch.tensor(co_labels).long()
        one_hots = torch.zeros(len(idx), 20).scatter_(1, idx.unsqueeze(1), 1.)
        occurrence, _ = one_hots.max(dim=0)
        occurrence_matrix += torch.matmul(occurrence.unsqueeze(1), occurrence.unsqueeze(0))

    print('(%d / %d) proceed' % (iteration, epoch_size))

for n in range(occurrence_matrix.size(0)):
    occurrence_matrix[n, n] = 0.
    print(occurrence_matrix[n])

torch.save(occurrence_matrix, 'A.pt')

