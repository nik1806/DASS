import os, sys
import argparse
import numpy as np
import torch
from model.DeeplabV2 import *#Res_Deeplab

from torch.utils import data
import torch.nn as nn
import os.path as osp
import yaml
from dataset.dataset import *
from easydict import EasyDict as edict
from tqdm import tqdm
from PIL import Image
from utils.mwc import mwc_refiner
from datetime import datetime
from matplotlib import pyplot as plt

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("-n", "--num-classes", type=int, default=12)
    parser.add_argument("--frm", type=str, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='synthia_seq') #'cityscapes'
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--model", default='deeplab')
    parser.add_argument("--source", default='synthia_seq')
    return parser.parse_args()

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def compute_iou(model, testloader, args):
    model = model.eval()

    interp = nn.Upsample(size=(730, 1460), mode='bilinear', align_corners=True) #(1024, 2048)
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            temp_dir = 'baseline'
            image, label, edge, _, name = batch
            #output, _ =  model(image.cuda(), source=True)
            output, _ =  model(image.cuda(), source=False)
            label = label.cuda()
            src_output = interp(output)

            ##!! MWC solver
            # output, mb = mwc_refiner(src_output, name)
            # pred = output.float().cuda()
            # _, H, W = output.shape; C=19

            #!! original method
            output = src_output.squeeze()
            pred = output.argmax(dim=0).float().cuda()
            C, H, W = output.shape 

            Mask = (label.squeeze())<C

            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            pred_e = pred_e.repeat(1, H, W).cuda()
            pred_mask = torch.eq(pred_e, pred).byte()
            pred_mask = pred_mask*Mask
            
            # save_results(image, label, src_output.squeeze(0).argmax(dim=0), None, None, args.num_classes) #pred,mb) ##!!

            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            label = label.view(1, H, W)
            label_mask = torch.eq(label_e, label.float()).byte()
            label_mask = label_mask*Mask

            tmp_inter = label_mask+pred_mask
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds

        iou = inter/union
        acc = inter/preds
        # print(args.source)
        if args.source=='synthia':
            iou = iou.squeeze()
            class16_iou = torch.cat((iou[:9], iou[10:14], iou[15:16], iou[17:]))
            class16_miou = class16_iou.mean().item()
            class13_iou = torch.cat((class16_iou[:3], class16_iou[6:]))
            class13_miou = class13_iou.mean().item()
            print('16-Class mIoU:{:.2%}'.format(class16_miou))
            print(class16_iou)
            print('13-Class mIoU:{:.2%}'.format(class13_miou))
            print(class13_iou)
        elif args.source=='synthia_seq':
            iou = iou.squeeze()
            iou = torch.cat((iou[:3], iou[4:])).unsqueeze(dim=1)
            mIoU = iou.mean().item()
            print('11-Class mIoU:{:.2%}'.format(mIoU))
            # acc = [None for _ in range(len(iou))]
            # mAcc = None
            acc = acc.squeeze()
            acc = torch.cat((acc[:3], acc[4:])).unsqueeze(dim=1)
            mAcc = acc.mean().item()
            # print(class11_iou)
        else:
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
        
        print_iou(iou, acc, mIoU, mAcc)
        return iou, mIoU, acc, mAcc
    
def decode_segmap(temp, kwargs=None, cls=19):
    colors = [  # [  0,   0,   0],
        # [128, 64, 128],
        # [244, 35, 232],
        # [70, 70, 70],
        # [102, 102, 156],
        # [190, 153, 153],
        # [153, 153, 153],
        # [250, 170, 30],
        # [220, 220, 0],
        # [107, 142, 35],
        # [152, 251, 152],
        # [0, 130, 180],
        # [220, 20, 60],
        # [255, 0, 0],
        # [0, 0, 142],
        # [0, 0, 70],
        # [0, 60, 100],
        # [0, 80, 100],
        # [0, 0, 230],
        # [119, 11, 32],
        [128,64,128],
        [244,35,232],
        [70,70,70],
        [190,153,153],
        [153,153,153],
        [250,170,30],
        [220,220,0],
        [107,142,35],
        [70,130,180],
        [220,20,60],
        [255,0,0],
        [0,0,142],
    ]

    label_colours = dict(zip(range(cls), colors))

    '''
    Convert interger class labels to color image for representing segmentations.
    '''
    # print(temp.shape)
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, cls):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb    

def save_results(image, gt, source, out, mb, cls):
    now = datetime.now()

    fig = plt.figure(figsize=(40,20))#figsize
    col=3# if mb else 4
    
    fig.add_subplot(1,col,1)
    image = image.squeeze(0).permute(1,2,0)
    image = image.data.cpu().numpy()
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    image = (image + IMG_MEAN).astype(np.uint8)
    plt.imshow(image)
    plt.title("Original"); plt.axis('off')

    fig.add_subplot(1,col,2)
    gt_segmap = decode_segmap(gt.squeeze(0).data.cpu().numpy(), cls=cls)
    plt.imshow(gt_segmap)
    plt.title("Ground truth"); plt.axis('off')

    fig.add_subplot(1,col,3)
    source_segmap = decode_segmap(source.data.cpu().numpy(), cls=cls)
    plt.imshow(source_segmap)
    plt.title("DASS"); plt.axis('off')

    # fig.add_subplot(1,col,4)
    # out_segmap = decode_segmap(out.squeeze(0).data.cpu().numpy())
    # plt.imshow(out_segmap)
    # plt.title("MWC"); plt.axis('off')

    # if True:
    #     fig.add_subplot(1,col,5)
    #     plt.imshow(mb)
    #     plt.title("Motion boundary"); plt.axis('off')

    plt.savefig(f"results/synthia2city_{now.day}_{now.hour}_{now.minute}_{now.second}.png", bbox_inches='tight')
    plt.plot()
    # exit() ##!! change to show multiple random instances


def main():
    args = get_arguments()
    # with open('./config/damnet_config_upsize.yml') as f:
    with open('config/so_config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.num_classes=args.num_classes

    for i in range(args.start, 2):
        #model_path = osp.join(cfg['snapshot'], args.frm, 'GTA5_{0:d}.pth'.format(i*2000))# './snapshots/GTA2Cityscapes/source_only/GTA5_{0:d}.pth'.format(i*2000)
        #model_path = './weights/gta5_source_only_adaptation_3.pth'
        #model_path = './results/dam/snapshot/train/GTA5_baseline.pth'
        #model_path = './results/dam/snapshot/train/GTA5_best.pth'
        #model_path = './results/synthia_source_only/snapshot/train/wonkyung.pth'
        # model_path = './results/dam2/snapshot/train/Synthia_best.pth'
        # model_path = '/share_chairilg/weights/gta5_deepv2_trained_dass51.pth'
        model_path = './results/synthia_source_only/snapshot/train/synthia_best_source_only.pth'
        model = ResPair_Deeplab(num_classes=args.num_classes)
        #model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        model.eval().cuda()
        testloader = init_test_dataset(cfg, args.dataset, set='val')

        # compute_iou(model, testloader, args)
        iou, mIoU, acc, mAcc = compute_iou(model, testloader, args)

        print('Iter {}  finished, mIoU is {:.2%}'.format(i*2000, mIoU))

if __name__ == '__main__':
    main()
