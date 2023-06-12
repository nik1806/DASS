import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import torch
import cv2

class BaseDataSet(data.Dataset):
    def __init__(self, root, list_path,dataset, num_class,  joint_transform=None, transform=None, label_transform = None,
                 max_iters=None, ignore_label=255, set='val', plabel_path=None, max_prop=None, selected=None,centroid=None,
                 wei_path=None):

        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.dataset = dataset
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.plabel_path = plabel_path
        self.centroid = centroid

        if self.set !='train':
            self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids =[]
        if selected is not None:
            self.img_ids = selected
        else:
            with open(self.list_path) as f:
                for item in f.readlines():
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        elif max_prop is not None:
            total = len(self.img_ids)
            to_sel = int(np.floor(total * max_prop))
            index = list( np.random.choice(total, to_sel, replace=False) )
            self.img_ids = [self.img_ids[i] for i in index]

        self.files = []
        self.id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                          19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                          26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.id2train_synthia = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                          15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                          8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        
        self.id2train_synthia_seq = {3: 0, 4: 1, 2: 2, 5: 3, 7: 4, 15: 5, 9: 6, 6: 7, 1: 8, 10: 9, 11: 10, 8: 11,} ##!!

        if dataset == 'synthia_seq':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'GT/LABELS/Stereo_Left/Omni_F/')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "RGB/Stereo_Left/Omni_F/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

        if dataset =='synthia':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'GT/synthia_mapped_to_cityscapes')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "RGB/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
            #imageio.plugins.freeimage.download()

        if dataset=='gta5':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'labels')
            else:
                label_root = self.plabel_path

            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(label_root, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

        elif dataset=='cityscapes':
            if self.plabel_path is None:
                label_root = osp.join(self.root, 'gtFine', self.set)
            else:
                label_root = self.plabel_path
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file =osp.join(label_root, '%s' % (label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

#        try:
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]) if self.dataset!='synthia_seq' else cv2.imread(datafiles['label'], cv2.IMREAD_UNCHANGED)[:,:,2] # only one channel contain the label
        name = datafiles["name"]

        label = np.asarray(label, np.uint8)
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        
        if self.dataset=='gta5' or self.dataset=='cityscapes' :
            if self.plabel_path is None:
                for k, v in self.id2train.items():
                    label_copy[label == k] = v
            else:
                label_copy = label

        elif self.dataset=='synthia':
            if self.plabel_path is None:
                for k, v in self.id2train_synthia.items():
                    label_copy[label == k] = v
            else:
                label_copy = label
                
        elif self.dataset=='synthia_seq': ##!!
            if self.plabel_path is None:
                for k, v in self.id2train_synthia_seq.items():
                    label_copy[label == k] = v
            else:
                label_copy = label

        label = Image.fromarray(label_copy.astype(np.uint8))
        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label, None)
        if self.transform is not None:
            image = self.transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)
#        except Exception as e:
#            print(index)
#            index = index - 1 if index > 0 else index + 1
#            return self.__getitem__(index)
        return image, label, 0, 0, name


