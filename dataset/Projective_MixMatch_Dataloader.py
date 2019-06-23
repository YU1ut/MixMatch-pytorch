import torch
import torch.utils.data as data
import numpy as np
import random
import os
from PIL import Image, PILLOW_VERSION
import numbers
from torchvision.transforms.functional import _get_inverse_affine_matrix
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class Projective_MixMatch_Data_Loader(data.Dataset):
    def __init__(self, dataset_dir,shift=6, train_label=True, scale=None,  resample=False,
                 fillcolor=0,matrix_transform=None,
                 transform_pre=None, transform=None, target_transform=None, rand_state=888,
                 valid_size=0.1,uniform_label=False,num_classes=10,unlabel_Data=False):
        super(Projective_MixMatch_Data_Loader, self).__init__()
        self.root=os.path.abspath(dataset_dir)
        self.shift=shift
        self.trainsetFile = []
        self.aimsetFile = []
        listfiles = os.listdir(dataset_dir)
        self.trainlist = [os.path.join(dataset_dir, x) for x in listfiles if "trainset" in x]
        self.aimlist = [os.path.join(dataset_dir, x) for x in listfiles if "aimset" in x]
        self.trainlist.sort()
        self.aimlist.sort()
        self.train_label=train_label
        self.valid_size=valid_size
        self.unlabel_Data=unlabel_Data
        # here update this with 80% as training, 20%as validation
        if valid_size>0:
            if uniform_label==False:

                X_train, X_test, y_train, y_test = train_test_split(self.trainlist, self.aimlist, test_size=valid_size,
                                                            random_state=rand_state)
                if train_label:
                    self.trainlist = X_train
                    self.aimlist = y_train

                else:
                    self.trainlist = X_test
                    self.aimlist = y_test
            else:
                #pick the uniform valid size indicated
                shuffle_range=np.arange(len(self.trainlist))
                random.seed(rand_state)
                random.shuffle(shuffle_range)
                require_size=int(len(self.aimlist)*valid_size/num_classes)
                self.trainlist,self.aimlist=self.pick_top_k_example(require_size,shuffle_range,num_classes)
        if uniform_label==True and len(self.trainlist)<50000:
            #to accelerate training to avoid dataloader load again and again for small data
            repeat_times=int(50000/len(self.trainlist))
            self.trainlist=self.trainlist*repeat_times
            self.aimlist=self.aimlist*repeat_times
        self.transform_pre = transform_pre
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        self.resample = resample
        self.fillcolor = fillcolor
        self.transform = transform
        self.target_transform = target_transform
        self.matrix_transform=matrix_transform
    def pick_top_k_example(self,img_per_cat,shuffle_range,num_class):
        record_dict=defaultdict(list)
        for i in range(len(shuffle_range)):
            tmp_id=shuffle_range[i]
            label=int(np.load(self.aimlist[tmp_id]))
            if label not in record_dict:
                record_dict[label].append(tmp_id)
            elif len(record_dict[label])<img_per_cat:
                record_dict[label].append(tmp_id)
            break_flag=True
            if len(record_dict)<num_class:
                break_flag=False
            for tmp_label in record_dict.keys():
                tmp_length=len(record_dict[tmp_label])
                if tmp_length<img_per_cat:
                    break_flag=False
                    break
            if break_flag:
                break
        #specify new trainlist and aimlist
        assert len(record_dict)==num_class
        for tmp_label in record_dict.keys():
            tmp_length = len(record_dict[tmp_label])
            assert tmp_length==img_per_cat
        train_list=[]
        aim_list=[]
        for tmp_label in record_dict.keys():
            tmp_list=record_dict[tmp_label]
            for tmp_id in tmp_list:
                train_list.append(self.trainlist[tmp_id])
                aim_list.append(self.aimlist[tmp_id])
        return train_list,aim_list



    @staticmethod
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def normalise(self,x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616) ):
        x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
        x -= mean * 255
        x *= 1.0 / (255 * std)
        return x
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        train_path = self.trainlist[index]
        aim_path = self.aimlist[index]
        img1 = np.load(train_path)
        target = np.load(aim_path)
        #if self.unlabel_Data:

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #print (img1.shape)
        img1 = img1.transpose((1, 2, 0))
        #print(img1.shape)
        #img1 = self.normalise(img1)#normalize
        #print(img1.shape)
        img1 = Image.fromarray(img1)
        if self.transform_pre is not None:
            if self.unlabel_Data:
                self.transform_now=TransformTwice(self.transform_pre)
                img1,img1_another=self.transform_now(img1)
            else:
                img1 = self.transform_pre(img1)

        # projective transformation on image2
        width, height = img1.size
        center = (img1.size[0] * 0.5 + 0.5, img1.size[1] * 0.5 + 0.5)
        shift = [float(random.randint(-int(self.shift), int(self.shift))) for ii in range(8)]
        scale = random.uniform(self.scale[0], self.scale[1])
        rotation = random.randint(0, 3)

        pts = [((0 - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (height - center[1]) * scale + center[1]),
               ((0 - center[0]) * scale + center[0], (height - center[1]) * scale + center[1])]
        pts = [pts[(ii + rotation) % 4] for ii in range(4)]
        pts = [(pts[ii][0] + shift[2 * ii], pts[ii][1] + shift[2 * ii + 1]) for ii in range(4)]

        coeffs = self.find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )

        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img2 = img1.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

        img1 = np.array(img1).astype('float32')
        img2 = np.array(img2).astype('float32')
        #print (img1.shape)
        img1 = self.normalise(img1)
        #print(img1)
        #print(img1.shape)
        img2 = self.normalise(img2)
        #img1 = torch.from_numpy(img1)
        #img2 = torch.from_numpy(img2)

        img1 = img1.transpose((2, 0, 1))
        img2 = img2.transpose((2, 0, 1))#do not use pca any more

        if self.transform is not None:

            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)

        if self.matrix_transform is not None:
            coeffs = self.matrix_transform(coeffs)
        if self.unlabel_Data:
            img1_another = np.array(img1_another).astype('float32')
            img1_another = self.normalise(img1_another)
            img1_another = img1_another.transpose((2, 0, 1))
            if self.transform is not None:
                img1_another = self.transform(img1_another)

            img1_another = torch.from_numpy(img1_another)
            return (img1,img1_another), img2, coeffs, target
        else:
            return img1, img2, coeffs, target

    def __len__(self):
        return len(self.aimlist)