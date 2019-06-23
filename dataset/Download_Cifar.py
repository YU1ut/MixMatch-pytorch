
from ops.os_operation import mkdir
import os
from torchvision.datasets.utils import download_url, check_integrity
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np

class CIFAR10(object):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

        download ():  downloads the dataset from the internet and
            puts it in root directory
    """

    def __init__(self, save_path):
        self.root=save_path
        self.download_init()
        if not self._check_integrity():
            mkdir(save_path)
            self.download()
        self.final_path=os.path.join(save_path,'cifar10')
        mkdir(self.final_path)
        #generate npy files here
        self.train_path=os.path.join(self.final_path,'trainset')
        self.test_path = os.path.join(self.final_path, 'testset')
        mkdir(self.train_path)
        mkdir(self.test_path)
        if os.path.getsize(self.train_path)<10000:
            self.Process_Dataset(self.train_list,self.train_path)
        if os.path.getsize(self.test_path)<10000:
            self.Process_Dataset(self.test_list,self.test_path)
    def download_init(self):
        self.base_folder = 'cifar-10-batches-py'
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.filename = "cifar-10-python.tar.gz"
        self.tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        self.train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        self.test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _check_integrity(self):

        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def Process_Dataset(self,train_list,train_path):
        train_data=[]
        train_labels=[]
        for fentry in train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, 'rb') as fo:
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                train_data.append(entry['data'])
                if 'labels' in entry:
                    train_labels += entry['labels']
                else:
                    train_labels += entry['fine_labels']
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((len(train_data), 3, 32, 32))
        train_labels=np.array(train_labels)
        #following Channel,height,width format
        #self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        for i in range(len(train_data)):
            tmp_train_path=os.path.join(train_path,'trainset'+str(i)+'.npy')
            tmp_aim_path = os.path.join(train_path, 'aimset' + str(i) + '.npy')
            np.save(tmp_train_path,train_data[i])
            np.save(tmp_aim_path,train_labels[i])
