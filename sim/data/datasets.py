'''Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, CINIC-10
Note: In FL, it is impossible to get the mean and std of the whole training set (yipeng, 2023-11-16)
'''
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import csv
from collections import Counter
import torch

def build_dataset(dataset_name='mnist', dataset_dir = '../datasets/'):
    if dataset_name == 'mnist':
        origin_dataset = OriginMNIST()
    elif dataset_name == 'fashionmnist':
        origin_dataset = OriginFashionMNIST()
    elif dataset_name == 'cifar10':
        origin_dataset = OriginCIFAR10()
    elif dataset_name == 'cifar100':
        origin_dataset = OriginCIFAR100()
    elif dataset_name == 'cinic10':
        origin_dataset = OriginCINIC10()
    elif dataset_name == 'sst2':
        origin_dataset = OriginSST2(root=dataset_dir)
    return origin_dataset


class SST2Dataset(Dataset):
    """一个简单的自定义 Dataset 类，用于包装从 tsv 文件加载的数据"""

    def __init__(self, data):
        self.data = data
        self.targets = [int(label) for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OriginSST2():
    '''
    Manually loads the SST-2 dataset from .tsv files without torchtext.
    Handles tokenization, vocabulary building, and numericalization.
    Provides a compatible interface for frameworks expecting torchvision-style datasets.
    '''

    def __init__(self, root='../datasets/', min_freq=2):
        self.root = root
        self.min_freq = min_freq

        # 添加空的 transform 属性，以兼容期望 torchvision-style 数据集的框架
        self.transform_train = None
        self.transform_test = None

        self.train_path = os.path.join(root, 'SST-2', 'train.tsv')
        self.test_path = os.path.join(root, 'SST-2', 'test.tsv')

        self.tokenizer = lambda s: s.lower().split()

        self.train_data = self._load_data(self.train_path)
        self.test_data = self._load_data(self.test_path)

        self.vocab = self._build_vocab(self.train_data)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab['<pad>']
        self.unk_idx = self.vocab['<unk>']

        self.text_pipeline = lambda x: [self.vocab.get(token, self.unk_idx) for token in self.tokenizer(x)]
        self.label_pipeline = lambda x: int(x)

    def _load_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) == 2:
                    sentence, label = row
                    data.append((sentence, label))
        return data

    def _build_vocab(self, data):
        counter = Counter()
        for sentence, _ in data:
            counter.update(self.tokenizer(sentence))

        vocab = {'<pad>': 0, '<unk>': 1}
        word_idx = 2
        for word, count in counter.items():
            if count >= self.min_freq:
                vocab[word] = word_idx
                word_idx += 1
        return vocab

    def get_trainset(self, transform=None):  # <-- 已修改
        """
        返回包装后的训练集 Dataset 对象。
        `transform` 参数被接收但在此处被忽略，以保持与 torchvision-style 数据集的接口兼容性。
        """
        return SST2Dataset(self.train_data)

    def get_testset(self, transform=None):  # <-- 已修改
        """
        返回包装后的测试集 Dataset 对象。
        `transform` 参数被接收但在此处被忽略，以保持与 torchvision-style 数据集的接口兼容性。
        """
        return SST2Dataset(self.test_data)

    def collate_batch(self, batch):
        label_list, text_list, lengths = [], [], []
        for (text, label) in batch:
            label_list.append(self.label_pipeline(label))
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(len(processed_text))

        labels = torch.tensor(label_list, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        padded_texts = pad_sequence(text_list, batch_first=True, padding_value=self.pad_idx)

        return labels, padded_texts, lengths
class OriginMNIST():
    '''
    https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    def get_trainset(self, transform=None):
        trainset = MNIST(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = MNIST(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginFashionMNIST():
    '''
    Calculate the mean and std of the training dataset manually
    Some links also use `mean=0.1307, std=0.3081` (MNIST) [1] or `mean=0.5, std=0.5` [2]
    [1] https://github.com/IBM/fl-arbitrary-participation/blob/main/dataset/dataset.py
    [2] https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        self.transform_train = transforms.Compose([
            # https://discuss.pytorch.org/t/data-augmentation-fashion-mnist-image/136762
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            ])
    
    def get_trainset(self, transform=None):
        trainset = FashionMNIST(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = FashionMNIST(root=self.root, train=False, download=True, transform=transform)
        return testset
    

class OriginCIFAR10():
    '''https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar10/data_loader.py'''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    def get_trainset(self, transform=None):
        trainset = CIFAR10(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = CIFAR10(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginCIFAR100():
    '''https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar100/data_loader.py'''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    def get_trainset(self, transform=None):
        trainset = CIFAR100(root=self.root, train=True, download=True, transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = CIFAR100(root=self.root, train=False, download=True, transform=transform)
        return testset


class OriginCINIC10():
    '''
    https://github.com/BayesWatch/cinic-10
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cinic10/data_loader.py
    '''
    def __init__(self, root='../datasets/'):
        self.root = root
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def get_trainset(self, transform=None):
        trainset = ImageFolder('{}/{}'.format(self.root, '/CINIC-10/train/'), transform=transform)
        return trainset
    
    def get_testset(self, transform=None):
        testset = ImageFolder('{}/{}'.format(self.root, '/CINIC-10/test/'), transform=transform)
        return testset





if __name__ == '__main__':
    pass
    # for i in range(0, 3):
    #     # to judge if the sample sequence is the same at different times
    #     train_dataset, test_dataset = dataset_mnist('../datasets/')
    #     print(train_dataset.targets[:30])
   