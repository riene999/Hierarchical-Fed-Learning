'''Three partition strategies are included: IID, Dir, ExDir.
适用于 SST2 数据集（二分类情感分析数据集）
'''
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

CHECK = True


def build_partition(dataset_name='sst2', num_clients=10, num_groups=10, partition='iid', alpha=[]):
    r"""Build local data distributions to assign data sample indices to clients.
    Args:
        num_clients (int): number of clients
        partition (str): partition way, e.g., iid, dir and exdir
        alpha (list): parameters of partition ways

    Returns:
        dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
    """
    map_dir = './maps/'  # map directory
    if partition == 'iid':
        map_path = "{}/{}_M{}_{}.txt".format(map_dir, dataset_name, num_clients, partition)
    elif partition == 'dir':
        alpha = alpha[0]
        map_path = "{}/{}_M{}_{}{}.txt".format(map_dir, dataset_name, num_clients, partition, alpha)
    elif partition == 'exdir':
        C = int(alpha[0])
        alpha = alpha[1]
        map_path = "{}/{}_M{}_{}{},{}.txt".format(map_dir, dataset_name, num_clients, partition, C, alpha)
    elif partition == 'exdirb':
        C = int(alpha[0])
        alpha = alpha[1]
        map_path = "{}/{}_M{}_{}{},{}.txt".format(map_dir, dataset_name, num_clients, partition, C, alpha)
    elif partition == 'group_dir':
        alpha = alpha[1]
        map_path = "{}/{}_M{}_{}_G{}_alpha{}.txt".format(map_dir, dataset_name, num_clients, partition, num_groups, alpha)
    elif partition == 'client_dir':
        alpha = alpha[1]
        map_path = "{}/{}_M{}_{}_G{}_alpha{}.txt".format(map_dir, dataset_name, num_clients, partition, num_groups,                                                 alpha)
    else:
        raise ValueError
    dataidx_map = Partitioner.read_dataidx_map(map_path)
    return dataidx_map


class Partitioner():
    def __init__(self):
        pass

    def partition_data(self):
        r"""Partition data indices to clients.
        Returns:
            dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
        """
        pass

    def gen_dataidx_map(self, labels, num_clients, num_classes, map_dir):
        r"""Generate dataidx_map"""
        dataidx_map = self.partition_data(labels, num_clients, num_classes)

        # Check the dataidx_map
        if CHECK == True:
            self.check_dataidx_map(dataidx_map, labels, num_clients, num_classes)
        map_path = "{}/{}_M{}_{}.txt".format(map_dir, self.dataset_name, num_clients, self.output_name)
        self.dumpmap(dataidx_map, map_path)

    @classmethod
    def read_dataidx_map(self, map_path):
        dataidx_map = self.loadmap(map_path)
        return dataidx_map

    @classmethod
    def check_dataidx_map(cls, dataidx_map=None, labels=None, num_clients=10, num_classes=10):
        r"""Check whether the map is reasonable by displaying some map information.
        Args:
            labels (numpy.ndarray, list): labels of the whole dataset
        """
        # Count the number of data samples per class per client
        n_sample_per_class_per_client = {cid: [] for cid in range(num_clients)}  # cid: client id
        for cid in range(num_clients):
            # number of data samples per class of any one client
            n_sample_per_class_one_client = [0 for _ in range(num_classes)]
            for j in range(len(dataidx_map[cid])):
                n_sample_per_class_one_client[int(labels[dataidx_map[cid][j]])] += 1
            n_sample_per_class_per_client[cid] = n_sample_per_class_one_client
        print("\n***** the number of samples per class per client *****")
        print(n_sample_per_class_per_client)

        # Count the number of samples per client
        n_sample_per_client = []
        for cid in range(num_clients):
            n_sample_per_client.append(sum(n_sample_per_class_per_client[cid]))
        n_sample_per_client = np.array(n_sample_per_client)
        print("\n***** the number of samples per client *****")
        # print(n_sample_per_client.mean(), n_sample_per_client.std())
        print(n_sample_per_client)

        # Count the number of samples per label
        n_sample_per_label = []
        n_client_per_label = []
        for i in range(num_classes):
            n_s = 0  # number of samples of any one label
            n_c = 0  # number of clients of any one label
            for j in range(num_clients):
                n_s = n_s + n_sample_per_class_per_client[j][i]
                n_c = n_c + int(n_sample_per_class_per_client[j][i] != 0)
            n_sample_per_label.append(n_s)
            n_client_per_label.append(n_c)
        n_sample_per_label = np.array(n_sample_per_label)
        n_client_per_label = np.array(n_client_per_label)
        print("\n*****the number of samples per label*****")
        print(n_sample_per_label)
        print("\n*****the number of clients per label*****")
        # print(n_client_per_label.mean(), n_client_per_label.std())
        print(n_client_per_label)

        cls.bubble(n_sample_per_class_per_client, num_clients, num_classes)
        # cls.heatmap(n_sample_per_class_per_client, num_clients, num_classes)

    @classmethod
    def bubble(cls, n_sample_per_class_per_client, num_clients, num_classes):
        r"""Draw bubble chart to display the local data distribution.
        Args:
            n_sample_per_class_per_client (set): { client id: [number of samples of Class 0, number of samples of Class 1, ...] }
        """
        x = []
        for i in range(num_clients):
            x.extend([i for _ in range(num_classes)])

        y = []
        for i in range(num_clients):
            y.extend([j for j in range(num_classes)])

        size = []
        for i in range(len(x)):
            size.append(n_sample_per_class_per_client[x[i]][y[i]])
        size = [i * 0.2 for i in size]

        plt.figure()
        plt.scatter(x, y, s=size, alpha=1)
        # plt.title(title)
        plt.xlabel("Client ID")
        plt.ylabel("Label")
        # plt.savefig('./raw_partition/{}/{}.png'.format(dataset, title))
        plt.show()

    @classmethod
    def heatmap(cls, n_sample_per_class_per_client, num_clients, num_classes):
        r"""Draw heat map to display the local data distribution"""
        num_sample_per_client = []
        heatmap_data = np.zeros((num_classes, num_clients), int)
        for i in range(num_clients):
            heatmap_data[:, i] = np.array(n_sample_per_class_per_client[i])
            num_sample_per_client.append(sum(n_sample_per_class_per_client[i]))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.heatmap(heatmap_data, ax=ax, annot=True, fmt="d", linewidths=.9, cmap="YlGn", )  #
        ax.set_xticklabels(['{}'.format(i) for i in range(num_clients)], rotation=0)
        # ax.set_xticklabels(['{} ({})'.format(i, num_sample_per_client[i]) for i in range(num_clients)], rotation=0)
        ax.set_yticklabels([str(i) for i in range(num_classes)], rotation=0)
        ax.set_xlabel("Client ID", fontsize=15)
        ax.set_ylabel("Label", fontsize=15)
        # ax.set_title(title, fontsize=16)
        # plt.savefig('./raw_partition/{}/{}.png'.format(dataset, title), bbox_inches='tight')
        plt.show()

    @classmethod
    def dumpmap(cls, dataidx_map, map_path):
        for i in range(len(dataidx_map)):
            if isinstance(dataidx_map[i], list) == False:
                dataidx_map[i] = dataidx_map[i].tolist()
        with open(map_path, 'w') as f:
            json.dump(dataidx_map, f)

    @classmethod
    def loadmap(cls, map_path):
        with open(map_path, 'r') as f:
            temp = json.load(f)
        # Since `json.load` will form dict{ '0': [] }, instead of dict{ 0: [] },
        # we need to turn dict{ '0': [] } to dict{ 0: [] }
        dataidx_map = dict()
        for i in range(len(temp)):
            dataidx_map[i] = np.array(temp[str(i)])
        return dataidx_map


class IIDPartitioner(Partitioner):
    r"""https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py"""

    def __init__(self, dataset_name='sst2'):
        super(IIDPartitioner, self).__init__()
        self.name = 'iid'
        self.dataset_name = dataset_name
        self.output_name = self.name

    def partition_data(self, labels, num_clients, num_classes):
        # Note: now 'balance' is ready, 'unbalance' is not completed (yipeng, 2023-11-14)
        num_labels = len(labels)
        idxs = np.random.permutation(num_labels)
        client_idxs = np.array_split(idxs, num_clients)
        dataidx_map = {cid: client_idxs[cid] for cid in range(num_clients)}
        return dataidx_map


class DirPartitioner(Partitioner):
    r"""The implementation of Dir-paritition way is from
    https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    """

    def __init__(self, dataset_name='sst2', alpha=10.0):
        super(DirPartitioner, self).__init__()
        self.name = 'dir'
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.output_name = '{}{}'.format(self.name, self.alpha)

    def partition_data(self, labels, num_clients, num_classes):
        alpha = self.alpha
        min_size = 0
        min_require_size = 10  # the minimum size of samples per client is required to be 10
        num_labels = len(labels)
        labels = np.array(
            labels)  # Note: to make `np.where(labels == k)[0]` succesful, turn labels to `np.ndarray` (yipeng, 2023-11-14)

        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)]  # data sample indices per client
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]  # data sample indices of class k
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Note: Balance the number of data samples of clients.
                # Don't assign samples to client j when its number of data samples is larger than the average (yipeng, 2023-11-14)
                proportions = np.array(
                    [p * (len(idx_j) < num_labels / num_clients) for p, idx_j in zip(proportions, idx_per_client)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(idx_per_client, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        dataidx_map = {}
        for j in range(num_clients):
            np.random.shuffle(idx_per_client[j])
            dataidx_map[j] = idx_per_client[j]
        return dataidx_map


class ExDirPartitioner(Partitioner):
    def __init__(self, dataset_name='sst2', C=10, alpha=10.0):
        super(ExDirPartitioner, self).__init__()
        self.name = 'exdir'
        self.dataset_name = dataset_name
        self.C, self.alpha = C, alpha
        self.output_name = '{}{},{}'.format(self.name, self.C, self.alpha)

    def allocate_classes(self, num_clients, num_classes, p_classes):
        '''Allocate `C` classes to each client
        Returns:
            clientidx_map (dict): { class id (int): client indices (list) }
        '''
        min_size_per_class = 0
        # min_require_size_per_class = max(self.C * num_clients // num_classes // 5, 1)
        min_require_size_per_class = 1
        while min_size_per_class < min_require_size_per_class:
            clientidx_map = {k: [] for k in range(num_classes)}
            for cid in range(num_clients):
                slected_classes = np.random.choice(range(num_classes), self.C, replace=False, p=p_classes)
                for k in slected_classes:
                    clientidx_map[k].append(cid)
            min_size_per_class = min([len(clientidx_map[k]) for k in range(num_classes)])
        return clientidx_map

    def partition_data(self, labels, num_clients, num_classes):
        C, alpha = self.C, self.alpha
        labels = np.array(labels)
        min_size = 0
        min_require_size = 1  # 10
        num_examples = len(labels)

        """2024-11-08. Dealing with Logistic Regression in JMLR.
        If the classes are allocated with equal probability,
        the number of examples will vary largely,
        when meeting the probability of each class varies largely
        """
        # p_classes = [] # compute the probability of each class
        # for i in range(num_classes):
        #     p = len(np.where(labels == i)[0])
        #     p_classes.append(p / num_examples)
        # print(p_classes)
        # we provide the previous setting as a choice. 2024-11-10.
        p_classes = [1 / num_classes for _ in range(num_classes)]

        clientidx_map = self.allocate_classes(num_clients, num_classes, p_classes)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[cid]) for cid in range(num_classes)])

        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Case 1 (original case in Dir): Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_examples / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                     enumerate(zip(proportions, idx_per_client))])
                # Case 2: Don't balance
                # proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients - 1):
                        proportions[w] = len(idx_k)

                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(idx_per_client, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        dataidx_map = {}
        for j in range(num_clients):
            np.random.shuffle(idx_per_client[j])
            dataidx_map[j] = idx_per_client[j]
        return dataidx_map


class ExDirBPartitioner(Partitioner):
    def __init__(self, dataset_name='sst2', C=10, alpha=10.0):
        super(ExDirBPartitioner, self).__init__()
        self.name = 'exdirb'
        self.dataset_name = dataset_name
        self.C, self.alpha = C, alpha
        self.output_name = '{}{},{}'.format(self.name, self.C, self.alpha)

    def allocate_classes(self, num_clients, num_classes, p_classes):
        '''Allocate `C` classes to each client
        Returns:
            clientidx_map (dict): { class id (int): client indices (list) }
        '''
        min_size_per_class = 0
        # min_require_size_per_class = max(self.C * num_clients // num_classes // 5, 1)
        min_require_size_per_class = 1
        while min_size_per_class < min_require_size_per_class:
            clientidx_map = {k: [] for k in range(num_classes)}  # clientidx_map={0:[],1:[]……} 代表标签0分配给的客户端
            for cid in range(num_clients):
                slected_classes = np.random.choice(range(num_classes), self.C, replace=False, p=p_classes)
                for k in slected_classes:
                    clientidx_map[k].append(cid)
            min_size_per_class = min([len(clientidx_map[k]) for k in range(num_classes)])
        return clientidx_map

    def partition_data(self, labels, num_clients, num_classes):
        C, alpha = self.C, self.alpha
        labels = np.array(labels)
        min_size = 0
        min_require_size = 1  # 10
        num_examples = len(labels)

        """2024-11-08. Dealing with Logistic Regression in JMLR.
        If the classes are allocated with equal probability,
        the number of examples will vary largely,
        when meeting the probability of each class varies largely
        """
        p_classes = []  # compute the probability of each class
        for i in range(num_classes):
            p = len(np.where(labels == i)[0])
            p_classes.append(p / num_examples)
        print(p_classes)
        # p_classes = [1/num_classes for _ in range(num_classes)]

        clientidx_map = self.allocate_classes(num_clients, num_classes, p_classes)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[cid]) for cid in range(num_classes)])

        while min_size < min_require_size:
            idx_per_client = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Case 1 (original case in Dir): Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_examples / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                     enumerate(zip(proportions, idx_per_client))])
                # Case 2: Don't balance
                # proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients - 1):
                        proportions[w] = len(idx_k)

                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(idx_per_client, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        # Make the examples equal among all the clients
        base_examples_per_user = int(num_examples // num_clients)
        remaining_examples = num_examples % num_clients
        examples_per_user = [base_examples_per_user] * num_clients

        for i in range(remaining_examples):
            examples_per_user[i] += 1

        idx_clients = []
        for c in range(num_clients):
            idx_clients.extend(idx_per_client[c])

        dataidx_map = {}
        cum = np.concatenate((np.array([0]), np.cumsum(examples_per_user)))
        for c in range(num_clients):
            dataidx_map[c] = np.random.permutation(idx_clients[cum[c]: cum[c + 1]])

        return dataidx_map

class GroupIIDClientNonIIDPartitioner(Partitioner):
    """
    Partition data such that it's IID between groups of clients,
    but Non-IID (Dirichlet) within each group.
    """
    def __init__(self, dataset_name='sst2', num_groups=10, alpha=1.0):
        super(GroupIIDClientNonIIDPartitioner, self).__init__()
        self.name = 'group_dir'
        self.dataset_name = dataset_name
        self.num_groups = num_groups
        self.alpha = alpha
        self.output_name = '{}_G{}_alpha{}'.format(self.name, self.num_groups, self.alpha)

    def partition_data(self, labels, num_clients, num_classes):
        """
        Partitions data: IID across groups, Non-IID (Dirichlet) within groups.

        Args:
            labels (np.ndarray or list): Labels of the entire dataset.
            num_clients (int): Total number of clients (M).
            num_classes (int): Number of classes in the dataset.

        Returns:
            dict: dataidx_map { global_client_id: np.ndarray(indices) }
        """
        num_groups = self.num_groups
        alpha = self.alpha
        labels = np.array(labels) # 确保是 numpy array
        num_samples = len(labels)

        clients_per_group = num_clients // num_groups


        # 全局 IID 划分到组
        all_indices = np.random.permutation(num_samples)
        # 将所有索引分成 G 组，每组获得一份 IID 的数据子集
        group_indices_split = np.array_split(all_indices, num_groups)

        # 最终的客户端数据索引映射
        dataidx_map = {}
        current_client_id = 0 # 全局客户端ID计数器


        # 对每个组内部进行 Non-IID (Dirichlet) 划分
        for g in range(num_groups):
            group_indices = group_indices_split[g] # 当前组获得的样本索引 (全局索引)
            group_labels = labels[group_indices]   # 当前组对应的标签
            num_group_samples = len(group_indices)

            if num_group_samples == 0:
                # 为该组的客户端分配空列表
                for _ in range(clients_per_group):
                     if current_client_id < num_clients:
                        dataidx_map[current_client_id] = np.array([], dtype=int)
                        current_client_id += 1
                continue

            min_size = 0
             # 对每个组内部设置一个最小样本要求，可以设小一点，因为组内数据量少了
            min_require_size_per_client_in_group = 5 # 或者根据需要调整

            indices_per_local_client = None # 存储当前组内部分配结果

            while min_size < min_require_size_per_client_in_group:
                # 存储当前组内每个本地客户端的索引
                indices_per_local_client = [[] for _ in range(clients_per_group)]

                for k in range(num_classes):
                    # 找到当前组中类别为k的样本的本地索引
                    k_local_indices_in_group = np.where(group_labels == k)[0]
                    if len(k_local_indices_in_group) == 0:
                        continue # 当前组没有这个类别的样本

                    # 获取这些样本对应的全局索引
                    k_global_indices = group_indices[k_local_indices_in_group]
                    np.random.shuffle(k_global_indices)

                    # 为当前组内的 clients_per_group 个客户端生成Dir
                    proportions = np.random.dirichlet(np.repeat(alpha, clients_per_group))

                    proportions = proportions / proportions.sum()


                    proportions_indices = (np.cumsum(proportions) * len(k_global_indices)).astype(int)[:-1]
                    splitted_global_indices = np.split(k_global_indices, proportions_indices)


                    indices_per_local_client = [local_list + global_idx_part.tolist()
                                                for local_list, global_idx_part
                                                in zip(indices_per_local_client, splitted_global_indices)]

                # 计算当前组内客户端的最小样本量
                current_group_client_sizes = [len(idx_j) for idx_j in indices_per_local_client]
                if not current_group_client_sizes: # 如果组内一个样本都没分出去
                     min_size = 0
                else:
                     min_size = min(current_group_client_sizes)


                if min_size == 0 and num_group_samples > 0 and all(p == 0 for p in proportions):
                     break


            # 组内部分配完成，映射到全局客户端ID
            for local_client_idx in range(clients_per_group):
                if current_client_id < num_clients:
                    global_client_id = current_client_id
                    client_indices = np.array(indices_per_local_client[local_client_idx])
                    np.random.shuffle(client_indices) # 打乱每个客户端的样本顺序
                    dataidx_map[global_client_id] = client_indices
                    current_client_id += 1
                else:
                     break


        return dataidx_map


class GroupNonIIDClientIIDPartitioner(Partitioner):
    """
    Partition data such that it's Non-IID (Dirichlet based on labels) between groups,
    but IID within each group.
    """
    def __init__(self, dataset_name='sst2', num_groups=10, alpha=1.0):
        super(GroupNonIIDClientIIDPartitioner, self).__init__()
        self.name = 'client_dir'
        self.dataset_name = dataset_name
        self.num_groups = num_groups
        self.alpha = alpha
        self.output_name = '{}_G{}_alpha{}'.format(self.name, self.num_groups, self.alpha)

    def partition_data(self, labels, num_clients, num_classes):
        """
        Partitions data: Non-IID (Dirichlet by label) across groups, IID within groups.

        Args:
            labels (np.ndarray or list): Labels of the entire dataset.
            num_clients (int): Total number of clients (M).
            num_classes (int): Number of classes in the dataset.

        Returns:
            dict: dataidx_map { global_client_id: np.ndarray(indices) }
        """
        num_groups = self.num_groups
        alpha = self.alpha
        labels = np.array(labels) # 确保是 numpy array
        num_samples = len(labels)

        clients_per_group = num_clients // num_groups


        # 组间 Non-IID 分配
        # 创建 G 个列表，用来存储每个组最终获得的样本索引
        indices_per_group = [[] for _ in range(num_groups)]
        indices_per_group_labels = [[] for _ in range(num_groups)] # 同时记录标签，方便检查

        for k in range(num_classes):
            # 获取类别 k 的所有样本的全局索引
            idx_k = np.where(labels == k)[0]
            if len(idx_k) == 0:
                continue
            np.random.shuffle(idx_k) # 打乱当前类别的样本顺序

            # 为 G 个组生成狄利克雷分布比例 (alpha 控制组间差异)
            proportions = np.random.dirichlet(np.repeat(alpha, num_groups))
            proportions = proportions / proportions.sum() # 归一化

            # 计算分割点
            proportions_indices = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # 根据比例分割类别 k 的索引
            splitted_k_indices = np.split(idx_k, proportions_indices)

            # 将分割后的索引分配给对应的组
            for g in range(num_groups):
                indices_per_group[g].extend(splitted_k_indices[g].tolist())
                # 记录对应的标签 (仅用于下面的打印检查)
                indices_per_group_labels[g].extend(labels[splitted_k_indices[g]].tolist())

            # print(f"  类别 {k} (共 {len(idx_k)} 个) 已分配给各组，比例大致为: {[len(s)/len(idx_k) if len(idx_k)>0 else 0 for s in splitted_k_indices]}")


        total_assigned = 0
        for g in range(num_groups):
            group_size = len(indices_per_group[g])
            total_assigned += group_size
            if group_size > 0:
                 group_label_counts = {cls: count for cls, count in zip(*np.unique(indices_per_group_labels[g], return_counts=True))}
            np.random.shuffle(indices_per_group[g])


        # 组内 IID 分配
        dataidx_map = {}
        current_client_id = 0 # 全局客户端ID计数器

        for g in range(num_groups):
            group_global_indices = np.array(indices_per_group[g]) # 当前组拥有的所有全局索引
            num_group_samples = len(group_global_indices)


            if num_group_samples == 0:
                # 为该组的客户端分配空列表
                for _ in range(clients_per_group):
                    if current_client_id < num_clients:
                        dataidx_map[current_client_id] = np.array([], dtype=int)
                        current_client_id += 1
                continue

            # 使用 array_split 将组内样本尽可能均匀地分给 clients_per_group 个客户端
            client_indices_split = np.array_split(group_global_indices, clients_per_group)

            # 将分割后的索引块分配给对应的全局客户端ID
            group_client_sizes = []
            for local_client_idx in range(clients_per_group):
                 if current_client_id < num_clients:
                     global_client_id = current_client_id
                     # client_indices_split[local_client_idx] 已经是 numpy array
                     dataidx_map[global_client_id] = client_indices_split[local_client_idx]
                     group_client_sizes.append(len(dataidx_map[global_client_id]))
                     current_client_id += 1
                 else:
                     break

        return dataidx_map


# python sim/data/partition_SST2.py -d sst2 -n 10 --partition exdir -C 1 --alpha 1.0
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='sst2', help='dataset name')
    parser.add_argument('-n', type=int, default=100, help='divide into n clients')
    parser.add_argument('--partition', type=str, default='iid', help='iid')
    parser.add_argument('--balance', type=bool, default=True, help='balanced or imbalanced')
    parser.add_argument('--alpha', type=float, default=1.0, help='the alpha of dirichlet distribution')
    parser.add_argument('-C', type=int, default=1, help='the classes of pathological partition')
    parser.add_argument('-G', '--num_groups', type=int, default=10,help='Number of client groups G (used in group_dir)')

    args = parser.parse_args()
    print(args)

    dataset_dir = '../datasets/'  # the directory path of datasets
    output_dir = 'D:\Code\Hierarchical-Fed-Learning\maps'  # 'maps/raw/' the directory path of outputs
    dataset_name = args.d  # the name of the dataset
    num_clients = args.n  # number of clients
    partition = args.partition  # partition way
    balance = args.balance
    alpha = args.alpha
    C = args.C
    num_groups = args.num_groups

    # Prepare the dataset
    # SST2 是二分类数据集：0 (negative), 1 (positive)
    num_class_dict = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10, 'sst2': 2, 'test': 4}
    
    # 加载 SST2 数据集
    # 注意：这里假设您已经实现了 build_dataset 函数来支持 SST2
    # 如果使用 HuggingFace datasets，可以这样加载：
    try:
        from datasets import load_dataset
        # 加载 SST2 数据集
        dataset = load_dataset('glue', 'sst2')
        train_dataset = dataset['train']
        # 获取标签列表
        labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
    except ImportError:
        # 如果 datasets 库不可用，尝试使用 build_dataset
        try:
            origin_dataset = build_dataset(dataset_name=dataset_name, dataset_dir=dataset_dir)
            train_dataset = origin_dataset.get_trainset()
            # 如果 train_dataset 有 targets 属性
            if hasattr(train_dataset, 'targets'):
                labels = list(train_dataset.targets)
            # 如果 train_dataset 是 HuggingFace 数据集格式
            elif hasattr(train_dataset, '__getitem__'):
                labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
            else:
                raise ValueError("无法从数据集中提取标签，请检查数据集格式")
        except:
            raise ImportError("请安装 datasets 库: pip install datasets，或实现 build_dataset 函数支持 SST2")
    
    num_classes = num_class_dict[dataset_name]

    if partition == 'iid':
        p = IIDPartitioner(dataset_name=dataset_name)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'dir':
        p = DirPartitioner(dataset_name=dataset_name, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'exdir':
        p = ExDirPartitioner(dataset_name=dataset_name, C=C, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'exdirb':
        p = ExDirBPartitioner(dataset_name=dataset_name, C=C, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'group_dir':
        p = GroupIIDClientNonIIDPartitioner(dataset_name=dataset_name, num_groups=num_groups, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)
    elif partition == 'client_dir':
        p = GroupNonIIDClientIIDPartitioner(dataset_name=dataset_name, num_groups=num_groups, alpha=alpha)
        p.gen_dataidx_map(labels=labels, map_dir=output_dir, num_clients=num_clients, num_classes=num_classes)


if __name__ == '__main__':
    try:
        from datasets import build_dataset
    except ImportError:
        pass
    main()

