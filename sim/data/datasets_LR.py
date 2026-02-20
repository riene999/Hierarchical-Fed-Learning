import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file

def get_dataset(dataset, data_path='./'):
    # https://github.com/konstmish/random_reshuffling/tree/master
    if dataset in ['covtype', 'real-sim', 'webspam']:
        A, b = load_svmlight_file(data_path + dataset + '.bz2')
    elif dataset in ['a1a', 'a5a', 'a9a', 'mushrooms', 'gisette', 'w8a']:
        A, b = load_svmlight_file(data_path + dataset)
    elif dataset == 'rcv1':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
    elif dataset == 'YearPredictionMSD':
        A, b = load_svmlight_file(data_path + dataset + '.bz2')
        b = b > 2000
    elif dataset == 'rcv1_binary':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
    
    A = A.toarray()
    b = np.asarray(b)
    if (np.unique(b) == [1, 2]).all():
        # Transform labels {1, 2} to {0, 1}
        b = b - 1
    elif (np.unique(b) == [-1, 1]).all():
        # Transform labels {-1, 1} to {0, 1}
        b = (b+1) / 2
    else:
        assert (np.unique(b) == [0, 1]).all()
        b = b

    return A, b