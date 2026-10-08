import numpy as np


def load_data(data_path):
    """
    Return the dataset as numpy arrays.

    Arguments:
        data_path (str): path to the dataset directory
    Returns:
        xtrain (array): images of the train set
        xtest (array): images of the test set (unlabelled)
        ytrain (array): labels of the train set, of shape (N,)
    """
    xtrain = np.load(data_path + '/train_data.npy', allow_pickle=True)
    ytrain = np.load(data_path + '/train_label.npy', allow_pickle=True)
    xtest = np.load(data_path + '/test_data.npy', allow_pickle=True)

    return xtrain, xtest, ytrain

