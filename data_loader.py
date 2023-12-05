import os
import pickle
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml, fetch_olivetti_faces
import logging
logging.basicConfig(level=logging.INFO)


def load_data(dset_name, save_if_not_found=False, classes=None, samples_per_class=None):
    """Loads mnist, olivetti faces (orl) and coil20 datasets. If the data is not saved locally
    in a pickle file at :datasets/:dset_name:.pkl, then it will download the data. If :save_if_not_found:
    is True, then downloaded data is saved to :datasets/:dset_name:.pkl."""

    assert dset_name in ['mnist', 'orl', 'coil20'], f'No dataset named {dset_name}'

    if not os.path.isfile(f'data/datasets/{dset_name}.pkl'):
        logging.debug(f'File datasets/{dset_name}.pkl not found.')
        logging.debug(f'Downloading {dset_name}...')
        if dset_name == 'mnist':
            mnist = fetch_openml('mnist_784', version=1)
            X = mnist.data.to_numpy() / 255.0
            y = mnist.target.astype(int).to_numpy()

        elif dset_name == 'orl':
            olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
            X = olivetti_faces.data
            y = olivetti_faces.target

        else:
            X, y = convert_coil20_from_png('data/datasets/coil-20-proc')

        logging.debug('Done.')

    else:   # otherwise, load data from dataset
        logging.debug(f'Loading data from datasets/{dset_name}.pkl...')
        with open(f'data/datasets/{dset_name}.pkl', 'rb') as f:
            data = pickle.load(f)
        logging.debug('Done.')

        X = data['X']
        y = data['y']

    if not os.path.isfile(f'data/datasets/{dset_name}.pkl') and save_if_not_found:
        data = {'X': X, 'y': y}
        logging.debug(f'Saving dataset to datasets/{dset_name}.pkl...')
        with open(f'data/datasets/{dset_name}.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.debug('Done.')

    if classes is not None and samples_per_class is not None:
        return get_subset(X, y, classes, samples_per_class)

    return X, y

def convert_coil20_from_png(directory):
    """Converts png images from COIL-20 dataset into numpy arrays.
    Assumes :directory: is directory containing the  1440 *processed* Coil-20 images.
    A compressed version of this directory can be downloaded from downloaded from:
        https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
    """

    X = np.empty((1440, 128, 128))
    y = np.empty((1440,))

    for i, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        img = Image.open(f)
        X[i] = np.array(img)

        # filename is objLL__xyz.png, where LL is the class label
        label = filename[3:5]
        if label[1] == '_': # remove trailing underscore for single digits
            label = label[0]

        y[i] = int(label)

    return X, y


def get_subset(X, y, classes, samples_per_class):
    """Select a subset of the data with size len(:classes:)*samples_per_class"""
    assert min(classes) >= y.min() and max(classes) <= y.max(), 'Classes out of range.'

    idx = []
    for cls in classes:
        idx += list(np.where(y == cls)[0])[:samples_per_class]

    return X[idx], y[idx]