from data_loader import load_data
from symmetric_sne import SymmetricSNE
import numpy as np
import os


if __name__=='__main__':

    # Take a subset of the samples
    samples_per_class = 200
    classes = [0, 1, 8]
    #X_mnist, y_minst = load_data('mnist', save_if_not_found=True, classes=classes, samples_per_class=200)
    #X_orl, y_orl = load_data('orl', save_if_not_found=True)
    X_coil, y_coil = load_data('coil20', save_if_not_found=True)

    X_coil = X_coil.reshape(X_coil.shape[0], -1)
    SymmetricSNE = SymmetricSNE(perplexity=40)

    X_sym_sne = SymmetricSNE.fit_transform(X_coil, save_joint_probs=True, save_P_file='data/coil_P.pkl')
