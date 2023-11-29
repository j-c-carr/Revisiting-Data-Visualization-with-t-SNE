from data_loader import load_data
import numpy as np
import os


if __name__=='__main__':
    X_mnist, y_minst = load_data('mnist', save_if_not_found=True)
    X_orl, y_orl = load_data('orl', save_if_not_found=True)
    X_coil, y_coil = load_data('coil20', save_if_not_found=True)

    print(X.shape, y.shape)