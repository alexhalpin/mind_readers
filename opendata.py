import h5py
import numpy as np

with h5py.File("./fMRI data/sub-01_imagery_original_VC.h5", "r") as f:
    print(f["dataset"][()].shape)

    

