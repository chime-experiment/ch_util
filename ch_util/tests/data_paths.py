"""Module contains paths to test data."""


import glob
from os import path

path_root = path.join(path.dirname(path.realpath(__file__)), "data")

dir1_0 = path.join(path_root, "1.0")
paths1_0 = tuple(sorted(glob.glob(path.join(dir1_0, "test_acq.h5.*"))))

dir2_0 = path.join(path_root, "2.0")
paths2_0 = tuple(sorted(glob.glob(path.join(dir2_0, "*.h5"))))

dir2_2 = path.join(path_root, "2.2")
paths2_2 = tuple(sorted(glob.glob(path.join(dir2_2, "*.h5"))))


dir3_1 = path.join(path_root, "3.1")
archive3_1 = path.join(dir3_1, "testdata.tar.gz")
