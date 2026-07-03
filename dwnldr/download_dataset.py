#! /usr/bin/python3
# Created by AG on 03-07-2026

# This repo uses the MNIST dataset for model training. If anyone has/knows of better datasets, please let me know.

import os
import gzip
import numpy
import struct
import urllib.request

data_directory =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

download_urls = {
    "training_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "training_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}
