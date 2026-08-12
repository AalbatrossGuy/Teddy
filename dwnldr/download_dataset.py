#! /usr/bin/python3
# Created by AG on 03-07-2026

# This repo uses the MNIST dataset for model training. If anyone has/knows of better datasets, please let me know.

import os
import gzip
from os.path import exists
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

def download_training_files(url, dest):
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)

def parse_index_images(file_path):
    with gzip.open(file_path, 'rb') as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        data = numpy.frombuffer(file.read(), dtype=numpy.uint8)
        cleaned_data = data.reshape(count, rows * columns).astype(numpy.float32) / 255.0

    return cleaned_data

def parse_index_labels(file_path):
    with gzip.open(file_path, 'rb') as file:
        magic, count = struct.unpack(">II", file.read(8))
        data = numpy.frombuffer(file.read(), dtype=numpy.uint8).astype(numpy.float32)

    return data

def main() :
    os.makedirs(data_directory, exist_ok=True)
    gzip_directory = os.path.join(data_directory, "gz")
    os.makedirs(gzip_directory, exist_ok=True)
    gzip_paths: dict = {}

    for key, url in download_urls.items():
        destination = os.path.join(gzip_directory, os.path.basename(url))
        download_training_files(url, dest=destination)
        gzip_paths[key] = destination

    training_images = parse_index_images(gzip_paths["training_images"])
    training_labels = parse_index_labels(gzip_paths["training_labels"])
    test_images = parse_index_images(gzip_paths["test_images"])
    test_labels = parse_index_labels(gzip_paths["test_labels"])

    training_images.tofile(os.path.join(data_directory, "training_images.bin"))
    training_labels.tofile(os.path.join(data_directory, "training_labels.bin"))

    test_images.tofile(os.path.join(data_directory, "test_images.bin"))
    test_labels.tofile(os.path.join(data_directory, "test_labels.bin"))

    print("========= Stats =========")
    print(f"Training images: {training_images.shape}")
    print(f"Training labels: {training_labels.shape}")
    print(f"Test images: {test_images.shape}")
    print(f"Test labels: {test_labels.shape}")
    print(f"Files saved in: {data_directory}")


if __name__ == "__main__":
    main()
