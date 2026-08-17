# Teddy
<img width="90" alt="teddy-logo" src="https://github.com/user-attachments/assets/67d0736b-19b9-4ba0-a06c-03d213135d68" align="right"/><br>
> A Machine Learning library in C because I want to ragebait myself

## Preface
I always had a deep appreciation for C as a language and the philosophy it follows. I then came across this [video](https://youtu.be/hL_n_GljC0I?si=qcqbFWwySUdKiIq) and I was genuinely impressed by what he did. I also have an interest in perceptrons and machine learning in general, hence it's safe to say that the video inspired me to make something like this. Teddy follows the same logic shown in the video (more or less) but it differs in implementation. I would also like to point out that Teddy is not vibecoded, in fact, 0% AI was used while building it. Although I did use AI to build the GIFs present in this README and the documentation of how Teddy functions simply because I didn't know how to make custom GIFs for this purpose and I didn't want to write docs. 

## How it works
<img width="800" height="445" alt="teddy_flow" src="https://github.com/user-attachments/assets/95f874a9-137b-4abb-9893-b6760a3bc69a" /> <br>
<img width="800" height="500" alt="teddy_neural_net" src="https://github.com/user-attachments/assets/d81a40a0-0e6e-4d28-97d1-0e571ca78f07" /> <br>

Teddy is a MLP (Multi Layer Perceptron) that can, as of now, classify MNIST datasets. At its core, Teddy is built the same way every modern deep learning framework is, just shrunk down. Underneath everything is plain array math wrapped in a small structure that keeps track of a value's shape alongside its numbers. On top of that layer sits the automatic differentiation engine. Instead of running each calculation instantly, Teddy builds up a graph of operations first — this value feeds into that one, which feeds into another — and only once the whole graph is assembled does it walk through and compute everything. That separation is what makes learning possible at all: after the graph produces its final answer, Teddy walks back through the exact same graph in reverse, applying the chain rule at every step to figure out how much each earlier value contributed to the final error.

Training Teddy is standard [stochastic gradient descent](https://www.mit.edu/~gfarina/2025/67220s25_L15_sgd/L15.pdf), just done one sample at a time. It runs a handful of examples through the graph, lets the errors accumulate across that little batch, and then nudges every learnable weight a small step in the direction that reduces the error — averaged over however many samples were in the batch, so one unusually easy or unusually hard example doesn't swing the weights too far on its own. Weights don't start at zero or at some arbitrary fixed value either; they're seeded with a small amount of carefully scaled randomness so that signals flowing through the network at the very start of training are neither vanishingly small nor wildly large, which in practice is the difference between a network that learns steadily and one that never gets off the ground.

Underneath all of that, Teddy also gets to decide where the actual computation happens. Every time it starts up, it first tries to hand the heavy lifting off to a GPU, since that's dramatically faster for the kind of repetitive matrix math machine learning is built on. If it can't find a compatible GPU on the machine it's running on — or the necessary drivers just aren't installed — it quietly falls back to doing the exact same math on the CPU instead, with no difference in behavior or accuracy.

Right now, the one thing Teddy has actually been taught to do is recognize handwritten digits from the MNIST datasets. Every training example is a small grayscale image, flattened into one long list of numbers, run through a couple of hidden layers with a shortcut connection between them so information doesn't have to funnel through every layer to get where it's going, and squeezed out the other end into ten numbers that together represent how confident the model is in each possible digit. Before training starts, those guesses are close to random. By the end of training, Teddy guesses the right number, slowly learning from the random start.

Rather than teaching Teddy how to parse image files, dataset preparation happens once, ahead of time, outside the core part of the library entirely: raw image data gets downloaded, decoded, and normalized into plain flat files of numbers that Teddy can read directly with no parsing logic of its own.

## Build & Run

```sh
make               # builds the training binary (auto-detects a usable GPU backend, falls back to CPU)
make FORCE_CPU=1   # force the CPU backend even if a GPU is available
make clean         # remove build output
make data          # download + preprocess the MNIST dataset
```

`make data` needs to run at least once before there's anything to train on. After that, just run the built binary — it'll pick a compute backend on its own and start training.

## Directory structure

```
Teddy/
├── Makefile
├── README.md
├── LICENSE
│
├── include/
│   ├── computation_engine.h
│   ├── compute_backend.h
│   ├── dataset_ops.h
│   ├── gpu_compute.h
│   ├── math_ops.h
│   ├── matrix_ops.h
│   └── model_train.h
│
├── src/
│   ├── computation_engine.c
│   ├── compute_backend.c
│   ├── dataset_ops.c
│   ├── gpu_compute.c
│   ├── math_ops.c
│   ├── matrix_ops.c
│   └── model_train.c
│
├── kernel/
│   └── opencl.c
│
├── train_teddy/
│   └── train_teddy.c
│
├── dwnldr/
│   ├── download_dataset.py
│   └── requirements.txt
│
├── assets/
│   └── generate_flowchart.py
│
├── data/
└── build/
```
