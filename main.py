import torch
import torch.nn as nn
from torch.nn import functional as F

def methods():
    while True:
        choice_ = int(input("Next layer to be appended...\n1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;\n5.Quit\n"))

        if choice_ > 5 or ValueError:
            print("Please type in the appropriate key\n")
            pass


if __name__ == "__main__":
    print("..playground.cnn")
    methods()

