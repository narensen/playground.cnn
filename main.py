import torch
import torch.nn as nn
from torch.nn import functional as F

def methods(model):
    while True:
        try:
            choice_ = int(input("Next layer to be appended...\n1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;\n5.Quit\n"))
        except ValueError:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ < 1 or choice_ > 5:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ == 1:
            in_channel = int(input("What are the in channels? "))
            out_channel = int(input("What are the out channels? "))
            kernel_size = int(input("What is the kernel size? "))
            model.add_module(f"conv2d_{len(model)}", nn.Conv2d(in_channel, out_channel, kernel_size))
        
        elif choice_ == 2:
            model.add_module(f"conv2d_{len(model)}", nn.BatchNorm2d())
        
        elif choice_ == 3:
            rate = float(input("What is the dropout rate? "))
            model.add_module(f"conv2d_{len(model)}", nn.Dropout(rate))
        
        elif choice_ == 4:
            try:
                pool_type = int(input("Which pooling layer\n 1.MaxPool2D; 2.AveragePool; 3.GlobalAveragePool;"))   

            except ValueError:
                print("\n\033[31mPlease type in the appropriate key\033[0m\n")
                continue

            if pool_type > 3 or pool_type < 1:
                print("\n\033[31mPlease type in the appropriate key\033[0m\n")
                continue

            elif pool_type == 1:
                pool_size = int(input("What is the pool size? "))
                model.add_module(f"conv2d_{len(model)}", nn.MaxPool2d(pool_size))

            elif pool_type == 2:
                pool_size = int(input("What is the pool size? "))
                model.add_module(f"conv2d_{len(model)}", nn.AvgPool2d(pool_size))

            elif pool_type == 3:
                model.add_module(f"conv2d_{len(model)}", nn.AdaptiveAvgPool2d(1))
        
        elif choice_ == 5:
            print("Exiting...")
            print(model)
            break


if __name__ == "__main__":
    print("..playground.cnn")
    model = nn.Sequential()
    methods(model)

