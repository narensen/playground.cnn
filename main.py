
import torch.nn as nn
import easygui

def print_layer_details(model):
    details = ""
    for layer in model.children():
        details += f"                                                             \n"
        details += f"+-----------------------------------------------------------+\n"
        details += f"| {str(layer):<40}                                          |\n" 
        details += f"+-----------------------------------------------------------+\n"
        details += f"                                                             \n"

    easygui.msgbox(details, title="Layer Details")

def methods(model):
    while True:
        try:
            choice_ = int(input("Next layer to be appended...\n1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;\n5.Quit; 6.Display layers\n"))
        except ValueError:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ < 1 or choice_ > 6:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ == 1:
            in_channels = int(input("Enter number of input channels: "))
            out_channels = int(input("Enter number of output channels: "))
            kernel_size = int(input("Enter kernel size: "))
            model.add_module(f"conv2d_{len(model)}", nn.Conv2d(in_channels, out_channels, kernel_size))
            model.add_module(f"relu_{len(model)}", nn.ReLU())

        elif choice_ == 2:
            num_features = int(input("Enter num_features for BatchNorm2d: "))
            model.add_module(f"batchnorm_{len(model)}", nn.BatchNorm2d(num_features))

        elif choice_ == 3:
            dropout_rate = float(input("Enter dropout rate: "))
            model.add_module(f"dropout_{len(model)}", nn.Dropout(dropout_rate))

        elif choice_ == 4:
            pool_type = int(input("Choose pooling layer type:\n1. MaxPool2D\n2. AvgPool2D\n3. AdaptiveAvgPool2D\n"))
            if pool_type == 1:
                pool_size = int(input("Enter pool size: "))
                model.add_module(f"maxpool2d_{len(model)}", nn.MaxPool2d(pool_size))
            elif pool_type == 2:
                pool_size = int(input("Enter pool size: "))
                model.add_module(f"avgpool2d_{len(model)}", nn.AvgPool2d(pool_size))
            elif pool_type == 3:
                model.add_module(f"adaptiveavgpool2d_{len(model)}", nn.AdaptiveAvgPool2d(1))

        elif choice_ == 5:
            print("Exiting...")
            break

        elif choice_ == 6:
            print_layer_details(model)

if __name__ == "__main__":
    model = nn.Sequential()
    methods(model)
