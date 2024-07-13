import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import easygui
from tqdm import tqdm

def print_layer_details(model):
    details = ""
    for layer in model.children():
        details += f"                                                             \n"
        details += f"+-----------------------------------------------------------+\n"
        details += f"| {str(layer):<40}                                          |\n" 
        details += f"+-----------------------------------------------------------+\n"
        details += f"                                                             \n"

    easygui.msgbox(details, title="Layer Details")

def save_model(model):
    filepath = easygui.filesavebox(title="Save model", default="model.pth")
    if filepath:
        torch.save(model.state_dict(), filepath)
        easygui.msgbox(f"Model saved to {filepath}", title="Save model")

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def train_model(model):
    # Dummy dataset
    inputs = torch.randn(64, 3, 32, 32)
    targets = torch.randint(0, 10, (64,))

    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            accuracy = calculate_accuracy(outputs, targets)
            epoch_accuracy += accuracy
            
            progress_bar.set_postfix(loss=epoch_loss/len(dataloader), accuracy=epoch_accuracy/len(dataloader))
        
        average_loss = epoch_loss / len(dataloader)
        average_accuracy = epoch_accuracy / len(dataloader)
    
    easygui.msgbox(f"Epoch {epoch + 1}/{epochs} completed with loss: {average_loss:.4f}, accuracy: {average_accuracy:.4f}", title="Training Progress")


def methods(model):
    conv_counter=0
    prev_out=0
    while True:
        try:
            choice_ = int(input("Layer to be appended...\n1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;\n5.Save; 6.Display layers; 7.Train; 8.Quit\n"))
        except ValueError:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ < 1 or choice_ > 8:
            print("\n\033[31mPlease type in the appropriate key\033[0m\n")
            continue

        if choice_ == 1:
            conv_counter += 1

            if conv_counter != 1:
                
                out_channels = int(input("Enter number of output channels: "))                
                kernel_size = int(input("Enter kernel size: "))
                in_channels = prev_out

            else:
                in_channels = int(input("Enter number of input channels: "))
                out_channels = int(input("Enter number of output channels: "))
                kernel_size = int(input("Enter kernel size: "))
                prev_out = out_channels

            model.add_module(f"conv2d_{len(model)}", nn.Conv2d(in_channels, out_channels, kernel_size))
            model.add_module(f"relu_{len(model)}", nn.ReLU())


            while True:
                try:
                    add_norm = int(input("Do you want to add normalization 1.yes 2.no): "))

                except ValueError:
                    print("\n\033[31mPlease type in the appropriate key\033[0m\n")
                    continue

                if add_norm == 2:
                    break
                else:
                    model.add_module(f"batchnorm_{len(model)}", nn.BatchNorm2d(out_channels))
                    break



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
            save_model(model)

        elif choice_ == 6:
            print_layer_details(model)

        elif choice_ == 7:
            model.add_module(f"flatten_{len(model)}", nn.Flatten())
            model.add_module(f"linear_{len(model)}", nn.Linear(6272, 10))
            try:
                train_model(model)
            except RuntimeError as e:
                print(e)
                break
            
            
        elif choice_ == 8:
            print("Exiting...")
            break

if __name__ == "__main__":
    model = nn.Sequential()
 
    methods(model)
