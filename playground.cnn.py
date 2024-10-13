import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ModelBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Builder")

        self.model = nn.Sequential()
        self.trainloader = None
        self.input_channels = 0
        self.input_size = 0
        self.current_size = 0
        self.conv_counter = 0
        self.prev_out = 0

        # Create buttons for actions
        self.add_layer_button = tk.Button(root, text="Add Layer", command=self.add_layer_menu)
        self.add_layer_button.pack(pady=5)

        self.choose_dataset_button = tk.Button(root, text="Choose Dataset", command=self.choose_dataset)
        self.choose_dataset_button.pack(pady=5)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Model", command=self.save_model)
        self.save_button.pack(pady=5)

        self.display_button = tk.Button(root, text="Display Layers", command=self.display_layers)
        self.display_button.pack(pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=5)

    def add_layer_menu(self):
        layer_choice = simpledialog.askinteger("Layer Type", "Choose Layer:\n1. Conv2D\n2. Normalization\n3. Dropout\n4. Pooling")

        if layer_choice == 1:
            self.add_conv_layer()
        elif layer_choice == 2:
            self.add_normalization_layer()
        elif layer_choice == 3:
            self.add_dropout_layer()
        elif layer_choice == 4:
            self.add_pooling_layer()
        else:
            messagebox.showerror("Error", "Invalid choice")

    def add_conv_layer(self):
        if self.conv_counter == 0:
            in_channels = self.input_channels
            out_channels = int(simpledialog.askinteger("Input", "Enter number of output channels:"))
            kernel_size = int(simpledialog.askinteger("Input", "Enter kernel size:"))
            self.prev_out = out_channels
            self.current_size = self.calculate_output_size(self.input_size, kernel_size, stride=1, padding=1)
        else:
            in_channels = self.prev_out
            out_channels = int(simpledialog.askinteger("Input", "Enter number of output channels:"))
            kernel_size = int(simpledialog.askinteger("Input", "Enter kernel size:"))
            self.prev_out = out_channels
            self.current_size = self.calculate_output_size(self.current_size, kernel_size, stride=1, padding=1)

        self.model.add_module(f"conv2d_{len(self.model)}", nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        self.model.add_module(f"relu_{len(self.model)}", nn.ReLU())

        # Prompt for optional normalization
        add_norm = simpledialog.askinteger("Input", "Add normalization?\n1. Yes\n2. No")
        if add_norm == 1:
            self.model.add_module(f"batchnorm_{len(self.model)}", nn.BatchNorm2d(out_channels))

        self.conv_counter += 1
        messagebox.showinfo("Layer Added", "Conv2D Layer Added")

    def add_normalization_layer(self):
        if self.conv_counter == 0:
            messagebox.showerror("Error", "Add Conv2D layer first")
            return

        self.model.add_module(f"batchnorm_{len(self.model)}", nn.BatchNorm2d(self.prev_out))
        messagebox.showinfo("Layer Added", "Normalization Layer Added")

    def add_dropout_layer(self):
        dropout_rate = float(simpledialog.askfloat("Input", "Enter dropout rate:"))
        self.model.add_module(f"dropout_{len(self.model)}", nn.Dropout(dropout_rate))
        messagebox.showinfo("Layer Added", "Dropout Layer Added")

    def add_pooling_layer(self):
        pool_type = simpledialog.askinteger("Input", "Choose pooling layer type:\n1. MaxPool2D\n2. AvgPool2D\n3. AdaptiveAvgPool2D")
        if pool_type == 1:
            pool_size = int(simpledialog.askinteger("Input", "Enter pool size:"))
            self.model.add_module(f"maxpool2d_{len(self.model)}", nn.MaxPool2d(pool_size))
            self.current_size = self.calculate_output_size(self.current_size, pool_size, stride=2)
        elif pool_type == 2:
            pool_size = int(simpledialog.askinteger("Input", "Enter pool size:"))
            self.model.add_module(f"avgpool2d_{len(self.model)}", nn.AvgPool2d(pool_size))
            self.current_size = self.calculate_output_size(self.current_size, pool_size, stride=2)
        elif pool_type == 3:
            self.model.add_module(f"adaptiveavgpool2d_{len(self.model)}", nn.AdaptiveAvgPool2d(1))
            self.current_size = 1
        else:
            messagebox.showerror("Error", "Invalid pool type")

        messagebox.showinfo("Layer Added", "Pooling Layer Added")

    def choose_dataset(self):
        dataset_choice = simpledialog.askstring("Input", "Choose a dataset: CIFAR-10, Fashion MNIST, MNIST")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        if dataset_choice == 'CIFAR-10':
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            self.input_channels = 3
            self.input_size = 32
        elif dataset_choice == 'Fashion MNIST':
            trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            self.input_channels = 1
            self.input_size = 28
        elif dataset_choice == 'MNIST':
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            self.input_channels = 1
            self.input_size = 28
        else:
            messagebox.showerror("Error", "Invalid dataset choice")
            return

        self.conv_counter = 0  # Reset the conv counter when a new dataset is chosen
        self.model = nn.Sequential()  # Reset the model when a new dataset is chosen
        messagebox.showinfo("Dataset Selected", f"Dataset {dataset_choice} loaded. Input channels: {self.input_channels}, Input size: {self.input_size}")
        self.trainloader = data.DataLoader(trainset, batch_size=16, shuffle=True)
        self.current_size = self.input_size
        messagebox.showinfo("Dataset Selected", f"Dataset {dataset_choice} loaded")


    def train_model(self):
        if self.trainloader is None:
            messagebox.showerror("Error", "Please choose a dataset first")
            return

        self.model.add_module(f"flatten_{len(self.model)}", nn.Flatten())
        fc_input_size = self.prev_out * self.current_size * self.current_size
        self.model.add_module(f"linear_{len(self.model)}", nn.Linear(fc_input_size, 10))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            progress_bar = tqdm(self.trainloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in progress_bar:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                accuracy = self.calculate_accuracy(outputs, targets)
                epoch_accuracy += accuracy
                progress_bar.set_postfix(loss=epoch_loss/len(self.trainloader), accuracy=epoch_accuracy/len(self.trainloader))

            avg_loss = epoch_loss / len(self.trainloader)
            avg_accuracy = epoch_accuracy / len(self.trainloader)
            messagebox.showinfo("Training Complete", f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    def save_model(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch models", "*.pth")])
        if filepath:
            torch.save(self.model.state_dict(), filepath)
            messagebox.showinfo("Model Saved", f"Model saved to {filepath}")

    def display_layers(self):
        details = "\n".join([str(layer) for layer in self.model.children()])
        messagebox.showinfo("Layer Details", details)

    def calculate_output_size(self, input_size, kernel_size, stride=1, padding=0):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def calculate_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)
    
def run_tkinter():
    root = tk.Tk()
    app = ModelBuilderApp(root)
    root.mainloop()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/open-model-builder")
async def open_model_builder():
    threading.Thread(target=run_tkinter, daemon=True).start()
    return {"message": "Model Builder opened"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

