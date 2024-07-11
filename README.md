# playground.cnn

Playground.cnn is a Python script designed to interactively build Convolutional Neural Network (CNN) models using PyTorch. This project aims to provide an intuitive and interactive way for rising ML engineers to learn and experiment with CNN architectures.

## Impact

The goal of this project is to help ML engineers understand the flexibility and adaptability of CNNs by allowing them to dynamically create and modify models. By providing an interactive interface, users can intuitively grasp the concepts of CNN layers and their configurations, facilitating a deeper understanding of model architecture design.

## Features

- **Interactive Layer Addition**: Add various layers to your CNN model interactively.
- **Layer Types Supported**: Convolutional layers, Normalization layers, Dropout layers, and Pooling layers.
- **Display Layer Details**: View the details of the current layers in the model in a GUI window using EasyGUI.

## Requirements

- Python 3.x
- PyTorch
- EasyGUI
- tqdm
## Installation

1. **Clone the Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/YourUsername/playground.cnn.git
   cd playground.cnn
Install Dependencies: Install the required Python packages.

`pip install torch easygui torch tqdm`

## Usage
Run the script and follow the prompts to add layers to your CNN model.
`playground.cnn.py`

Example Interaction
```bash
Next layer to be appended...
1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;
5.Quit; 6.Display layers
1
Enter number of input channels: 16
Enter number of output channels: 32
Enter kernel size: 3
Next layer to be appended...
1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;
5.Quit; 6.Display layers
2
Enter num_features for BatchNorm2d: 100
Next layer to be appended...
1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;
5.Quit; 6.Display layers
3
Enter dropout rate: 0.5
Next layer to be appended...
1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;
5.Quit; 6.Display layers
4
Choose pooling layer type:
1. MaxPool2D
2. AvgPool2D
3. AdaptiveAvgPool2D
1
Enter pool size: 2
Next layer to be appended...
1.Conv2D; 2.Normalization; 3.Dropout; 4.Pooling layer;
5.Quit; 6.Display layers
6
```
![Screenshot 2024-07-11 113741](https://github.com/narensen/playground.cnn/assets/106871870/6698cf24-3591-49f4-970c-e5fe3aaef7a7)
