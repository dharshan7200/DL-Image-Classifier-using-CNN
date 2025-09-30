# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import necessary libraries for PyTorch, torchvision, visualization, and evaluation metrics.

### STEP 2: 

Define image transformations: convert tensors and normalize dataset between -1 and 1.


### STEP 3: 

Load MNIST training and testing datasets with transformations and enable downloading automatically.


### STEP 4: 

Create DataLoader objects for training and testing datasets with batch processing.


### STEP 5: 

Define CNNClassifier class with convolution, pooling, and fully connected layers.


### STEP 6: 

Implement forward function using ReLU activations, pooling, and flattening before classification.

### STEP 7: 

Initialize CNN model, move to GPU if available, and display summary.

### STEP 8:

Define cross-entropy loss function and Adam optimizer with learning rate 0.001.


### STEP 9:

Train model for epochs: forward pass, compute loss, backpropagate, and update parameters.


### STEP 10: 

Test model: predict outputs, calculate accuracy, store predictions, and generate confusion matrix.

### STEP 11: 

Visualize confusion matrix using heatmap and display classification report with precision metrics.


### STEP 12: 

Predict single image: load from dataset, infer, and display actual-predicted labels.



## PROGRAM

### Name:  DHARSHAN D

### Register Number:  212223230045

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 1: Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Define CNN
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 16 filters, 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 32 filters, 3x3
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
def train_model(model, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, num_epochs=5)

# Step 5: Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test Accuracy: {correct/total:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

test_model(model, test_loader)

# Step 6: Predict a single image
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f'Actual: {label} | Predicted: {predicted.item()}')
    plt.axis("off")
    plt.show()

predict_image(model, image_index=80, dataset=test_dataset)


```

### OUTPUT

## Training Loss per Epoch

<img width="268" height="130" alt="image" src="https://github.com/user-attachments/assets/7ee17b91-e318-4abc-8577-748505b9044a" />

## Confusion Matrix

<img width="742" height="584" alt="image" src="https://github.com/user-attachments/assets/985f85ed-1f3e-4e9b-8e51-e34d79fb07d2" />

## Classification Report
<img width="517" height="357" alt="image" src="https://github.com/user-attachments/assets/1a5cc786-46ad-4c92-916e-02582c85ed59" />

### New Sample Data Prediction
<img width="481" height="470" alt="image" src="https://github.com/user-attachments/assets/90ce9e58-1d88-4340-bc60-0b6f016450c8" />

## RESULT
Developing a convolutional neural network (CNN) classification model for the given dataset was executed successfully.
