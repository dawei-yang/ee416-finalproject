import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import cv2 as cv
import os
import pandas as pd

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class DataGenerator:
    def __init__(self, fold):
        dataFolders = ["Covid", "Healthy", "Others"]
        train_index = 0
        test_index = 0
        os.makedirs("./Dataset/Train/image/data", exist_ok=True)
        os.makedirs("./Dataset/Train/label", exist_ok=True)
        os.makedirs("./Dataset/Test/image/data", exist_ok=True)
        os.makedirs("./Dataset/Test/label", exist_ok=True)
   
        f_train = open("Dataset/Train/label/trainLabels.csv", "w")
        f_test = open("Dataset/Test/label/testLabels.csv", "w")   
        f_train.write("index\n")#0: Covid, 1: Healthy, 2: Others 
        f_test.write("index\n")
        for datasetName in dataFolders:
            for folderName in os.listdir(datasetName):
                if not folderName.startswith('.'):
                    number_of_files = len(os.listdir(os.path.join(datasetName, folderName)))
                    print("{} folder {} has files: {}".format(datasetName, folderName, number_of_files))
                    count = 0

                    for imageName in os.listdir(os.path.join(datasetName, folderName)):
                        if imageName.endswith(".png"):
                            if count >= number_of_files/5:
                                src = os.path.join(datasetName, folderName, imageName)
                                img = cv.imread(src)
                                img = cv.resize(img, (360, 360))
                                dist = os.path.join("Dataset/Train/image/data", str(train_index)+".png")
                                cv.imwrite(dist, img)
                                f_train.write(str(dataFolders.index(datasetName)) + "\n")
                                train_index += 1
                                img90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                                cv.imwrite(os.path.join("Dataset/Train/image/data", str(train_index) + ".png"), img90)
                                f_train.write(str(dataFolders.index(datasetName))  + "\n")
                                train_index += 1
                                img180 = cv.rotate(img, cv.ROTATE_180)
                                cv.imwrite(os.path.join("Dataset/Train/image/data", str(train_index) + ".png"), img180)
                                f_train.write(str(dataFolders.index(datasetName))  + "\n")
                                train_index += 1
                                img270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                                cv.imwrite(os.path.join("Dataset/Train/image/data", str(train_index) + ".png"), img270)
                                f_train.write(str(dataFolders.index(datasetName))  + "\n")
                                train_index += 1
                                count += 1
                            else:
                                src = os.path.join(datasetName, folderName, imageName)
                                img = cv.imread(src)
                                img = cv.resize(img, (360, 360))
                                dist = os.path.join("Dataset/Test/image/data", str(test_index)+".png")
                                cv.imwrite(dist, img)
                                f_test.write(str(dataFolders.index(datasetName))  + "\n")
                                test_index += 1
                                img90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                                cv.imwrite(os.path.join("Dataset/Test/image/data", str(test_index) + ".png"), img90)
                                f_test.write(str(dataFolders.index(datasetName))  + "\n")
                                test_index += 1
                                img180 = cv.rotate(img, cv.ROTATE_180)
                                cv.imwrite(os.path.join("Dataset/Test/image/data", str(test_index) + ".png"), img180)
                                f_test.write(str(dataFolders.index(datasetName))  + "\n")
                                test_index += 1
                                img270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                                cv.imwrite(os.path.join("Dataset/Test/image/data", str(test_index) + ".png"), img270)
                                f_test.write(str(dataFolders.index(datasetName))  + "\n")
                                test_index += 1
                                count += 1

class DataSet:
    def __init__(self, root):                  
        print("root: {}".format(root))
        self.ROOT = root
        # self.images = read_images(root + "/image")
        # self.labels = read_labels(root + "/label")
        # print(os.listdir(root + "/image"))
        MyTransform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1), # Convert image to grayscale
            transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
            transforms.Normalize([0.485], [0.224]) # TODO: Normalize to zero mean and unit variance with appropriate parameters
        ])
        self.images = datasets.ImageFolder(root = root + "/image", transform = MyTransform)
        if root == "./Dataset/Train":
            self.labels = pd.read_csv(root + "/label/trainLabels.csv")
        else: 
            self.labels = pd.read_csv(root + "/label/testLabels.csv")
        
        print("images: {}".format(len(self.images)))
        print("labels: {}".format(len(self.labels)))
        """ plt.imshow(self.images[0][0].permute(1, 2, 0))
        plt.show() """

        """ self.ROOT = root
        self.images = datasets.ImageFolder(root = root + "/image", transform = transforms.ToTensor())
        d = DataLoader(self.images, batch_size=1, shuffle=False)
      
        print("self.image: {}".format(len(self.images)))
        print("size of images: " + str(len(d)))
        img = cv.imread('1.png', cv.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap = 'gray')
        self.labels = datasets.ImageNet(root + "/label") """

    def __len__(self):
        # Return number of points in the dataset

        return len(self.images)

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        
        img = self.images[idx][0]
        label = torch.tensor(self.labels.to_numpy()[idx].item())

        return img, label

# Load the dataset and train and test splits
print("Loading datasets...")

# Data path
DataGenerator(0)
DATA_train_path = DataSet('./Dataset/Train')
DATA_test_path = DataSet('./Dataset/Test')

""" for i in range(53, 63):
    img, label = DATA_train_path[i]
    print("label of [{}] is {}".format(i, label)) """


# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(DATA_train_path, batch_size=8, shuffle=True)
testloader = DataLoader(DATA_test_path, batch_size=8, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: [Transfer learning with pre-trained ResNet-50] Design your own fully-connected network (FCN) classifier.
        # Design your own FCN classifier. Here I provide a sample of two-layer FCN.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, ReLU, Dropout, MaxPool2d, AvgPool2d
        # If you have many layers, consider using nn.Sequential() to simplify your code
        
        # Load pretrained ResNet-50
        self.model_resnet = models.resnet50(pretrained=True)
                
        # Set ResNet-50's FCN as an identity mapping
        num_fc_in = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        
        # TODO: Design your own FCN
        self.fc1 = nn.Linear(num_fc_in, 3, bias = 1) # from input of size num_fc_in to output of size ?
        self.fc2 = nn.Linear(3, 3, bias = 1) # from hidden layer to 3 class scores

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        
        relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        
        with torch.no_grad():
            features = self.model_resnet(x)
            
        x = self.fc1(features) # Activation are flattened before being passed to the fully connected layers
        x = relu(x)
        x = self.fc2(x)
        
        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0001) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength (default: lr=1e-2, weight_decay=1e-4)
num_epochs = 10 # TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = num_epochs): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            """ print("label{}".format(label))
            print("predict {}".format(np.argmax(pred))) """
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("length: {}, correct: {}".format(len(loader.dataset), correct))
    print("Evaluation accuracy: {}".format(acc))
    return acc


train(model, trainloader, num_epochs)
print("Evaluate on test set")
evaluate(model, testloader)

inputs, classes = next(iter(trainloader))
out = utils.make_grid(inputs)
# plt.imshow(out)
