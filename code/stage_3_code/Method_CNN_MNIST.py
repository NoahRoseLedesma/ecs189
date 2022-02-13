# Convolutional Neural Network for MNIST dataset
from sklearn.metrics import classification_report
from code.base_class.method import method
import torch
from torch import nn
import numpy as np
import wandb

from code.stage_2_code.Evaluate_Classifier import Evaluate_Classifier

# Default hyperparameters
hyperperameter_defaults = {
    'batch_size': 100,
    'epochs': 10,
    'learning_rate': 0.001,
    'dropout': 0.4,
}

wandb.init(config=hyperperameter_defaults, project="mnist_classification", entity="noahr")
config = wandb.config

class Method_CNN_MNIST(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = config["epochs"]
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = config["learning_rate"]
    # Training Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Batch size
    batch_size = config["batch_size"]

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # ReLU activation function
        self.conv1_activation = nn.ReLU()
        # Batch normalization
        self.conv1_batchnorm = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        # ReLU activation function
        self.conv2_activation = nn.ReLU()
        # Batch normalization
        self.conv2_batchnorm = nn.BatchNorm2d(32)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # ReLU activation function
        self.conv3_activation = nn.ReLU()
        # Batch normalization
        self.conv3_batchnorm = nn.BatchNorm2d(32)

        # First dropout layer
        self.dropout1 = nn.Dropout(config["dropout"])

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        # ReLU activation function
        self.conv4_activation = nn.ReLU()
        # Batch normalization
        self.conv4_batchnorm = nn.BatchNorm2d(64)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        # ReLU activation function
        self.conv5_activation = nn.ReLU()
        # Batch normalization
        self.conv5_batchnorm = nn.BatchNorm2d(64)

        # Sixth convolutional layer
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        # ReLU activation function
        self.conv6_activation = nn.ReLU()
        # Batch normalization
        self.conv6_batchnorm = nn.BatchNorm2d(64)

        # Second dropout layer
        self.dropout2 = nn.Dropout(config["dropout"])

        # Flatten layer
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(64, 128)
        # ReLU activation function
        self.fc1_activation = nn.ReLU()
        # Batch normalization
        self.fc1_batchnorm = nn.BatchNorm1d(128)
        # Third dropout layer
        self.dropout3 = nn.Dropout(config["dropout"])

        # Output fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # Softmax activation function
        self.fc2_activation = nn.Softmax(dim=1)

        # Put the model on the GPU
        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # First convolutional layer
        x = self.conv1(x)
        x = self.conv1_activation(x)
        x = self.conv1_batchnorm(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.conv2_activation(x)
        x = self.conv2_batchnorm(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.conv3_activation(x)
        x = self.conv3_batchnorm(x)
        # First dropout layer
        x = self.dropout1(x)
        # Fourth convolutional layer
        x = self.conv4(x)
        x = self.conv4_activation(x)
        x = self.conv4_batchnorm(x)
        # Fifth convolutional layer
        x = self.conv5(x)
        x = self.conv5_activation(x)
        x = self.conv5_batchnorm(x)
        # Sixth convolutional layer
        x = self.conv6(x)
        x = self.conv6_activation(x)
        x = self.conv6_batchnorm(x)
        # Second dropout layer
        x = self.dropout2(x)
        # Flatten layer
        x = self.flatten(x)
        # First fully connected layer
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc1_batchnorm(x)
        # Third dropout layer
        x = self.dropout3(x)
        # Output fully connected layer
        x = self.fc2(x)
        x = self.fc2_activation(x)

        return x
    
    def train(self, X, y):
        '''Train the model'''
        # Add channel dimension to the input data
        X = np.expand_dims(X, 1)

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        evaluator = Evaluate_Classifier('training evaluator', '')

        # Train the model
        for epoch in range(self.max_epoch):
            # Mini-batch training
            for i in range(0, len(X), self.batch_size):
                # Get the mini-batch
                X_batch = torch.FloatTensor(X[i:i + self.batch_size]).to(self.device)
                y_true_batch = torch.LongTensor(y[i:i + self.batch_size]).to(self.device)
                # Forward pass
                y_pred_batch = self.forward(X_batch)
                # Compute the loss
                loss = loss_function(y_pred_batch, y_true_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 1 == 0:
                evaluator.data = {"true_y": y_true_batch.cpu(), "pred_y": y_pred_batch.max(1)[1].cpu()}
                classification_report = evaluator.evaluate()
                print("*******")
                print('Epoch:', epoch)
                print(classification_report)
                loss = loss.item()
                print('Loss:', loss)
                report = evaluator.evaluate_to_dict()
                wandb.log(report)
                wandb.log({'loss': loss})

    def test(self, X):
        # Add channel dimension to the input data
        X = np.expand_dims(X, 1)
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1].cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}