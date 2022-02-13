# Convolutional Neural Network for CIFAR dataset
from code.base_class.method import method
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb

# Define default hyperparameters
hyperperameter_defaults = {
  "learning_rate": 0.002774128312253735,
  "dropout": 0.5,
  "conv_channels": [128, 256, 512],
  "batch_size": 1000,
  "epochs": 15
}

wandb.init(config=hyperperameter_defaults, project="cifar_classification", entity="noahr")
config = wandb.config

if config["conv_channels"][0] == 32:
    FC_1_SIZE = 2048
elif config["conv_channels"][0] == 64:
    FC_1_SIZE = 4096
elif config["conv_channels"][0] == 128:
    FC_1_SIZE = 8192

from code.stage_2_code.Evaluate_Classifier import Evaluate_Classifier

class Method_CNN_CIFAR(method, nn.Module):
    data = None

    # Training Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, config["conv_channels"][0], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv1_activation = nn.ReLU()
        # Batch normalization
        self.conv1_batchnorm = nn.BatchNorm2d(config["conv_channels"][0])

        # Second convolutional layer
        self.conv2 = nn.Conv2d(config["conv_channels"][0], config["conv_channels"][0], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv2_activation = nn.ReLU()
        # Batch normalization
        self.conv2_batchnorm = nn.BatchNorm2d(config["conv_channels"][0])

        # Max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Dropout layer
        self.dropout1 = nn.Dropout(p=0.4)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(config["conv_channels"][0], config["conv_channels"][1], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv3_activation = nn.ReLU()
        # Batch normalization
        self.conv3_batchnorm = nn.BatchNorm2d(config["conv_channels"][1])

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(config["conv_channels"][1], config["conv_channels"][1], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv4_activation = nn.ReLU()
        # Batch normalization
        self.conv4_batchnorm = nn.BatchNorm2d(config["conv_channels"][1])

        # Max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Dropout layer
        self.dropout2 = nn.Dropout(p=0.4)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(config["conv_channels"][1], config["conv_channels"][2], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv5_activation = nn.ReLU()
        # Batch normalization
        self.conv5_batchnorm = nn.BatchNorm2d(config["conv_channels"][2])

        # Sixth convolutional layer
        self.conv6 = nn.Conv2d(config["conv_channels"][2], config["conv_channels"][2], kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.conv6_activation = nn.ReLU()
        # Batch normalization
        self.conv6_batchnorm = nn.BatchNorm2d(config["conv_channels"][2])

        # Max pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout3 = nn.Dropout(p=0.4)

        # Flatten layer
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(FC_1_SIZE, 1024)
        # ReLU activation function
        self.fc1_activation = nn.ReLU()

        # Batch normalization
        self.fc1_batchnorm = nn.BatchNorm1d(1024)

        # Second fully connected layer
        self.fc2 = nn.Linear(1024, 10)
        # Softmax activation function
        self.fc2_activation = nn.Softmax(dim=1)

        # Put the model on the GPU
        self.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        ''' Forward pass '''
        # First convolutional layer
        x = self.conv1(x)
        x = self.conv1_activation(x)
        x = self.conv1_batchnorm(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.conv2_activation(x)
        x = self.conv2_batchnorm(x)
        # Max pooling layer
        x = self.pool1(x)
        # Dropout layer
        x = self.dropout1(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.conv3_activation(x)
        x = self.conv3_batchnorm(x)
        # Fourth convolutional layer
        x = self.conv4(x)
        x = self.conv4_activation(x)
        x = self.conv4_batchnorm(x)
        # Max pooling layer
        x = self.pool2(x)
        # Dropout layer
        x = self.dropout2(x)
        # Fifth convolutional layer
        x = self.conv5(x)
        x = self.conv5_activation(x)
        x = self.conv5_batchnorm(x)
        # Sixth convolutional layer
        x = self.conv6(x)
        x = self.conv6_activation(x)
        x = self.conv6_batchnorm(x)
        # Max pooling layer
        x = self.pool3(x)
        # Dropout layer
        x = self.dropout3(x)
        # Flatten layer
        x = self.flatten(x)
        # First fully connected layer
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc1_batchnorm(x)
        # Fully connected output layer
        x = self.fc2(x)
        x = self.fc2_activation(x)
        return x

    def train(self, X, y):
        '''Train the model'''

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config["learning_rate"])
        evaluator = Evaluate_Classifier('training evaluator', '')
        # Create a tensor for the labels
        y_true = torch.LongTensor(y)

        # Watch the training
        wandb.watch(self)

        # Train the model
        for epoch in tqdm(range(wandb.config["epochs"])):
            # Predictions on the full training set
            y_pred = []
            total_loss = 0
            num_batches = 0
            # Mini-batch training
            for i in range(0, len(X), wandb.config["batch_size"]):
                # Get the mini-batch
                X_batch = torch.FloatTensor(X[i:i + wandb.config["batch_size"]]).to(self.device)
                y_true_batch = torch.LongTensor(y[i:i + wandb.config["batch_size"]]).to(self.device)
                # Forward pass
                y_pred_batch = self.forward(X_batch)
                # Save the predictions if this epoch will report test results
                if epoch % 5 == 0:
                    y_pred.append(y_pred_batch.max(1)[1].cpu().numpy())
                # Compute the loss
                loss = loss_function(y_pred_batch, y_true_batch)
                # Log the loss
                wandb.log({'batch_loss': loss}, step=epoch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Update the total loss
                total_loss += loss.item()
                # Update the number of batches
                num_batches += 1

            if epoch % 5 == 0:
                y_pred = torch.LongTensor(np.concatenate(y_pred))
                evaluator.data = {"true_y": y_true, "pred_y":y_pred}
                classification_report = evaluator.evaluate()
                print("*******")
                print('Epoch:', epoch)
                print(classification_report)
                # Print the average loss
                print('Average loss:', total_loss / num_batches)
                # Log the average loss
                wandb.log({'average_loss': total_loss / num_batches}, step=epoch)
                # Log the classification report
                wandb.log(evaluator.evaluate_to_dict())
            
            # Delete the tensors to free memory
            del X_batch, y_true_batch, y_pred_batch, y_pred
            # Clear the cache
            torch.cuda.empty_cache()

    def test(self, X):
        # do the testing, and result the result
        # Forward the data through the model in batches
        y_pred = []
        for i in range(0, len(X), wandb.config["batch_size"]):
            # Get the mini-batch
            X_batch = torch.FloatTensor(X[i:i + wandb.config["batch_size"]]).to(self.device)
            # Forward pass
            y_pred_batch = self.forward(X_batch)
            # Get the prediction
            y_pred.append(y_pred_batch.max(1)[1].cpu().numpy())
            # Delete the tensors to free memory
            del X_batch, y_pred_batch
            # Clear the cache
            torch.cuda.empty_cache()

        # Concatenate the prediction
        y_pred = np.concatenate(y_pred)

        return y_pred
    
    # Evaluate the model on a test set and log the results
    def evaluate(self, pred_y, true_y):
        evaluator = Evaluate_Classifier('test evaluator', '')
        evaluator.data = {"true_y": true_y, "pred_y": pred_y}
        classification_report = evaluator.evaluate()
        report = evaluator.evaluate_to_dict()
        # Log the classification report
        wandb.log({"evaluation", report})
        return classification_report
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}