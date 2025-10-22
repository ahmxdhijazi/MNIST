from data_loader import load_data_numpy
from data_loader import get_pytorch_dataloaders
from models.knn import KNNClassifier
from models.naive_bayes import NaiveBayesClassifier
import argparse #Built in Python argument to handle CLI arguments with Gemini Assistance
import numpy as np
import time #Just a cool addition for my notes
from models.linear_classifier import LinearClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp_classifier import MLPClassifier
from models.cnn import CNNClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_lin_weights(model):
    #Visualizes the learned weights of the LinearClassifier's single layer.
    weights = model.linear_layer.weight.data.to('cpu')
    #Create a figure and a grid of subplots (2 rows, 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    #Loop through each of the 10 digits (0 to 9)
    for i in range(10):
        #Reshape the weight vector for the current digit into a 28x28 image
        weight_image = weights[i].reshape(28, 28)
        #Select the correct subplot axes based on the current digit index
        ax = axes[i // 5, i % 5]
        #Display the weight image using a Red-Blue colormap (negative=red, positive=blue)
        ax.imshow(weight_image, cmap='RdBu')
        ax.set_title(f"Weight Template for: {i}")
        ax.axis('off')
    plt.suptitle("Linear Classifier Learned Weights (W)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, model_name):
#Generates and displays a confusion matrix for model's predictions.

    #Use scikit-learn to compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    #Create new figure
    plt.figure(figsize=(10, 8))

    #Using seaborn's heatmap for visuals
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))

    #Add labels and a title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name.upper()}')

    #Display the plot in a pop-up window
    plt.show()

def main():
    # <--- GEMINI CODE --->
    #Set up the argument parser to choose the model
    parser = argparse.ArgumentParser(description="Run MNIST classifiers.")
    parser.add_argument('--model',
                        type=str,
                        choices=['knn', 'nb', 'lc', 'mlp', 'cnn'],
                        required=True,
                        help="Model to run: 'knn' for K-Nearest Neighbor, 'nb' for Naive Bayes, 'lc' for Linear Classifier, mlp for MLP Classifier, cnn for CNN")
    args = parser.parse_args()

    # --- Set device (important for PyTorch) ---
    # This will use your M1 Max GPU ("mps") if available!
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    #<--- GEMINI CODE END --->

    #If KNN is chosen to run in CLI argument
    if args.model == 'knn':
        #Load the data
        X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

        #Just for reproducibility for different k values without altering my code.
        while True:
            try:
                k_input = input("Enter the value for k (e.g., 1, 3, 5): ")
                k_value = int(k_input)
                if k_value > 0:
                    break  #Exit the loop if input is a valid positive integer
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        #Create an instance of the classifier
        model = KNNClassifier(k=k_value)

        #Train the model
        model.train(X_train, y_train)

        #Begin Timer on prediction since KNN instantly trains, we can just time the prediciton
        print("Starting prediction timer.")
        start_time = time.time()  #Record start time
        #Make our predictions
        predictions = model.predict(X_test)
        end_time = time.time()  #Record end time

        #Used to output confusion matrix/failure analysis
        true_labels = y_test
        predicted_labels = predictions

        #(Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
        accuracy = np.sum(predictions == y_test) / len(y_test)

    #If NAIVE BAYES is chosen
    elif args.model == 'nb':
        #Load the data
        X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

        print("Running Naive Bayes Classifier.")

        #Binarize the data
        #Converting pixels to 0 or 1 on 0.5 threshold: (1 if pixel > 0.5, else 0).
        print("Binarizing data (pixel > 0.5).")
        X_train = (X_train > 0.5).astype(int) #True = 1 as int, False = 0 as int
        X_test = (X_test > 0.5).astype(int)

        #Instance of classifier
        model = NaiveBayesClassifier(alpha=1)  # Using alpha=1 for smoothing

        #Train for training
        print("Starting NB training timer.")
        start_time_train = time.time()
        model.train(X_train, y_train)
        end_time_train = time.time() #End Time
        train_duration = end_time_train - start_time_train
        print(f"Training took {train_duration:.4f} seconds.")

        #Timer for prediction, same variable names so outside calculation works for either
        print("Starting NB prediction timer.")
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()

        #variable for confusion matrix
        true_labels = y_test
        predicted_labels = predictions

        #(Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
        accuracy = np.sum(predictions == y_test) / len(y_test)

    #Linear Classifier (PYTORCH)
    elif args.model == 'lc':

        print("Running Linear Classifier (PyTorch).")

        #Load the PyTorch DataLoaders using the dataloader method
        train_loader, test_loader = get_pytorch_dataloaders(root_dir='data/MNIST_Data', batch_size=64)

        #Initialize Model, Loss, and Optimizer
        model = LinearClassifier().to(device)  # Move model to my M1GPU
        criterion = nn.MSELoss()  # L2 Loss, can experiment with this after successful attempt
        optimizer = optim.SGD(model.parameters(), lr=0.01)  #SGD

        #For reproducibility for different epoch values with CLI.
        while True:
            try:
                epoch_input = input("Enter the number of epochs to train for (e.g., 5, 10): ")
                num_epochs = int(epoch_input)
                if num_epochs > 0:
                    break  #Exit the loop if input is a valid positive integer
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        print(f"Training for {num_epochs} epochs.")

        start_time = time.time()  #Time the whole training & evaluation

        #Training loop
        model.train()  #Set the model to training mode
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                #Move data to the device
                images = images.to(device)
                labels = labels.to(device)

                #One-Hot Encode Labels, L2 Loss needs labels to be vectors of 10, not single numbers [0-9]
                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()

                #Forward pass to get the model's predictions
                outputs = model(images)

                #Calculate the loss
                loss = criterion(outputs, labels_one_hot)

                #Backward pass to calculate the gradients
                optimizer.zero_grad()  #clear old
                loss.backward()  # calculate new

                #Update model's weights
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        print("Training finished.")

        #Evaluation/Accuracy check
        print("Evaluating model.")
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        # Create empty lists to store all labels and predictions
        true_labels = []
        predicted_labels = []
        with torch.no_grad():  #no gradients needed
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                #Aquire model's predictions
                outputs = model(images)

                #Get prediction by finding the index of the max score
                #torch.max will return the (values, indices)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        end_time = time.time()
        accuracy = correct / total

    #IF MULTI-LAYER PERCEPTRON IS CHOSEN
    elif args.model == 'mlp':
        print("Running Multilayer Perceptron Classifier.")
        #Aquire data
        train_loader, test_loader = get_pytorch_dataloaders(root_dir='data/MNIST_Data', batch_size=64)

        #Get number of epochs from user same way I did for lc
        while True:
            try:
                epoch_input = input("Enter the number of epochs to train for (e.g., 5, 10): ")
                num_epochs = int(epoch_input)
                if num_epochs > 0:
                    break
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        #Reuse most of the lc code, however we need to se the MLP model instead obviously
        model = MLPClassifier().to(device)

        #Another change is to use the cross entropy loss
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=0.01)


        print(f"Training for {num_epochs} epochs.")
        start_time = time.time() #Begin time

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                #Move data to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # no more one-hot encoding compared to lc, and pass labels directly
                loss = criterion(outputs, labels)

                #Backward pass
                optimizer.zero_grad() #Clear old
                loss.backward() #Calculate new
                optimizer.step() #Update model weights

            #Print the average loss for the whole epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        #Evaluation
        model.eval() #Entering eval mode turn off special behaviors only used during training
        correct = 0
        total = 0

        true_labels = []
        predicted_labels = []

        with torch.no_grad(): #Saves a massive amount of memory and computation, making evaluation much faster
            for images, labels in test_loader: #Loop through all batches
                #Move batch data to correct device (CPU/GPU)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images) #Forward pass
                _, predicted = torch.max(outputs.data, 1) #Find index of highest score

                #Update counter and total
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        #End Testing time and calculate accuracy
        end_time = time.time()
        accuracy = correct / total

    #IF CNN IS CHOSEN
    elif args.model == 'cnn':
        print("Running Convolutional Neural Network (PyTorch).")
        #Aquire the data
        train_loader, test_loader = get_pytorch_dataloaders(root_dir='data/MNIST_Data', batch_size=64)

        #Get number of epochs from user
        while True:
            try:
                epoch_input = input("Enter the number of epochs to train for (e.g., 5, 10): ")
                num_epochs = int(epoch_input)
                if num_epochs > 0:
                    break
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        #Use the CNN model
        model = CNNClassifier().to(device)

        #Using Cross Entropy Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        #Begin training time
        print(f"Training for {num_epochs} epochs.")
        start_time = time.time()

        #Identical to MLP
        model.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #Print the average loss for each epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training finished.")
        print("Evaluating model.")

        model.eval() #Entering eval mode turn off special behaviors only used during training
        #Counters for total and correct predictions made
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad(): #Saves a massive amount of memory and computation, making evaluation much faster
            for images, labels in test_loader: #Loop through all batches
                #Move batch data to correct device (CPU/GPU)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images) #Forward pass
                _, predicted = torch.max(outputs.data, 1) #Find index of highest score

                #Update counter and total
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        end_time = time.time()
        accuracy = correct / total


    #Calculate time and output for every model for fun
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    #Output accuracy and train/testing time for corresponding methods
    print("--------------------")
    if args.model == 'knn':
        print(f"KNN Classifier Accuracy (k={model.k}): {accuracy * 100:.2f}%")
        print(f"Prediction took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'nb':
        print(f"Naive Bayes Classifier Accuracy (alpha={model.alpha}): {accuracy * 100:.2f}%")
        print(f"Prediction took {minutes} minutes and {seconds} seconds.")
        # Plotting within the method cause the accuracy and times to output with a delay while the plot loaded,
        # This will allow the output to complete before the plot arrives.
        model.plot_prob_maps()
    elif args.model == 'lc':
        print(f"Linear Classifier Accuracy: {accuracy * 100:.2f}%")
        print(f"Training & Evaluation took {minutes} minutes and {seconds} seconds.")
        plot_lin_weights(model)
    elif args.model == 'mlp':
        print(f"MLP Classifier Accuracy: {accuracy * 100:.2f}%")
        print(f"Training & Evaluation took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'cnn':
        print(f"CNN Accuracy: {accuracy * 100:.2f}%")
        print(f"Training & Evaluation took {minutes} minutes and {seconds} seconds.")
    print("--------------------")
    # Automatically generate confusion matrix after results are printed
    if args.model in ['knn', 'nb', 'lc', 'mlp', 'cnn']:
        print("Displaying confusion matrix...")
        plot_confusion_matrix(true_labels, predicted_labels, args.model)


if __name__ == '__main__':
    main()
