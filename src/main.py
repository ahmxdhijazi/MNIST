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

def main():
    # <--- GEMINI ASSISTANCE --->
    #Set up the argument parser to choose the model
    parser = argparse.ArgumentParser(description="Run MNIST classifiers.")
    parser.add_argument('--model',
                        type=str,
                        choices=['knn', 'nb', 'lc', 'mlp'],
                        required=True,
                        help="Model to run: 'knn' for K-Nearest Neighbor, 'nb' for Naive Bayes, 'lc' for Linear Classifier, mlp for MLP Classifier")
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
    #<--- GEMINI ASSISTANCE END --->

    #If KNN is chosen to run in CLI argument
    if args.model == 'knn':
        # Load the data
        X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

        #Just for reproducibility for different k values without altering my code.
        while True:
            try:
                k_input = input("Enter the value for k (e.g., 1, 3, 5): ")
                k_value = int(k_input)
                if k_value > 0:
                    break  # Exit the loop if input is a valid positive integer
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # Create an instance of the classifier
        model = KNNClassifier(k=k_value)

        #Train the model
        model.train(X_train, y_train)

        #Begin Timer on prediction since KNN instantly trains, we can just time the prediciton
        print("Starting prediction timer.")
        start_time = time.time()  #Record start time
        #Make our predictions
        predictions = model.predict(X_test)
        end_time = time.time()  #Record end time

        # (Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
        accuracy = np.sum(predictions == y_test) / len(y_test)


    #If Naive Bayes is chosen in CLI argument
    elif args.model == 'nb':
        # Load the data
        X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

        print("Running Naive Bayes Classifier.")

        #Binarize the data
        #Converting pixels to 0 or 1 on 0.5 threshold: (1 if pixel > 0.5, else 0).
        print("Binarizing data (pixel > 0.5).")
        X_train = (X_train > 0.5).astype(int) #True = 1 as int, False = 0 as int
        X_test = (X_test > 0.5).astype(int)

        #instance of classifier
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

        # (Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
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
                    break  # Exit the loop if input is a valid positive integer
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

        end_time = time.time()
        accuracy = correct / total

    elif args.model == 'mlp':
        print("Running Multilayer Perceptron Classifier.")
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

        # Reuse most of the lc code, however we need to se the MLP model instead obviously
        model = MLPClassifier().to(device)

        #another change is to use the cross entropy loss
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=0.01)

        print(f"Training for {num_epochs} epochs.")
        start_time = time.time()

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                #move data to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # no more one-hot encoding compared to lc, and pass labels directly
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad() #clear old
                loss.backward() #calculate new
                optimizer.step() #update model weights

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): # no gradients needed
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.time()
        accuracy = correct / total


    #Common steps for every model to calculate time and output results
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    #Output accuracy and train/testing time for corresponding methods
    if args.model == 'knn':
        print("--------------------")
        print(f"KNN Classifier Accuracy (k={model.k}): {accuracy * 100:.2f}%")
        print(f"Prediction took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'nb':
        print("--------------------")
        print(f"Naive Bayes Classifier Accuracy (alpha={model.alpha}): {accuracy * 100:.2f}%")
        print(f"Prediction took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'lc':
        print("--------------------")
        print(f"Linear Classifier Accuracy: {accuracy * 100:.2f}%")
        print(f"Training & Evaluation took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'mlp':
        print("--------------------")
        print(f"MLP Classifier Accuracy: {accuracy * 100:.2f}%")
        print(f"Training & Evaluation took {minutes} minutes and {seconds} seconds.")
    elif args.model == 'cnn':
        pass
    print("--------------------")



if __name__ == '__main__':
    main()
