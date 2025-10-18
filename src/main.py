from data_loader import load_data_numpy
from models.knn import KNNClassifier
from models.naive_bayes import NaiveBayesClassifier
import argparse #Built in Python argument to handle CLI arguments with Gemini Assistance
import numpy as np
import time #Just a cool addition for my notes

def main():
    #Load the data
    X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

    # <--- GEMINI ASSISTANCE --->
    #Set up the argument parser to choose the model
    parser = argparse.ArgumentParser(description="Run MNIST classifiers.")
    parser.add_argument('--model',
                        type=str,
                        choices=['knn', 'nb'],
                        required=True,
                        help="Model to run: 'knn' for K-Nearest Neighbor, 'nb' for Naive Bayes")
    args = parser.parse_args()
    #<--- GEMINI ASSISTANCE END --->

    #If KNN is chosen to run in CLI argument
    if args.model == 'knn':
        #Create an instance of the classifier
        k_value = 1
        knn = KNNClassifier(k=k_value)

        #Train the model
        knn.train(X_train, y_train)

        #Begin Timer on prediction since KNN instantly trains, we can just time the prediciton
        print("Starting prediction timer.")
        start_time = time.time()  #Record start time
        #Make our predictions
        predictions = knn.predict(X_test)
        end_time = time.time()  #Record end time


    #If Naive Bayes is chosen in CLI argument
    elif args.model == 'nb':
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


    #requirements for both
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    #(Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)

    #Output finally accuracy for k-NN
    if args.model == 'knn':
        print("--------------------")
        print(f"KNN Classifier Accuracy with (k={knn.k}): {accuracy * 100:.2f}%")
        print("--------------------")
    elif args.model == 'nb':
        print("--------------------")
        print(f"Naive Bayes Classifier Accuracy (alpha={model.alpha}) {accuracy * 100:.2f}%")
        print("--------------------")
    #Time Output
    print(f"Prediction took {minutes} minutes and {seconds} seconds.")



if __name__ == '__main__':
    main()
