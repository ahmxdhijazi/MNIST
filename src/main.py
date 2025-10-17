from data_loader import load_data_numpy
from models.knn import KNNClassifier
import numpy as np
import time #Just a cool addition for my notes

def main():
    #Load the data
    X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')

    #Create an instance of the classifier
    k_value = 1
    knn = KNNClassifier(k=k_value)

    #Train the model
    knn.train(X_train, y_train)

    #Begin Timer
    print("Starting prediction timer...")
    start_time = time.time()  #Record start time

    #Make our predictions
    predictions = knn.predict(X_test)

    end_time = time.time()  #Record end time
    #Calculate the duration
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    #(Number of Correct Predictions) / (Total Number of Predictions) to evaluate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)

    #Output finally accuracy for k-NN
    print("--------------------")
    print(f"KNN Classifier Accuracy with (k={knn.k}): {accuracy * 100:.2f}%")
    print(f"Prediction took {minutes} minutes and {seconds} seconds.")
    print("--------------------")


if __name__ == '__main__':
    main()
