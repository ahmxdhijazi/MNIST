import numpy as np
from collections import Counter

class KNNClassifier:
    #Compare with k = 3 is not told otherwise, good practice incase I mess up...
    def __init__(self, k=3):
        self.k = k
        print(f"Initialized KNN Classifier with k={self.k}")

    def train(self, X_train, y_train):
        # For KNN, we just store the data.
        self.X_train = X_train
        self.y_train = y_train
        print("Training data successfully stored.")

    def predict(self, X_test):
        print(f"Predicting labels for {len(X_test)} test samples.")
        predictions = []
        #Loop through each image in X_test.
        for test_image in X_test:
            #Calculate the Euclidean distance (distance between two vectors) from test_image to EVERY image in self.X_train.
            distances =  np.sqrt(np.sum((self.X_train - test_image) ** 2, axis=1)) #axis ensures the sum of 784 features and not 48000 images
            #Get the indices of the 'k' smallest distances.
            sorted_indices = np.argsort(distances) #far more efficient than manually sorting
            k_nearest_indices = sorted_indices[:self.k]
            #Get the labels of those 'k' neighbors from self.y_train.
            k_nearest_labels = self.y_train[k_nearest_indices]
            #Find the most common label among the neighbors using Counter class for its efficiency
            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            #Append the prediction to the `predictions` list.
            predictions.append(prediction)
        return np.array(predictions)
