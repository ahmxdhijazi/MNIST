import numpy as np
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self, alpha=1):  #alpha is the smoothing factor
        self.alpha = alpha #chosen above ^
        self.priors = None
        self.likelihoods = None
        self.classes = None
        print(f"Initialized Naive Bayes Classifier with alpha={self.alpha}.")

    def train(self, X_train, y_train): # Trains the classifier by calculating prior probabilities and likelihoods
        print("Training Naive Bayes model.")
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)

        #calculating prior probabilities P(class), AKA the frequency of each class in the training data
        self.priors = np.zeros(n_classes, dtype=np.float64) #Floating point to ensure we store values between [0.0-0.1] with double precision
        for index, val in enumerate(self.classes):
            self.priors[index] = np.sum(y_train == val) / n_samples

        #calculating likelihoods
        self.likelihoods = np.zeros((n_classes, n_features), dtype=np.float64)

        for index, val in enumerate(self.classes):
            #boolean mask
            mask = (y_train == val)
            X_c = X_train[mask]

            #Calculate the numerator using Laplace Smoothing, then add the smoothing factor (alpha).
            numerator = np.sum(X_c, axis=0) + self.alpha
            #Calculate the denominator using Laplace Smoothing
            denominator = len(X_c) + 2 * self.alpha
            #Compute the final likelihood probability for each pixel for class 'c'
            self.likelihoods[index, :] = numerator / denominator

        print("Training complete.")

    def predict(self, X_test):
        print(f"Predicting labels for {len(X_test)} test samples...")
        predictions = []
        # Loop through each test image
        for test_image in X_test:
            # We will calculate a 'log score' for each of the 10 classes
            class_scores = []

            # Loop through each class (0, 1, 2, ...) to get its score
            for index, c in enumerate(self.classes):
                #To Begin, start with log of the prior probability for this class
                log_prev = np.log(self.priors[index])

                #get pre-calculated log probabilities for the current class 'c'
                log_probs_on = np.log(self.likelihoods[index])  #pixels being 'on' prob
                log_probs_off = np.log(1 - self.likelihoods[index])  #pixels being 'off' prob

                #Calculate the score contributed by all the 'on' and 'off' pixels in the test image, selecting only the relevant probabilities.
                score_from_on_pixels = np.sum(log_probs_on[test_image == 1])
                score_from_off_pixels = np.sum(log_probs_off[test_image == 0])

                #total log likelihood is the sum of these two scores.
                log_likelihood = score_from_on_pixels + score_from_off_pixels
                #total score = the sum of the two logs
                total_score = log_prev + log_likelihood
                class_scores.append(total_score)

            #prediction = class with the highest score
            #np.argmax() returns index of largest value within list
            prediction = np.argmax(class_scores)
            predictions.append(prediction)

        return np.array(predictions)

    def plot_prob_maps(self):
        # Create grid of subplots
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))

        for index, c in enumerate(self.classes):
            # Determine which subplot to draw on
            ax = axes[index // 5, index % 5]
            # Reshape the 784-element pro-vector back into a 28x28 image
            prob_map = self.likelihoods[index, :].reshape(28, 28)
            #display image with grayscale color map
            ax.imshow(prob_map, cmap='gray')
            ax.set_title(f"Probability Map for: {c}")
            ax.axis('off')  # Hide the x/y axes for a cleaner look, looks cluttered with it

        plt.suptitle("Naive Bayes Learned Probability Maps")
        plt.tight_layout()  # Adjusts spacing
        #plt.savefig("naive_bayes_maps.png")  #would save the plot to a file
        plt.show()

