MNIST Digit Classifier

A PyTorch project using the MNIST dataset — a classic collection of handwritten digits commonly used to train and test image classification models. This project is part of my journey to learn PyTorch, master the fundamentals of classification, and strengthen my technical ability to learn and adapt to new tools and frameworks.

Project Structure

MNIST/
├── README.md
├── requirements.txt
├── data/
│   └── MNIST_Data/ # Contains the digit image folders (0/, 1/, ...)
└── src/
    ├── main.py     # Main script to run classifiers
    ├── data_loader.py # Data loading functions
    └── models/     # Directory containing model definitions
        ├── knn.py
        ├── naive_bayes.py
        ├── linear_classifier.py
        ├── mlp_classifier.py
        └── cnn.py


Setup

Clone the repository:

git clone https://github.com/ahmxdhijazi/MNIST.git
cd MNIST


Install dependencies: Ensure you have Python 3 installed. Then, install the required packages using pip:

pip install -r requirements.txt


Data: Place your MNIST dataset folder (containing subfolders 0 through 9 with .png images) inside the data/ directory. Ensure the data folder is named MNIST_Data.

Usage

To run a classifier, use the main.py script from the root directory of the project. You must specify which model to run using the --model argument.

python src/main.py --model <model_abbreviation>


Replace <model_abbreviation> with one of the following:

knn: K-Nearest Neighbors (NumPy)

nb: Naive Bayes (NumPy)

lc: Linear Classifier (PyTorch)

mlp: Multilayer Perceptron (PyTorch)

cnn: Convolutional Neural Network (PyTorch)

Examples:

To run K-Nearest Neighbors:

python src/main.py --model knn


(The script will prompt you to enter a value for 'k'.)

To run the Convolutional Neural Network:

python src/main.py --model cnn


(The script will prompt you to enter the number of epochs.)

The script will output the model's accuracy and the time taken for training/prediction. Visualizations for Naive Bayes and the Linear Classifier will be displayed automatically in pop-up windows.
