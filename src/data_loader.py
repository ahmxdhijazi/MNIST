import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class CustomMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  #Fill this with (image_path, label) tuples

        print("Finding all image paths.")
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # fill this list

        # Get the subdirectories ('0', '1', etc.)
        for label in sorted(os.listdir(root_dir)):
            digit_path = os.path.join(root_dir, label)

            #ensure we are working with directory
            if not os.path.isdir(digit_path):
                continue

            # Get the individual image files from directory
            for image_file in os.listdir(digit_path):
                if image_file.endswith('.png'):
                    #full path to the image
                    full_path = os.path.join(digit_path, image_file)

                    # Store the (path, label) tuple. We convert the label to an integer.
                    item = (full_path, int(label))
                    self.samples.append(item)

        print(f"Found {len(self.samples)} images.")

    def __len__(self):
        #return the total number of images.
        return len(self.samples)

    def __getitem__(self, idx):
        # This is where we'll actually load and transform one image.
        img_path, label = self.samples[idx]

        # Open the image, and ensured its in grayscale format
        img = Image.open(img_path).convert('L')

        # Apply transformation
        if self.transform:
            img = self.transform(img)
        return img, label


#data pipeline to load dataset numpy only classifiers
def load_data_numpy(root_dir, test_split = .2, random_state = 42): #Test split be 20 percent of the data with a state/seed of 42 (always the same split data)
    imgs = []
    labels = []
    print("Loading data for models requiring numpy.")
    # Implement the nested loop to go through each digit folder and each image.
    for label in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, label) #Build the path to the digit folder

        if not os.path.isdir(folder_path): #Ensure a folder
            continue

        #Continue into the digit folder to access individual .png
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.png'):
                #Open the image with PIL and convert it to a NumPy array.
                full_image_path = os.path.join(folder_path, image_file)
                # Now, we open the correct path
                image = Image.open(full_image_path).convert('L')
                #conver to numpy array
                image_array = np.array(image)

                #normalize then flatten image array and append to imgs
                normalized_array = image_array / 255.0
                flattened_array = normalized_array.flatten()
                imgs.append(flattened_array)
                labels.append(int(label))


    print("Converting lists to NumPy arrays.")
    X = np.array(imgs)
    y = np.array(labels)

    print("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )

    print("Data loading complete.")
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # Normalize pixel values to [0, 1]
    # transforms.ToTensor() handles this automatically!
    transform_0_to_1 = transforms.Compose([
        transforms.ToTensor()
    ])

    # Option 2: Normalize pixel values to [-1, 1]
    # by chaining together ToTensor() with Normalize().
    # For a [0, 1] input, if we set mean=0.5 and std=0.5
    transform_neg1_to_1 = transforms.Compose([
        transforms.ToTensor(),
        # A pixel at 0 becomes (0 - 0.5) / 0.5 = -1
        # A pixel at 1 becomes (1 - 0.5) / 0.5 = 1
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # An instance of our dataset with the desired transform.
    mnist_dataset = CustomMNISTDataset(root_dir='data/MNIST_Data', transform=transform_neg1_to_1)

    # The DataLoader takes Dataset and prepares batches.
    # A batch size of 64 is common. Shuffling is crucial for good training.
    data_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)

    # Let's get one batch of data.
    images_batch, labels_batch = next(iter(data_loader))

    print("\n--- Testing the DataLoader ---")
    print(f"Batch of images shape: {images_batch.shape}")
    print(f"Batch of labels shape: {labels_batch.shape}")

    # Let's check the pixel value range of the first image in the batch.
    first_image = images_batch[0]
    print(f"Min pixel value: {first_image.min()}")
    print(f"Max pixel value: {first_image.max()}")
    print(f"Label of first image: {labels_batch[0]}")

    #NUMPY LOADER TEST: significantly slower than using pytorch
    print("\n--- Testing NumPy Data Loader ---")
    X_train, y_train, X_test, y_test = load_data_numpy('data/MNIST_Data')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Pixel value range: {X_train.min()} to {X_train.max()}")