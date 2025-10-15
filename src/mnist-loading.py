import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  #Fill this with (image_path, label) tuples

        print("Finding all image paths...")
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