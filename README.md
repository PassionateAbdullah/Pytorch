torch.utils.data.Dataset and torch.utils.data.DataLoader serve different but complementary purpose
1. torch.utils.data.Dataset :
When to Use:
  Custom Dataset: When you have a dataset not already available in libraries like torchvision.datasets or torchtext, you'll define your own dataset class by           subclassing torch.utils.data.Dataset.
  Preprocessing: When you need to define custom preprocessing or transformations applied to each data sample.
Example:
Creating a custom dataset:
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
# Example usage
import torch
data = torch.randn(100, 3, 28, 28)  # 100 samples, 3 channels, 28x28 images
labels = torch.randint(0, 10, (100,))  # 100 random labels (10 classes)
dataset = CustomDataset(data, labels)



2. torch.utils.data.DataLoader
This is used to load data from a Dataset object. It provides functionalities like:
   Batching: Dividing data into batches.
  Shuffling: Randomizing the order of data for each epoch.
  Parallelism: Using multiple workers to load data in parallel (via the num_workers parameter).
  Collation: Combining multiple samples into a batch (via collate_fn).
When to Use:
  Training/Inference: When you need to feed data to your model in batches during training or inference.
  Efficiency: When you want to utilize parallel data loading for efficiency.
Example:
Using a DataLoader with a dataset:
from torch.utils.data import DataLoader

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    X, y = batch
    print(f"Batch shape: {X.shape}, Labels shape: {y.shape}")

