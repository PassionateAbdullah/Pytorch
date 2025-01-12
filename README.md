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


Model Evaluation and prediction
What Happens in pred = model(X)?
Forward Propagation:

The input data X (e.g., images or feature vectors) is passed through the model.
The model is defined earlier (e.g., NeuralNetwork) and contains layers like Flatten, Linear, and activation functions (ReLU).
The input tensor flows through these layers sequentially, and the model computes the output predictions for each sample in X.
pred:

pred is the raw output (logits) produced by the model. These logits are not yet probabilities but are scores indicating how likely each class is for each input sample.
For example:

If the batch size is 64 and there are 10 classes, pred will have the shape [64, 10].
Each row corresponds to a sample in the batch, and each column corresponds to a class.
Why Use model.eval() Before This?
The model is set to evaluation mode using model.eval() before the test loop.
Purpose:
Disables behaviors specific to training (e.g., dropout, batch normalization updates) to ensure consistent and accurate results during testing.


