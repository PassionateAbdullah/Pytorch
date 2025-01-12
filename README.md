torch.utils.data.Dataset and torch.utils.data.DataLoader serve different but complementary purpose
1. torch.utils.data.Dataset :
When to Use:
  Custom Dataset: When you have a dataset not already available in libraries like torchvision.datasets or torchtext, you'll define your own dataset class by           subclassing torch.utils.data.Dataset.
  Preprocessing: When you need to define custom preprocessing or transformations applied to each data sample.
