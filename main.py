from torch.utils.data import DataLoader, random_split
import NpyDataset
import CRNN
from torch.optim.lr_scheduler import StepLR
import torch
from torch import optim

dataset = NpyDataset(root_dir='mfcc')

# Set split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate split lengths
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




model = CRNN(input_height=20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()  # for classification
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # or another optimizer
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)


torch.save(model, 'models/model_5.pth')