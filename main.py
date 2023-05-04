import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from network import UNet
from train import train_one_epoch
from test import test_model
from data_loader import SegmentationDataGenerator

from loss import DiceLoss

if __name__ == '__main__':

    # Set up dataset and data loaders
    train_dataset = SegmentationDataGenerator('./sample_data/train_data.csv')
    test_dataset_path = 'sample_data/test_data.csv'
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model, optimizer, and criterion
    model = UNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = DiceLoss(n_classes=2)

    # Train for multiple epochs
    for epoch in range(10):
        # Train for one epoch
        train_info = train_one_epoch(model, criterion, optimizer, train_dataloader, device)

        # Print training and test metrics
        print(f"Epoch {epoch}: train_loss = {train_info['train_loss']:.4f}")

        # Test on test data
        test_info = test_model(epoch, train_info['model'], test_dataset_path, device, save_flag=False)

        # Save model checkpoint
        # torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pt")
