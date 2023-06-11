import os
import torch
from network import UNet
from dataset_train import FetalBrainDataset, preprocess
from utils import DiceLoss

def train(args):
    # Set up dataset and data loaders
    images_folder = os.path.join(args.training_data_path, "images")
    masks_folder = os.path.join(args.training_data_path, "masks")
    train_dataset = FetalBrainDataset(images_folder, masks_folder, img_size=224, transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=True)

    # Set up device
    device = torch.device(args.device)

    # Set up model, optimizer, and criterion
    model = UNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = DiceLoss(n_classes=args.num_classes)

    # Train for multiple epochs
    print("Initializing training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, softmax=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Completed batch {i+1} of epoch {epoch+1}")
        # Print training and test metrics
        print(f"Epoch {epoch}: train_loss = {running_loss / i+1:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pt")
