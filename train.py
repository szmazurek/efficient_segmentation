import os
import torch
from network import UNet
from dataset_train import FetalBrainDataset, preprocess
from utils import DiceLoss

from mobilenetsmall import MobileNetV3Seg

from micronet import MicroNet

from efficientnet import get_efficientnet_seg

def train(args):
    # Set up dataset and data loaders
    images_folder = os.path.join(args.training_data_path, "images")
    masks_folder = os.path.join(args.training_data_path, "masks")
    train_dataset = FetalBrainDataset(images_folder, masks_folder, img_size=224, transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   pin_memory=True,
                                                   drop_last=True)

    # Set up device
    device = torch.device(args.device)

    # Set up model, optimizer, and criterion
    #model = MobileNetV3Seg().to(device)


    #model = MobileNetV3Seg(nclass=args.num_classes, width_mult=1.0).to(device)

    #model = MicroNet(nb_classes=2).to(device)

    model = get_efficientnet_seg()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = DiceLoss(n_classes=args.num_classes)

    # Train for multiple epochs
    print("Initializing training...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0  # Initialize running_loss here
        model.train()
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Apply softmax to each tensor in the tuple
            outputs_softmax = tuple(torch.softmax(output, dim=1) for output in outputs)
            
            # Calculate the loss for each output tensor
            losses = [criterion(output, targets) for output in outputs_softmax]
            loss = sum(losses)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Completed batch {i+1} of epoch {epoch+1}")
        
        print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_dataloader)}")




        # Save model checkpoint
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pt")

class Args:
    def __init__(self):
        self.num_classes = 2
        self.device = "cuda"  # or "cpu"
        self.epochs = 10
        self.batch_size = 4
        self.lr = 0.001
        self.training_data_path = "/net/tscratch/people/plgmnkpltrz/energyeff/efficient_segmentation/data/training_data"

args = Args()
args.mode = 'small'  # Set the mode

train(args)
