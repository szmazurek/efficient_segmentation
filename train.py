
def train_one_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, softmax=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Return dictionary with training information and updated model parameters
    return {"train_loss": running_loss / len(train_loader), "model": model}

