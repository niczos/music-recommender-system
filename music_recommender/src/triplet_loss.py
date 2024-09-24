import torch.nn as nn
import torch


# Define the Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative)
        target = torch.ones_like(distance_positive)
        loss = self.loss_fn(distance_positive, distance_negative, target)
        return loss


def validation_step(model, val_loader, criterion, device):
    """
    Performs a validation step on the model using the provided data loader.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        criterion (torch.nn.Module): The loss function to be used.
        device (torch.device): The device to use for validation (CPU or GPU).

    Returns:
        float: The average validation loss for the current epoch.
    """

    model.eval()  # Set model to evaluation mode

    epoch_loss = 0.0

    # Iterate over batches in the validation loader
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)  # Get model predictions

        # Depending on your model's output format, adjust loss calculation
        loss = criterion(outputs, targets)  # Calculate loss between predictions and true targets

        epoch_loss += loss.item()  # Accumulate loss for the epoch

    # Calculate average loss for the validation epoch
    avg_loss = epoch_loss / len(val_loader)

    return avg_loss


# Function to Perform a Single Training Step
def train_step(model, data, criterion, optimizer, device):
    anchor, positive, negative = data
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

    # Forward pass
    anchor_output = model(anchor)
    positive_output = model(positive)
    negative_output = model(negative)

    # Compute loss
    loss = criterion(anchor_output, positive_output, negative_output)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, val_loader=None):
    """
    Trains a model with validation (if provided).

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        criterion (torch.nn.Module): The loss function to be used.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to use for training (CPU or GPU).
        val_loader (torch.utils.data.DataLoader, optional): The data loader for validation data.

    Returns:
        tuple: A tuple containing the following elements:
            - training_loss_history (list): A list of average training losses for each epoch.
            - validation_loss_history (list): A list of validation losses for each epoch (if val_loader is provided).
    """

    model.to(device)
    model.train()

    training_loss_history = []
    validation_loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, data in enumerate(train_loader):
            loss = train_step(model, data, criterion, optimizer, device)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_loss_history.append(avg_epoch_loss)

        if val_loader is not None:
            validation_loss = validation_step(model, val_loader, criterion, device)
            validation_loss_history.append(validation_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.6f}')
        if val_loader is not None:
            print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {validation_loss:.6f}')

    return training_loss_history, validation_loss_history
