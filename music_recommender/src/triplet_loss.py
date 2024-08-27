import torch
import torch.nn as nn


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


# Function to Run the Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            loss = train_step(model, data, criterion, optimizer, device)
            epoch_loss += loss

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss:.6f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}')
