import torch.utils


# TODO come back after training
def evaluate_on_data(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store the total loss and number of correct predictions
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # We don't need to compute gradients for evaluation, so we use torch.no_grad()
    with torch.no_grad():
        for data in dataloader:
            # Unpack the data (assuming the dataloader returns input and target)
            inputs, targets = data

            # Move data to the same device as the model
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            # Forward pass: compute the model output
            outputs = model(inputs)

            # Compute the loss (assuming model's criterion is CrossEntropyLoss)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            # Update the total loss
            total_loss += loss.item() * inputs.size(0)

            # Get the predicted classes (assuming classification task)
            _, predicted = torch.max(outputs, 1)

            # Update the total number of correct predictions
            total_correct += (predicted == targets).sum().item()

            # Update the total number of samples
            total_samples += targets.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model: torch.nn.Module, data: dict[str, torch.utils.data.DataLoader]):
    for dataset_name, dataloader in data.items():
        evaluate_on_data(model, dataloader=dataloader)
    return None
