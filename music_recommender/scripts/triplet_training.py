import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.triplet_loss import TripletLoss, train_model
from music_recommender.src.trtiplet_dataset import TripletRecommendationDataset
from music_recommender.src.utils import get_config, generate_experiment_name

import matplotlib.pyplot as plt


def plot_loss_history(training_loss_history, validation_loss_history, filepath='loss_history.png'):
    """
    Plots the training and validation loss histories.

    Args:
        training_loss_history (list): A list of training losses for each epoch.
        validation_loss_history (list): A list of validation losses for each epoch.
        filename (str, optional): The filename for the saved plot. Defaults to 'loss_history.png'.
    """

    epochs = range(1, len(training_loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss_history, label='Training Loss')
    plt.plot(epochs, validation_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(filepath)
    plt.show()


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = generate_experiment_name(prefix="triplet_ssrl_learning")
    results_dir = os.path.join(config["output_dir"], experiment_name)
    os.mkdir(results_dir)

    # Replace with your actual model
    model = ConvNextTinyEncoder(pretrained=False)

    # Replace with your actual DataLoader
    train_loader = DataLoader(
        TripletRecommendationDataset(
            annotations_file=config["annotations_file"],
            music_dir=config["music_dir"],
            music_parts=config["music_parts"],
            transforms=transforms,
            temp_dir=config["temp_dir"],
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # Replace with your actual DataLoader
    val_loader = DataLoader(
        TripletRecommendationDataset(
            annotations_file=config["val_annotations_file"],
            music_dir=config["music_dir"],
            music_parts=config["music_parts"],
            transforms=transforms,
            temp_dir=config["temp_dir"],
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    criterion = TripletLoss(margin=config["triplet_loss_margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    training_loss_history, validation_loss_history = train_model(model=model, train_loader=train_loader,
                                                                 criterion=criterion,
                                                                 optimizer=optimizer, num_epochs=config["num_epochs"],
                                                                 device=device,
                                                                 val_loader=val_loader)

    model.save(path=config["output_dir"])
    plot_loss_history(training_loss_history,
                      validation_loss_history,
                      filepath=os.path.join(config["output_dir"], 'loss_history.png'))


if __name__ == "__main__":
    config = get_config()
    main(config=config)
