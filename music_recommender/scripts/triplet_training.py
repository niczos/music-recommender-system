import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.triplet_loss import TripletLoss, train_model
from music_recommender.src.trtiplet_dataset import TripletRecommendationDataset
from music_recommender.src.utils import get_config, generate_experiment_name


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    experiment_name = generate_experiment_name(prefix="triplet_ssrl_learning")
    results_dir = os.path.join(config["output_dir"], experiment_name)
    os.mkdir(results_dir)


    # Replace with your actual model
    model = ConvNextTinyEncoder(pretrained=False)

    # Replace with your actual DataLoader
    train_loader = DataLoader(TripletRecommendationDataset(annotations_file=config['annotations_file'],
                                                           music_dir=config['music_dir'],
                                                           music_parts=config['music_parts'],
                                                           transforms=transforms,
                                                           temp_dir=config['temp_dir'],
                                                           ), batch_size=config['batch_size'], shuffle=True)

    criterion = TripletLoss(margin=config["triplet_loss_margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_model(model, train_loader, criterion, optimizer, config["num_epochs"], device)

    model.save(path=config["output_dir"])


if __name__ == "__main__":
    config = get_config()
    main(config=config)
