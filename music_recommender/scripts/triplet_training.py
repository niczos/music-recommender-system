import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Example Usage
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.triplet_loss import TripletLoss, train_model
from music_recommender.src.trtiplet_dataset import TripletRecommendationDataset
from music_recommender.src.utils import get_config


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replace with your actual model
    model = ConvNextTinyEncoder(pretrained=False)

    # Replace with your actual DataLoader
    train_loader = DataLoader(TripletRecommendationDataset(annotations_file=config['annotations_file'],
                                                           music_dir=config['music_dir'],
                                                           music_parts=config['music_parts'],
                                                           transforms=transforms,
                                                           temp_dir=config['temp_dir'],
                                                           ), batch_size=config['batch_size'], shuffle=True)

    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)


if __name__ == "__main__":
    config = get_config()
    main(config=config)
