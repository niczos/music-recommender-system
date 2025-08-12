import os
import warnings

from torch.utils.data import DataLoader

from music_recommender.src.trtiplet_dataset import TripletRecommendationDataset

warnings.filterwarnings('ignore')

from music_recommender.src.dataloaders import get_dataloaders
from music_recommender.src.evaluate import evaluate
from music_recommender.src.image_utils import transforms
from music_recommender.src.model import ConvNextTinyEncoder
from music_recommender.src.utils import get_config


def main(config):
    # ["Chorus", "Verse"]
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


    model = ConvNextTinyEncoder(
        pretrained=os.path.join(config["models_path"], "model_weights.pth"))

    metrics = evaluate(model=model, data={"val_loader": val_loader})

    print(metrics)


if __name__ == "__main__":
    config = get_config()
    main(config)
