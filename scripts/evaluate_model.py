import os

from src.dataloaders import get_dataloaders
from src.evaluate import evaluate
from src.image_utils import transforms
from src.model import ConvNextTinyEncoder
from src.utils import get_config

# TODO come back after training

if __name__ == "__main__":
    config = get_config()
    # ["Chorus", "Verse"]
    train_dataloader = get_dataloaders(annotations_file=config['annotations_file'],
                                       music_dir=config['output_folder'],
                                       music_parts=config['music_parts'],
                                       transforms=transforms,
                                       temp_dir=config['temp_dir'],
                                       batch_size=config['batch_size'])

    # model = ConvNextTinyEncoder(pretrained=True)
    model = ConvNextTinyEncoder(pretrained=os.path.join(config["models_path"], "model_weights.pth"))

    metrics = evaluate(model=model, data={train_dataloader: train_dataloader})
