import os

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from src.dataloaders import get_dataloaders
from src.image_utils import transforms


class ConvNextTinyEncoder(nn.Module):
    def __init__(self, pretrained: bool | str = True):
        super(ConvNextTinyEncoder, self).__init__()
        self.convnext_tiny = convnext_tiny(pretrained=pretrained if pretrained is True else None)
        self.convnext_tiny.classifier = nn.Identity()
        if os.path.exists(pretrained):
            self.convnext_tiny.load_state_dict(torch.load(pretrained))
            print(f"Loaded model weights from {pretrained}")
        else:
            raise ValueError("Pretrained model not found.")

    def forward(self, images):
        # images: A  tensor of images with shape (N, V, C, H, W)
        if images.ndimension() == 4:  # If single image, add batch dimension
            images = images.unsqueeze(0)

        embeddings = []
        for view_nr in range(images.shape[1]):
            embedding = self.convnext_tiny(images[:, view_nr])  # Get the embedding from ConvNextTiny
            embeddings.append(embedding)

        # Concatenate embeddings along the feature dimension
        concatenated_embedding = torch.cat(embeddings, dim=1)
        return concatenated_embedding.squeeze(dim=(2, 3))

    def save(self, path: str):
        model_path = os.path.join(path, 'model_weights.pth')
        print(f"Saving model to {model_path}")
        torch.save(self.convnext_tiny.state_dict(), model_path)


    def get_embeddings(self, dataloader: torch.utils.data.DataLoader):
        with torch.no_grad():
            embeddings = []
            for batch in dataloader:
                batch_of_embeddings = self.forward(batch).detach()
                embeddings.append(batch_of_embeddings)
        return torch.concat(embeddings)


if __name__ == "__main__":

    output_folder = r"C:\Users\skrzy\Music\sample_music"
    annotations_file = os.path.join(output_folder, 'metadata.csv')
    temp_dir = output_folder
    train_dataloader = get_dataloaders(annotations_file=annotations_file,
                                       music_dir=output_folder,
                                       music_parts=["Chorus", "Verse"],
                                       transforms=transforms,
                                       temp_dir=temp_dir,
                                       batch_size=2)

    # model = ConvNextTinyEncoder(pretrained=True)
    model = ConvNextTinyEncoder(pretrained=r"C:\Users\skrzy\Music\sample_music\model_weights.pth")

    for batch in train_dataloader:
        output = model(batch)
        print(output.shape)  # The shape will depend on the number of images and feature size

    model.save(r"C:\Users\skrzy\Music\sample_music")
