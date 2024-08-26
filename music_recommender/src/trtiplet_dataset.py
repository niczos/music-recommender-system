import os
import random

import numpy as np
import pandas as pd
import torch

from music_recommender.src.audio_dataset import RecommendationDataset
from music_recommender.src.audio_processing import AudioSample
from music_recommender.src.image_utils import transforms
from music_recommender.src.utils import get_config


class TripletRecommendationDataset(RecommendationDataset):
    def __getitem__(self, idx: int):
        column_names = self.img_labels.columns
        row = {column_name: value for column_name, value in zip(column_names, self.img_labels.iloc[idx].values)}
        images = []
        audio_path = os.path.join(self.music_dir,
                                  f"{row[('filename', '')]}.wav".replace(":", "：").replace("\"", "＂").replace("/", "⧸"))
        audio = AudioSample(filepath=audio_path, temp_dir=self.temp_dir)
        type_column_names = [name[0] for name in column_names if
                             any(name[0].startswith(music_part) for music_part in self.music_parts)]
        existing_type_column_names = [name for name in type_column_names if not pd.isna(row[(name, "beginning_time")])]
        existing_types = list(set([name.split("_")[0] for name in existing_type_column_names]))
        selected_type: str = random.choice(existing_types)
        selected_music_parts = random.choices(
            [name for name in existing_type_column_names if name.startswith(selected_type)], k=2)

        other_music_id = random.choice(range(len(self)))
        while other_music_id == idx:
            other_music_id = random.choice(range(len(self)))

        # TODO paralellize
        for part in selected_music_parts:
            spectrogram = audio.to_spectrogram(beginning=row[(part, "beginning_time")], end=row[(part, "end_time")])
            transformed_image = self.transform(spectrogram)
            images.append(transformed_image)

        column_names = self.img_labels.columns
        row = {column_name: value for column_name, value in zip(column_names, self.img_labels.iloc[idx].values)}
        audio_path = os.path.join(self.music_dir,
                                  f"{row[('filename', '')]}.wav".replace(":", "：").replace("\"", "＂").replace("/", "⧸"))
        audio = AudioSample(filepath=audio_path, temp_dir=self.temp_dir)
        type_column_names = [name[0] for name in column_names if
                             any(name[0].startswith(music_part) for music_part in self.music_parts)]
        existing_type_column_names = [name for name in type_column_names if not pd.isna(row[(name, "beginning_time")])]
        selected_music_part: str = random.choice(existing_type_column_names)
        spectrogram = audio.to_spectrogram(beginning=row[(selected_music_part, "beginning_time")],
                                           end=row[(selected_music_part, "end_time")])
        transformed_image = self.transform(spectrogram)
        images.append(transformed_image)

        return torch.stack(images)

    def read_annotations(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotations_path)
        df = df[df["type"].isin(self.music_parts)]
        how_many = df.groupby("salami_id")["type"].value_counts().reset_index()
        has_duplicates = how_many[how_many["count"] > 1][["salami_id", "type"]]

        df = has_duplicates.merge(df, on=["salami_id", "type"], how="left")
        df["count"] = df.groupby(["salami_id", "type"]).cumcount()
        df["counted_type"] = df["type"] + "_" + df["count"].astype(str)

        df = df.pivot(index=['salami_id', 'filename'], columns='counted_type', values=['beginning_time', 'end_time'])
        df.columns = df.columns.reorder_levels(order=[1, 0])
        return df.reset_index()


if __name__ == '__main__':
    config = get_config()

    ds = TripletRecommendationDataset(annotations_file=config['annotations_file'],
                                      music_dir=config['music_dir'],
                                      music_parts=config['music_parts'],
                                      transforms=transforms,
                                      temp_dir=config['temp_dir'],
                                      )
    print(ds[4].shape)
