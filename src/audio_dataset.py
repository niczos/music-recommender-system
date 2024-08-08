import os

# import torch
import pandas as pd
# from torch.utils.data import Dataset


class RecommendationDataset: #(Dataset):
    def __init__(self, annotations_file: str, music_dir: str, music_parts: list[str], transform=None):
        self.annotations_path = annotations_file
        self.music_parts = music_parts
        self.img_labels = self.read_annotations()
        self.music_dir = music_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        column_names = self.img_labels.columns
        row = {column_name: value for column_name, value in zip(column_names, self.img_labels.iloc[idx].values)}


        # img_path = os.path.join(self.img_dir, )
    #     image = read_image(img_path)
    #     if self.transform:
    #         image = self.transform(image)
    #     return image

    def read_annotations(self) -> pd.DataFrame:
        df = pd.read_csv(self.annotations_path)
        df = df[df["type"].isin(self.music_parts)]
        df = df.drop_duplicates(subset=['salami_id', "type"], keep="first")
        df = df.pivot(index='salami_id', columns='type', values=['beginning_time', 'end_time'])
        df = df.dropna()
        df.columns = df.columns.reorder_levels(order=[1, 0])
        return df


if __name__ == '__main__':
    #output_folder = r"C:\Users\skrzy\Music\sample_music"
    output_folder =  r"/home/nika/music-recommender-system/sample_music/"
    annotations_file = os.path.join(output_folder, 'metadata.csv')

    ds = RecommendationDataset(annotations_file=annotations_file, music_dir=output_folder, music_parts=["Chorus", "Verse"])
    # print(ds.img_labels[["Chorus"]].head())

    print(ds[0])