import os
from typing import Any

import yt_dlp
import requests
import pandas as pd


def download_audio(url, output_folder: str):
    with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio',
                           'outtmpl': os.path.join(output_folder, '%(title)s.wav')}) as video:
        info_dict = video.extract_info(url, download=True)
        video_title = info_dict['title']
        print(video_title)
        # jeśli plik istnieje pomiń
        if any(fname.startswith(os.path.basename(url)) for fname in os.listdir(output_folder)):
            return
        try:
            video.download([url])
        except yt_dlp.utils.DownloadError as e:
            print(f"Nie udało się pobrać pliku WAV dla {url}: {e}")
            # TODO napraw
            return
    return video_title


def youtube_id_to_url(id: str) -> str:
    return 'https://www.youtube.com/watch?v=' + id


def get_part_info(salami_id: int) -> list[dict[str, Any]]:
    url = f"https://raw.githubusercontent.com/DDMAL/salami-data-public/master/annotations/{salami_id}/parsed/textfile1_functions.txt"
    txt_string = requests.get(url=url).text
    return parse_music_parts(txt_string)


def parse_music_parts(text: str) -> list[dict[str, Any]]:
    rows = text.split("\n")
    result = []
    for row in rows:
        sec, type_ = row.split("\t")
        result.append({"beginning_time": float(sec), "type": type_})
    return result


def get_metadata(salami_id: int, salami_length: float, youtube_id: str, filename:str) -> list[dict[str, float | str | int]]:
    info = {'salami_id': salami_id, 'salami_length': salami_length, 'youtube_id': youtube_id, 'filename':filename}
    part_info = get_part_info(salami_id=salami_id)
    rows = []
    for i, part in enumerate(part_info):
        info.update(part)
        info.update({"end_time": (part_info[i+1]["beginning_time"] if i + 1 != len(part_info) else salami_length)} )
        rows.append(info.copy())
    return rows


def run_data_download(output_folder: str, df: pd.DataFrame):
    metadata_filename = os.path.join(output_folder, "metadata.csv")
    metadata = []
    limit = 5
    for ind, row in df.iterrows():
        if ind > limit:
            break
        youtube_id = row['youtube_id']
        salami_id = row['salami_id']
        salami_length = row['salami_length']
        youtube_url = youtube_id_to_url(id=youtube_id)
        try:
            filename = download_audio(youtube_url, output_folder)
            sample_metadata = get_metadata(salami_id=salami_id,
                                           salami_length=salami_length,
                                           youtube_id=youtube_id,
                                           filename=filename)
            metadata.append(sample_metadata)
        except yt_dlp.utils.DownloadError as e:
            continue
    metadata = [x for xs in metadata for x in xs]
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(metadata_filename, index=False)


def run_parsing_test(df):
    for _, row in df.iterrows():
        youtube_id = row['youtube_id']
        salami_id = row['salami_id']
        salami_length = row['salami_length']
        sample_metadata = get_metadata(salami_id=salami_id,
                                       salami_length=salami_length,
                                       youtube_id=youtube_id,
                                       filename="Sample")
        print(sample_metadata)
        break


if __name__ == '__main__':
    output_folder = r"/home/nika/music-recommender-system/sample_music"
    # output_folder = r"C:\Users\skrzy\Music\sample_music"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(os.path.join(output_folder, 'salami_youtube_pairings.csv'))
    run_data_download(output_folder=output_folder, df=df)
    # run_parsing_test(df)
