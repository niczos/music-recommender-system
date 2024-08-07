import os
import yt_dlp
import requests
import pandas as pd


def download_audio(url, output_folder):
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
            return


def youtube_id_to_url(id: str) -> str:
    return 'https://www.youtube.com/watch?v=' + id


def get_part_info(salami_id: str) -> dict:
    # https://stackoverflow.com/questions/51239168/how-to-download-single-file-from-a-git-repository-using-python
    d = {}
    url = f"https://raw.githubusercontent.com/DDMAL/salami-data-public/master/annotations/{salami_id}/parsed/textfile1_functions.txt"
    txt_string = requests.get(url=url).text
    print(txt_string)
    return d

def get_metadata():
    info = {'salami_id': salami_id, 'salami_length': salami_length, 'youtube_id': row['youtube_id']}
    part_info = get_part_info(salami_id=salami_id)
    info.update(part_info)
    metadata.append(info)
    return metadata

if __name__ == '__main__':

    output_folder = r"/home/nika/music-recommender-system/sample_music"
    #output_folder = r"C:\Users\skrzy\Music\sample_music"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(os.path.join(output_folder, 'salami_youtube_pairings.csv'))
    metadata_filename = os.path.join(output_folder, "metadata.csv")
    metadata = []
    limit = 5
    for inx, row in df.iterrows():
        if inx > limit:
            break
        youtube_url = youtube_id_to_url(id=row['youtube_id'])
        salami_id = row['salami_id']
        salami_length = row['salami_length']
        try:
            download_audio(youtube_url, output_folder)
            metadata = get_metadata()
        except yt_dlp.utils.DownloadError as e:
            continue
    metadata_df = pd.DataFrame(metadata, columns=['salami_id', 'salami_length', 'youtube_id'])
    metadata_df.to_csv(metadata_filename, index=False)
