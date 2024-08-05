import os
import yt_dlp
import pandas as pd


def download_audio(url, output_folder):
    with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio',
                           'outtmpl': r'/home/nika/music-recommender-system/sample_music/%(title)s.wav'}) as video:
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


if __name__ == '__main__':

    output_folder = r"/home/nika/music-recommender-system/sample_music"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv('/home/nika/music-recommender-system/salami_youtube_pairings.csv')
    youtube_id_list = df['youtube_id'].tolist()
    youtube_url_list = ['https://www.youtube.com/watch?v=' + youtube_id for youtube_id in youtube_id_list]
    for youtube_url in youtube_url_list[:10]:
        try:
            download_audio(youtube_url, output_folder)
        except yt_dlp.utils.DownloadError as e:
            continue