import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class AudioSample:
    def __init__(self, filepath: str, temp_dir: str, sample_rate: int | None = None):
        self.filepath = filepath
        if sample_rate is None:
            self.y, self.sample_rate = librosa.load(path=self.filepath)
        else:
            self.y, _ = librosa.load(path=self.filepath, sr=sample_rate)
            self.sample_rate = sample_rate
        self.temp_dir = temp_dir

    def to_spectrogram(self) -> np.ndarray:
        # Extract Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sample_rate)

        # Convert Decibels (Log Scale)
        raw_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(raw_spectrogram, cmap='viridis')
        plt.axis('off')
        # output_filename = os.path.splitext(os.path.basename(self.filepath))[0]
        # output_filepath = os.path.join(self.temp_dir, output_filename)
        # plt.savefig(output_filepath)
        # return cv2.imread(output_filepath + ".png")
        figure = plt.gcf()
        figure.set_size_inches(12, 1)
        figure.set_dpi(50)

        figure.canvas.draw()

        b = figure.axes[0].get_window_extent()
        img = np.array(figure.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1)]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    folder_path = r"/home/nika/music-recommender-system/sample_music/"
    temp_dir = r"/home/nika/music-recommender-system/sample_music/"
    #folder_path = r"C:\Users\skrzy\Music\sample_music"
    #temp_dir = r"C:\Users\skrzy\Music\sample_music"

    filename = "01 - The Golden Age [Beck： Sea Change].wav"
    sample_path = os.path.join(folder_path, filename)
    audio_1 = AudioSample(filepath=sample_path, temp_dir=temp_dir)
    spec_1 = audio_1.to_spectrogram()
    print(spec_1)
    # TODO to nie działa na colabie
    cv2.imshow("Title", spec_1)
    cv2.waitKey(0)

    # for filename in os.listdir(folder_path):
    #     sample_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(sample_path):
    #         audio_1 = AudioSample(filepath=sample_path)
    #         spec_1 = audio_1.to_spectrogram()
    # image = cv2.imread(r"C:\Users\skrzy\Music\sample_music\01 - The Golden Age [Beck： Sea Change].png", cv2.IMREAD_UNCHANGED)
    # print(image)
