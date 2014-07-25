# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sys
import os
import glob

import numpy as np
import scipy
import scipy.io.wavfile

from utils import GENRE_DIR, CHART_DIR

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def write_fft(fft_features, fn):
    """
    Write the FFT features to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"

    np.save(data_fn, fft_features)
    print("Written "%data_fn)


def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)

    fft_features = abs(scipy.fft(X)[:1000])
    write_fft(fft_features, fn)


def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        assert(file_list), genre_dir
        for fn in file_list:
            fft_features = np.load(fn)

            X.append(fft_features[:2000])
            y.append(label)

    return np.array(X), np.array(y)


def plot_wav_fft(wav_filename, desc=None):
    plt.clf()
    plt.figure(num=None, figsize=(6, 4))
    sample_rate, X = scipy.io.wavfile.read(wav_filename)
    spectrum = np.fft.fft(X)
    freq = np.fft.fftfreq(len(X), 1.0 / sample_rate)

    plt.subplot(211)
    num_samples = 200.0
    plt.xlim(0, num_samples / sample_rate)
    plt.xlabel("time [s]")
    plt.title(desc or wav_filename)
    plt.plot(np.arange(num_samples) / sample_rate, X[:num_samples])
    plt.grid(True)

    plt.subplot(212)
    plt.xlim(0, 5000)
    plt.xlabel("frequency [Hz]")
    plt.xticks(np.arange(5) * 1000)
    if desc:
        desc = desc.strip()
        fft_desc = desc[0].lower() + desc[1:]
    else:
        fft_desc = wav_filename
    plt.title("FFT of %s" % fft_desc)
    plt.plot(freq, abs(spectrum), linewidth=5)
    plt.grid(True)

    plt.tight_layout()

    rel_filename = os.path.split(wav_filename)[1]
    plt.savefig("%s_wav_fft.png" % os.path.splitext(rel_filename)[0],
                bbox_inches='tight')

    plt.show()


def plot_wav_fft_demo():
    plot_wav_fft("sine_a.wav", "400Hz sine wave")
    plot_wav_fft("sine_b.wav", "3,000Hz sine wave")
    plot_wav_fft("sine_mix.wav", "Mixed sine wave")


def plot_specgram(ax, fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ax.specgram(X, Fs=sample_rate, xextent=(0, 30))


def plot_specgrams(base_dir=CHART_DIR):
    """
    Plot a bunch of spectrograms of wav files in different genres
    """
    plt.clf()
    genres = ["classical", "jazz", "country", "pop", "rock", "metal"]
    num_files = 3
    f, axes = plt.subplots(len(genres), num_files)

    for genre_idx, genre in enumerate(genres):
        for idx, fn in enumerate(glob.glob(os.path.join(GENRE_DIR, genre, "*.wav"))):
            if idx == num_files:
                break
            axis = axes[genre_idx, idx]
            axis.yaxis.set_major_formatter(EngFormatter())
            axis.set_title("%s song %i" % (genre, idx + 1))
            plot_specgram(axis, fn)

    specgram_file = os.path.join(base_dir, "Spectrogram_Genres.png")
    plt.savefig(specgram_file, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # for fn in glob.glob(os.path.join(sys.argv[1], "*.wav")):
    #    create_fft(fn)

    # plot_decomp()

    if len(sys.argv) > 1:
        plot_wav_fft(sys.argv[1], desc="some sample song")
    else:
        plot_wav_fft_demo()

    plot_specgrams()
