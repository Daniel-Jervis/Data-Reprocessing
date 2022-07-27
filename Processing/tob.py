# https://stackoverflow.com/q/56791652/5358968

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy import integrate
from matplotlib import pyplot as plt

THIS_DIR = Path(__file__).parent

filename = '../data/S4A05961_20180504_150405.wav'
output_dir = '../outputs/'
sensitivity = -240
gain = 35
Vpp = 2.5
nbits = 16

waterfall_spacing = 50

def main():

    fs, data_raw = sio.wavfile.read(THIS_DIR / filename)

    # data_v = data_raw * Vpp/(2**nbits)
    # data = data_v * 10.0 ** (-(sensitivity + gain) / 10)

    data = data_raw * Vpp/(2**nbits) * 10.0**(-(sensitivity + gain)/20)

    out_dir = THIS_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(data)) / fs

    fig, ax = plt.subplots()

    ax.plot(t, data)
    fig.savefig(out_dir / 'waveform.png')

    centreFrequency_Hz = np.array([
        39, 50, 63, 79, 99, 125, 157, 198, 250, 315, 397, 500, 630, 794, 1000,
        1260, 1588, 2000, 2520, 3176, 4000, 5040, 6352, 8000, 10080, 12704,
        16000
        ])

    filtered_list = tob_filters(data, fs, centreFrequency_Hz)

    sel_list = []
    fig, ax = plt.subplots()
    refilled = np.zeros_like(data)
    for i, d in enumerate(filtered_list):
        refilled += d
        ax.plot(t, d + waterfall_spacing * i)
        sel_list.append(get_sel(d, fs))

    fig.savefig(out_dir / '')

    fig, ax = plt.subplots()
    ax.plot(t, data, label='Data')
    ax.plot(t, refilled, label='Reconstructed data')
    ax.legend()
    fig.savefig(out_dir / 'data_vs_reconstructed.png')

    fig, ax = plt.subplots()
    ax.semilogx(centreFrequency_Hz, np.array(sel_list))
    ax.set_title(f'SEL = {log_sum(sel_list):.2f} dB re 1 uPa^2 s')
    fig.savefig(out_dir / 'TOB_SEL.png')

    print(f'SEL = {log_sum(sel_list):.2f} dB re 1 uPa^2 s')
    print(f'SEL_2 = {get_sel(data, fs)}')
    print(f'SEL_3 = {get_sel(data_raw, fs) - sensitivity - gain + 20*np.log10(Vpp/(2**nbits))}')

    abs_diff = abs(data - refilled)
    fig, ax = plt.subplots()
    #ax.plot(t, abs_diff, label='Data diff')
    ff, Pxx = signal.periodogram(data, fs)
    ff2, Pxx2 = signal.periodogram(refilled, fs)
    ax.loglog(ff, Pxx, label='Data')
    ax.plot(ff2, Pxx2, label='Refilled')
    ax.legend()
    fig.savefig(out_dir / 'Spectral_Diff.png')

    return None


def get_sel(data, fs):
    mean_square = ((data - data.mean())**2).mean()
    sel = 10*np.log10(mean_square * len(data)/fs)
    return sel


def log_sum(data):
    total = 0
    for i in data:
        total = total + 10**(i/10)
    log_sum = 10*np.log10(total)
    return log_sum

# def log_sum(data):
#     e = np.sum(10.0**(np.array(data)/10))
#     return 10.0*np.log10(e)


def tob_filters(data: np.array, fs: int, centerFrequency_Hz) -> list[np.array]:
    nyquistRate = fs/2.0

    G = 2
    factor = np.power(G, 1.0/6.0)

    lowerCutoffFrequency_Hz=centerFrequency_Hz/factor
    upperCutoffFrequency_Hz=centerFrequency_Hz*factor

    filtered_list = []
    for lower,upper in zip(lowerCutoffFrequency_Hz, upperCutoffFrequency_Hz):
        # Design filter
        sos = signal.butter(N=4, Wn=np.array(
            [lower, upper])/nyquistRate, btype='bandpass', analog=False, output='sos')

        # Compute frequency response of the filter.

        # Filter signal
        filt_data = signal.sosfiltfilt(sos, data)
        filtered_list.append(filt_data)
    return filtered_list

if __name__=='__main__':
    main()

