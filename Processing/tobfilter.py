import numpy as np
from scipy import signal

from aapy import tobtools


def tob_filters(data: np.array,
                fs: int,
                central_frequencies_Hz: np.array,
                filter_order: int = 4) -> list[np.array]:
    """Filter data into one-third octave bands

    Based on Stack Overflow answer at https://stackoverflow.com/q/56791652/5358968 ."""

    nyquist_rate = fs / 2.0

    filtered_list = []
    for central_frequency in central_frequencies_Hz:
        band_id = tobtools.which_tob(central_frequency)
        lower = tobtools.lower_bound(band_id)
        upper = tobtools.upper_bound(band_id)

        sos = signal.butter(N=filter_order,
                            Wn=np.array([lower, upper]) / nyquist_rate,
                            btype='bandpass',
                            analog=False,
                            output='sos')

        filtered_data = signal.sosfiltfilt(sos, data)
        filtered_list.append(filtered_data)
    return filtered_list
