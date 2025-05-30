"""
Contains the Spectrogram class as well as spectrogram related functions. This includes spectrogram
plotting and filtering algorithms
"""
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
from scipy.signal import spectrogram, get_window
from scipy.signal import istft, stft, windows


class Spectrogram:
    def __init__(self, data: np.ndarray, data_fs: float, spectrum_interval: float):
        """
        A Spectrogram. Fourier co-efficients over time

        :param data: 2D data (time x fourier co-eff)
        :param data_fs: sampling rate of the original data
        :param spectrum_interval: time interval between each spectrum (in Seconds)
        """
        self.data = data
        self.data_fs = data_fs
        self.fft_n = data.shape[1]
        self.num_data_point = data.shape[0]
        self.spectrum_interval = spectrum_interval

    def freq_to_index(self, frequency: np.ndarray):
        """
        Convert the given frequency to the lowest column index for this spectrogram

        :param frequency: in Hz
        :return: Column index number
        """
        return (2 * np.asarray(frequency) * self.fft_n / self.data_fs).astype(int)

    def index_to_freq(self, index: np.ndarray):
        """
        Convert the given index to its related frequency

        :param index: index
        :return: Frequency in Hz
        """
        return np.asarray(index) * self.data_fs / 2 / self.fft_n

    def plot(self, frequency_range: Tuple[float, float]) -> AxesImage:
        """
        Plot the given spectrogram. Make sure the format is in (time x fourier coe-effs). 
        This does not create a new plot

        :param frequency_range: Range of frequency to keep in the plot (in Hz)
        :return: matplotlib axis image
        """

        lower_index = self.freq_to_index(frequency_range[0])
        upper_index = self.freq_to_index(frequency_range[1]) + 1
        axes_image = plt.imshow(self.data[:, lower_index: upper_index],
                                extent=[frequency_range[0], frequency_range[1],
                                        self.num_data_point * self.spectrum_interval, 0],
                                interpolation=None,
                                aspect="auto")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel("Time (Seconds)")
        return axes_image


def generate_spectrogram(signal: np.ndarray, fs: float, window_length: float = 60, overlap_percent: float = 3 / 4,
                         window_type: str = 'hann') -> Spectrogram:
    """
    Generate a spectrogram from the given 1D Signal (in dB)

    :param signal: 1D data
    :param fs: Sampling Frequency (Hz)
    :param window_length: Length(in seconds) of each spectrogram window
    :param overlap_percent: 0 < p < 1, percentage overlap in spectrogram
    :param window_type: Type of window
    :return: custom Spectrogram object with plotting options
    """
    window_length_samples = int(window_length * fs)
    overlap_length_samples = int(window_length_samples * overlap_percent)
    spectrum_interval_length_samples = window_length_samples - overlap_length_samples
    window_array = get_window(window_type, window_length_samples)
    _, __, raw_spectrogram = spectrogram(signal, fs=fs, noverlap=overlap_length_samples, nfft=window_length_samples,
                                         mode='psd', scaling='spectrum', window=window_array)
    raw_spectrogram = np.log10(raw_spectrogram) * 10.  # dB conversion
    return Spectrogram(data=raw_spectrogram.T, data_fs=fs, spectrum_interval=spectrum_interval_length_samples / fs)


class STFT:

    def __init__(self, data: np.ndarray, data_f, data_t, fs: float,
                 spectrum_interval: float, window_length: float):
        """
        A Spectrogram. Fourier co-efficients over time

        :param data: 2D data (time x fourier co-eff)
        :param fs: sampling rate of the original data
        :param spectrum_interval: time interval between each spectrum (in Seconds)
        """
        self.data = data
        self.data_f = data_f
        self.data_t = data_t
        self.fs = fs
        self.fft_n = data.shape[1]
        self.num_data_point = data.shape[0]
        self.spectrum_interval = spectrum_interval
        self.window_length = window_length

    def freq_to_index(self, frequency: np.ndarray):
        """
        Convert the given frequency to the lowest column index for this spectrogram

        :param frequency: in Hz
        :return: Column index number
        """
        return (2 * np.asarray(frequency) * self.fft_n / self.fs).astype(int)

    def index_to_freq(self, index: np.ndarray):
        """
        Convert the given index to its related frequency

        :param index: index
        :return: Frequency in Hz
        """
        return np.asarray(index) * self.fs / self.fft_n / 2

    def inverse(self):
        window_length_samples = int(self.window_length * self.fs)
        window_array = get_window('hann', window_length_samples)
        noverlap_samples = int(window_length_samples -
                               self.spectrum_interval * self.fs)
        return istft(self.data.T, fs=self.fs, window=window_array, nperseg=window_length_samples,
                     noverlap=noverlap_samples, nfft=window_length_samples)

    def plot(self, frequency_range: Tuple[float, float]) -> AxesImage:
        """
        Plot the given spectrogram. Make sure the format is in (time x fourier coe-effs). This does not create a new plot

        :param frequency_range: Range of frequency to keep in the plot (in Hz)
        :return: matplotlib axis image
        """

        plt.figure()
        lower_index = self.freq_to_index(frequency_range[0])
        upper_index = self.freq_to_index(frequency_range[1]) + 1
        axes_image = plt.imshow(10 * np.log(np.abs((self.data[:, lower_index: upper_index]) ** 2)),
                                extent=[frequency_range[0], frequency_range[1],
                                        self.num_data_point * self.spectrum_interval, 0],
                                interpolation=None,
                                aspect="auto")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel("Time (Seconds)")
        plt.colorbar()
        return axes_image


def generate_STFT(signal: np.ndarray, fs: float, window_length: float = 60, overlap_percent: float = 3 / 4,
                  window_type: str = 'hann'):
    """
    Generate STFT from the given 1D Signal

    :param signal: 1D data
    :param fs: Sampling Frequency (Hz)
    :param window_length: Length(s) of each spectrogram window
    :param overlap_percent: 0 < p < 1, percentage overlap in spectrogram
    :param window_type: Type of window
    :return: custom STFT object with plotting options
    """
    window_length_samples = int(window_length * fs)
    overlap_length_samples = int(window_length_samples * overlap_percent)
    spectrum_interval_length_samples = window_length_samples - overlap_length_samples
    window_array = get_window('hann', window_length_samples)

    f, t, Zxx = stft(signal, fs=fs, noverlap=overlap_length_samples, window=window_array, nperseg=window_length_samples,
                     nfft=window_length_samples)

    return STFT(data=Zxx.T, data_f=f, data_t=t, fs=fs, spectrum_interval=spectrum_interval_length_samples / fs,
                window_length=window_length)


def get_manipulated_stft_inverse(sig: np.ndarray, FHR_arr, window_length=30, dc_bins_to_keep: int = 3,
                                 fhr_bins_to_keep: int = 3):
    """
    Filters the signal to keep only AC at FHR frequencies plus DC using STFT followed by filtering and an inverse STFT.

    :param sig: 1D channel signal to filter (Flattens internally)
    :param FHR_arr: 1Hz FHR for reference (in Hz)
    :param window_length: STFT window length in Seconds
    :param dc_bins_to_keep: Number of DC bins to keep in the reconstructed signal. 0 -> no DC
    :param fhr_bins_to_keep: Total number of FHR bins to keep in the reconstructed signal. Make sure it's an odd number.
    <Side bands L | FHR | Side bands R > : the total length
    1 -> only FHR, 3 -> FHR + 1 bin on both sides, 0 -> drop FHR AC
    :return: Recreated signal, same size as the original one
    """
    original_stft = generate_STFT(sig.flatten(), 80, window_length=window_length,
                                  overlap_percent=(window_length - 1) / window_length)

    # spectrum_replacement_value = np.min(np.abs(original_stft.data))
    # spectrum_replacement_value = np.min(np.abs(original_stft.data))
    spectrum_replacement_value = 0.
    if fhr_bins_to_keep != 0:  # If 0, keep as is
        # If even, change to the next largest odd number
        fhr_bins_to_keep = (fhr_bins_to_keep // 2) * 2 + 1

    # 'Same' Padding FHR_arr to make sure they have the same length
    l1 = len(FHR_arr) - window_length // 2
    l2 = len(original_stft.data)
    FHR_arr2 = np.copy(FHR_arr)
    if l1 < l2:
        FHR_arr2 = np.append(FHR_arr, np.repeat(FHR_arr[-1], l2 - l1))

    # Zero-out Non-FHR and non-DC components
    for time in range(len(original_stft.data)):
        select_ind = time + window_length // 2
        FHR = FHR_arr2[select_ind]
        fhr_index = original_stft.freq_to_index(
            FHR)  # From Hz to frequency index
        # Zero out everything except the left sideband of FHR and DC
        original_stft.data[time, dc_bins_to_keep:fhr_index -
                           fhr_bins_to_keep // 2] = spectrum_replacement_value
        # Zero out everything to the right of the right sideband of FHR
        original_stft.data[time, fhr_index +
                           (fhr_bins_to_keep - fhr_bins_to_keep // 2):] = spectrum_replacement_value

    _, reconstructed_signal = original_stft.inverse()
    reconstructed_signal = reconstructed_signal[:len(sig)]  # Cropping ends
    return reconstructed_signal


def get_manipulated_stft_inverse_kaiser(sig: np.ndarray, fhr_arr, window_length=30, M: int = 21, beta: int = 12,
                                        fhr_bins_to_keep: int = 3, fs=80):
    """
    Filters the signal to keep only AC at FHR frequencies plus DC using STFT followed by filtering and an inverse STFT.
    Uses a windowing instead of the direct one meow

    :param sig: 1D channel signal to filter (Flattens internally)
    :param fhr_arr: 1Hz FHR for reference (in Hz)
    :param M: Kaiser Window Param (Length - samples, should be an odd number)
    :param beta: Kaiser Window Param
    :param window_length: STFT window length in Seconds
    :param fhr_bins_to_keep: Total number of FHR bins to keep in the reconstructed signal. Make sure it's an odd number.
    <Side bands L | FHR | Side bands R > : the total length
    1 -> only FHR, 3 -> FHR + 1 bin on both sides, 0 -> drop FHR AC
    :param fs: Sampling Freq
    :return: Recreated signal, same size as the original one
    """
    original_stft = generate_STFT(sig.flatten(), fs, window_length=window_length,
                                  overlap_percent=(window_length - 1) / window_length)

    if fhr_bins_to_keep != 0:  # If 0, keep as is
        # If even, change to the next largest odd number
        fhr_bins_to_keep = (fhr_bins_to_keep // 2) * 2 + 1

    # 'Same' Padding FHR_arr to make sure they have the same length
    l1 = len(fhr_arr) - window_length // 2
    l2 = len(original_stft.data)
    FHR_arr2 = np.copy(fhr_arr)
    if l1 < l2:
        FHR_arr2 = np.append(fhr_arr, np.repeat(fhr_arr[-1], l2 - l1))

    crop_base_length = M//2 + 1   # make M odd
    fft_crop_base = windows.kaiser(crop_base_length, beta)

    # Zero-out Non-FHR and non-DC components using a window
    for time in range(len(original_stft.data)):
        select_ind = time + window_length // 2
        FHR = FHR_arr2[select_ind]
        # From Hz to frequency index - center of the crop window
        fhr_index = original_stft.freq_to_index(FHR)

        # Create fft crop window out of zeros and place the fft_crop_base centered at the fhr_index
        fft_crop_window = np.zeros((fs * window_length//2 + 1,))
        crop_base_start_index = fhr_index - crop_base_length // 2
        fft_crop_window[crop_base_start_index: crop_base_start_index +
                        crop_base_length] = fft_crop_base

        # Crop FFT with the window
        original_stft.data[time, :] *= fft_crop_window

    _, reconstructed_signal = original_stft.inverse()
    reconstructed_signal = reconstructed_signal[:len(sig)]  # Cropping ends
    return reconstructed_signal
