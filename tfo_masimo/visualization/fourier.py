"""
Includes 1D Fourier series plotting and visualization related functions
"""

from math import ceil
from typing import Tuple
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from scipy.signal import get_window


def plot_fourier(sequence: np.ndarray, n: int = None, sampling_frequency: int = None,
                 window: str = None) -> Tuple[plt.Axes, plt.Axes]:
    """
     Plot the real signal as well as the Fourier transform in a new Figure

     :param sequence: Original Sequence
     :param n: Number of points for FFT. Leave as None to get the same number of points as the sequence
     :param sampling_frequency: Useful for plotting the frequency axis
     :param window: Windowing function name to be passed into scipy.signal.get_window. If no name is given, uses
     rectangular window (i.e., no window)
     :return: Figure Axes (Time Series, Fourier Domain)
     """
    assert len(sequence) > 0, "Empty Sequence"
    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(sequence)
    ax1.set_title("Original Signal")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Seconds")
    current_ticks = ax1.get_xticks()
    ax1.set_xticks(current_ticks)
    ax1.set_xticklabels(np.round(current_ticks / sampling_frequency, 0))

    ax2.set_title("Fourier Domain")
    if window is not None:
        sequence = sequence * get_window(window, len(sequence))
    _plot_fourier(sequence, ax2, n, sampling_frequency)
    return ax1, ax2


def plot_fourier_exp(sequence: np.ndarray, n: int = None,
                     sampling_frequency: int = None) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot the real signal as well as the Fourier transform. This plot is made to mess around with the spectrum and
    try out different stuff

    :param sequence: Original Sequence
    :param n: Number of points for FFT. Leave as None to get the same number of points as the sequence
    :param sampling_frequency: Useful for plotting the frequency axis
    :return: Figure Axes (Time Series, Fourier Domain)
    """
    assert len(sequence) > 0, "Empty Sequence"
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(sequence)
    ax1.set_title("Original Signal")

    ax2.set_title("Fourier Domain")
    _plot_fourier_exp(sequence, ax2, n, sampling_frequency)
    return ax1, ax2


def _plot_fourier(sequence: np.ndarray, ax: plt.Axes, n: int = None, sampling_frequency: float = None):
    """
    Private function to plot the fourier transform of the sequence on a gives Axes
    """
    assert len(sequence) > 0, "Empty Sequence"
    if n is None:
        n = len(sequence)
    fourier = fft(sequence, n=n)
    half_point = ceil(len(fourier) / 2)
    if sampling_frequency is not None:
        # frequencies = np.arange(0, sampling_frequency, step=sampling_frequency / n)
        frequencies = fftfreq(n, 1 / sampling_frequency)
        ax.plot(frequencies[: half_point], np.abs(fourier[:half_point]))
        ax.set_xlabel("Hz")
    else:
        ax.plot(np.abs(fourier[:half_point]))
    return


def _plot_fourier_exp(sequence: np.ndarray, ax: plt.Axes, n: int = None, sampling_frequency: float = None) -> None:
    """
    Private function to plot the fourier transform of the sequence on a gives Axes
    """
    assert len(sequence) > 0, "Empty Sequence"
    if n is None:
        n = len(sequence)
    fourier = fft(sequence, n=n)
    energy = np.sum(np.square(fourier))
    fourier = fourier / energy
    half_point = ceil(len(fourier) / 2) + 1
    if sampling_frequency is not None:
        frequencies = np.arange(0, sampling_frequency,
                                step=sampling_frequency / n)
        ax.plot(frequencies[: half_point], np.abs(fourier[:half_point]))
        ax.set_xlabel("Hz")
    else:
        ax.plot(np.abs(fourier[:half_point]))
