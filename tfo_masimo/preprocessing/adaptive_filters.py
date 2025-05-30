"""
Contains RLS filtering related helper functions
"""
from typing import Tuple

import numpy as np
import padasip as pa


def adaptive_filter_rls(noise_reference: np.ndarray, mixed_signal: np.ndarray,
                        filter_lambda: float = 0.99, filter_length: int = 100) -> Tuple:
    """
    Custom RLS Adaptive filter using the padasip library
    :param noise_reference: 1D noise reference signal
    :param mixed_signal: 1D Noise + Signal
    :param filter_lambda: RLS filter parameter - Forgetting Factor
    :param filter_length: RLS filter parameter - Filter length
    :return: 1D Reconstructed Signal, Residual Signal, Filter Weights (Note: Both signal lengths 
    are reduced by filter_length from the noise reference & the mixed signal due to how the
    filter works. The RLS filter takes a few samples before settling. The initial few values 
    might be unusable)
    """

    # filtering
    # Make a 2D matrix from the reference. Keeps the most recent filter_length points on each row
    input_matrix = pa.input_from_history(noise_reference, filter_length)[:-1]
    mixed_signal_temp = mixed_signal[filter_length:]
    rls_filter = pa.filters.FilterRLS(mu=filter_lambda, n=filter_length)
    recon_signal, residual_signal, filter_w = rls_filter.run(
        mixed_signal_temp, input_matrix)
    return recon_signal, residual_signal, filter_w


def adaptive_filter_lms(noise_reference: np.ndarray, mixed_signal: np.ndarray,
                        filter_lambda: float = 0.01, filter_length: int = 100) -> Tuple:
    """
    Custom LMS Adaptive filter using the padasip library
    :param noise_reference: 1D noise reference signal
    :param mixed_signal: 1D Noise + Signal
    :param filter_lambda: LMS filter parameter - Mu
    :param filter_length: LMS filter parameter - Filter length
    :return: 1D Reconstructed Signal, Residual Signal, Filter Weights (Note: Both signal lengths 
    are reduced by filter_length from the noise reference & the mixed signal due to how the
    filter works. The RLS filter takes a few samples before settling. The initial few values 
    might be unusable)
    """

    # filtering
    # Make a 2D matrix from the reference. Keeps the most recent filter_length points on each row
    input_matrix = pa.input_from_history(noise_reference, filter_length)[:-1]
    mixed_signal_temp = mixed_signal[filter_length:]
    lms_filter = pa.filters.FilterLMS(mu=filter_lambda, n=filter_length)
    recon_signal, residual_signal, filter_w = lms_filter.run(
        mixed_signal_temp, input_matrix)
    return recon_signal, residual_signal, filter_w
