import numpy as np
from scipy import linalg
import scipy.signal as sg

window_step = int(128*1.) #128
window_size = window_step*2

def soundsc(X, gain_scale=.9, copy=True):
    """
    From https://github.com/kastnerkyle/tools
    Approximate implementation of soundsc from MATLAB.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    gain_scale : float
        Gain multipler, default .9 (90% of maximum representation)

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-32767, 32767) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """

    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')

def sinusoid_analysis(X, input_sample_rate, resample_block=window_step, copy=True):
    """
    From https://github.com/kastnerkyle/tools
    Contruct a sinusoidal model for the input signal.

    Parameters
    ----------
    X : ndarray
        Input signal to model

    input_sample_rate : int
        The sample rate of the input signal

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    frequencies_hz : ndarray
       Frequencies for the sinusoids, in Hz.

    magnitudes : ndarray
       Magnitudes of sinusoids returned in ``frequencies``

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """

    X = np.array(X, copy=copy)
    resample_to = 8000
    if input_sample_rate != resample_to:
        #if input_sample_rate % resample_to != 0:
        #    raise ValueError("Input sample rate must be a multiple of 8k!")
        # Should be able to use resample... ?

        resampled_count = round(len(X) * resample_to / input_sample_rate)
        X = sg.resample(X, resampled_count) #, window=sg.hanning(len(X)))

        #X = sg.decimate(X, input_sample_rate // resample_to, zero_phase=True)
    step_size = 2 * round(resample_block / input_sample_rate * resample_to / 2.)
    print(step_size)
    a, g, e = lpc_analysis(
        X, order=8,
        window_step=step_size,
        window_size=2 * step_size
        )
    f, m = lpc_to_frequency(a, g)
    f_hz = f * resample_to / (2 * np.pi)
    return f_hz, m

def sinusoid_synthesis(frequencies_hz, magnitudes, input_sample_rate=16000,
                       resample_block=window_step):
    """
    From https://github.com/kastnerkyle/tools
    Create a time series based on input frequencies and magnitudes.

    Parameters
    ----------
    frequencies_hz : ndarray
        Input signal to model

    magnitudes : int
        The sample rate of the input signal

    input_sample_rate : int, optional (default=16000)
        The sample rate parameter that the sinusoid analysis was run with

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """

    rows, cols = frequencies_hz.shape
    synthesized = np.zeros((1 + ((rows - 1) * resample_block),))
    for col in range(cols):
        mags = slinterp(magnitudes[:, col], resample_block)
        freqs = slinterp(frequencies_hz[:, col], resample_block)
        cycles = np.cumsum(2 * np.pi * freqs / float(input_sample_rate))
        sines = mags * np.cos(cycles)
        synthesized += sines
    return synthesized

def lpc_analysis(X, order=8, window_step=window_step, window_size=window_size,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    From https://github.com/kastnerkyle/tools
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = int(n_points // window_step)
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], int(pad_sizes[0]))), X,
                       np.zeros((X.shape[0], int(pad_sizes[1])))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        int(((n_windows - 1) * window_step + window_size)))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, int(2 * window_size - 1)))
    for window in range(max(n_windows - 1, 1)):
        wtws = int(window * window_step)
        XX = X.ravel()[wtws + np.arange(window_size, dtype="int32")]
        WXX = XX * sg.hanning(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order, dtype="int32")]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs
        per_frame_gain[window] = gain
        assign_range = wtws + np.arange(window_size, dtype="int32")
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[int(pad_sizes[0]):]
    return lp_coefficients, per_frame_gain, residual_excitation

def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    From https://github.com/kastnerkyle/tools
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``

    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``

    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi

    magnitudes : ndarray
       Magnitudes of resonant frequencies

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes

def slinterp(X, factor, copy=True):
    """
    From https://github.com/kastnerkyle/tools
    Slow-ish linear interpolation of a 1D numpy array. There must be some
    better function to do this in numpy.

    Parameters
    ----------
    X : ndarray
        1D input array to interpolate

    factor : int
        Integer factor to interpolate by

    Return
    ------
    X_r : ndarray
    """
    sz = np.product(X.shape)
    X = np.array(X, copy=copy)
    X_s = np.hstack((X[1:], [0]))
    X_r = np.zeros((factor, sz))
    for i in range(factor):
        X_r[i, :] = (factor - i) / float(factor) * X + (i / float(factor)) * X_s
    return X_r.T.ravel()[:(sz - 1) * factor + 1]
