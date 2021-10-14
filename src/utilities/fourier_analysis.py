import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import scipy.signal
import scipy.signal.signaltools


def simple_windowed_stft(x, sr=8000, nfft=512, nperseg_ms=0.025, noverlap_ms=0,
                         window="hann"):
    nperseg = int(sr * nperseg_ms)
    if noverlap_ms is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(sr * noverlap_ms)

    win, nperseg = scipy.signal.spectral._triage_segments("hann", nperseg, x.shape[-1])

    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    def detrend_func(d):
        return scipy.signal.signaltools.detrend(d, type="constant", axis=-1)

    # corresponding to density scaling, default in scipy.signal.stft
    scale = 1.0 / (sr * (win * win).sum())
    scale = np.sqrt(scale)

    frames = detrend_func(frames)
    # frames =


def compute_fft(s, sampling_rate, n=None, scale_amplitudes=True):
    """Computes an FFT on signal s using numpy.fft.fft.

       Parameters:
        s (np.array): the signal
        sampling_rate (num): sampling rate
        n (integer): If n is smaller than the length of the input, the input is cropped. If n is
            larger, the input is padded with zeros. If n is not given, the length of the input signal
            is used (i.e., len(s))
        scale_amplitudes (boolean): If true, the spectrum amplitudes are scaled by 2/len(s)
    """
    if n == None:
        n = len(s)

    fft_result = np.fft.fft(s, n)
    num_freq_bins = len(fft_result)
    fft_freqs = np.fft.fftfreq(num_freq_bins, d=1 / sampling_rate)
    half_freq_bins = num_freq_bins // 2

    fft_freqs = fft_freqs[:half_freq_bins]
    fft_result = fft_result[:half_freq_bins]
    fft_amplitudes = np.abs(fft_result)

    if scale_amplitudes is True:
        fft_amplitudes = 2 * fft_amplitudes / (len(s))
    return fft_freqs, fft_amplitudes


def get_top_n_frequency_peaks(n, freqs, amplitudes, min_amplitude_threshold=None):
    """ Finds the top N frequencies and returns a sorted list of tuples (freq, amplitudes) """

    # Use SciPy signal.find_peaks to find the frequency peaks
    # TODO: in future, could add in support for min horizontal distance so we don't find peaks close together
    fft_peaks_indices, fft_peaks_props = sp.signal.find_peaks(amplitudes, height=min_amplitude_threshold)

    freqs_at_peaks = freqs[fft_peaks_indices]
    amplitudes_at_peaks = amplitudes[fft_peaks_indices]

    if n < len(amplitudes_at_peaks):
        ind = np.argpartition(amplitudes_at_peaks, -n)[-n:]  # from https://stackoverflow.com/a/23734295
        ind_sorted_by_coef = ind[np.argsort(-amplitudes_at_peaks[ind])]  # reverse sort indices
    else:
        ind_sorted_by_coef = np.argsort(-amplitudes_at_peaks)

    return_list = list(zip(freqs_at_peaks[ind_sorted_by_coef], amplitudes_at_peaks[ind_sorted_by_coef]))
    return return_list, ind_sorted_by_coef


def apply_notch_filter(audio, notch_frequency, sample_rate, quality_factor=30):
    if isinstance(audio, torch.Tensor):
        audio_copy = audio.detach().squeeze().cpu().numpy()
    else:
        audio_copy = audio.copy()
    b_notch, a_notch = sp.signal.iirnotch(notch_frequency, quality_factor, sample_rate)
    freq, h = sp.signal.freqz(b_notch, a_notch, fs=sample_rate)
    y_notched = sp.signal.filtfilt(b_notch, a_notch, audio_copy)
    return y_notched


def plot_frequency_amplitudes(fft_freqs, fft_amplitudes_scaled,
                              label_top_n=False, top_n=5,
                              label='fft_amplitudes_scaled'):
    plt.plot(fft_freqs, fft_amplitudes_scaled, label=label)
    if label_top_n:
        top_n_peaks, sorted_indices = get_top_n_frequency_peaks(top_n, fft_freqs, fft_amplitudes_scaled)
        for ix in range(top_n):
            freq_ix, amp_ix = top_n_peaks[ix]
            plt.plot(freq_ix, amp_ix, marker='x', color='black', alpha=0.8)
            plt.text(freq_ix+3, amp_ix, "{:.4f} Hz".format(freq_ix), color='black')
    plt.ylabel("Frequency amplitudes (scaled)")
    plt.xlabel("Frequency")


def plot_frequency_amplitudes_from_waveform(waveform, sr=8000,
                                            label_top_n=False, top_n=5,
                                            label='fft_amplitudes_scaled'
                                            ):
    fft_freqs, fft_amplitudes_scaled = compute_fft(waveform, sr)
    plot_frequency_amplitudes(fft_freqs, fft_amplitudes_scaled, label_top_n=label_top_n,
                              top_n=top_n, label=label)


def perform_stft(w, sr=8000, nperseg_ms=0.025, noverlap_ms=0, nfft=8000, boundary=None):
    f, t, zxx = scipy.signal.stft(w, fs=sr, nperseg=int(sr*nperseg_ms), noverlap=int(sr*noverlap_ms), nfft=nfft,
                                  boundary=boundary)
    boundary_funcs = {'even': scipy.signal._arraytools.even_ext,
                      'odd': scipy.signal._arraytools.odd_ext,
                      'constant': scipy.signal._arraytools.const_ext,
                      'zeros': scipy.signal._arraytools.zero_ext,
                      None: None}
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(w, int(sr * nperseg_ms) // 2, axis=-1)
    else:
        x = None

    return f, t, zxx, x


def framewise_notch_filter(inp_waveform, fft_input, fft_decoded, freqs, times,
                           sr=8000, top_n_freqs=5, nperseg_ms=0.025):
    assert fft_input.shape == fft_decoded.shape

    # notched_audio = inp_waveform.copy()

    # framewise_to_drop_freqs = []
    output_audio = None
    prev_index = 0
    for i in range(fft_decoded.shape[1]):
        top_n_peaks, sorted_indices = get_top_n_frequency_peaks(top_n_freqs, freqs, fft_decoded[:, i],
                                                                min_amplitude_threshold=1e-4)
        top_n_peaks_signal, _ = get_top_n_frequency_peaks(top_n_freqs, freqs, fft_input[:, i],
                                                                min_amplitude_threshold=1e-4)
        # print("decoded:", top_n_peaks)
        # print("signal:", top_n_peaks_signal)
        to_drop_freqs = [t[0] for t in top_n_peaks]
        start_index = int(sr * times[i])
        end_index = int(sr * (times[i]+nperseg_ms))
        # end_index = int(sr * times[i])
        # time_index = int(sr * times[i])
        # print(i, start_index, end_index)
        notched_segment = inp_waveform.copy()[start_index: end_index]
        for drop_freq in to_drop_freqs:
            # print("\t", i, drop_freq)
            # print("before:", notched_audio[start_index: end_index])
            # output = apply_notch_filter(notched_audio[start_index: end_index], drop_freq, sample_rate=sr)
            # notched_audio[start_index: end_index] = output
            notched_segment = apply_notch_filter(notched_segment, drop_freq, sample_rate=sr)
            # print("after:", notched_audio[start_index: end_index])

        if output_audio is None:
            output_audio = notched_segment
        else:
            output_audio = np.append(output_audio, notched_segment)

    return output_audio


def get_top_k_coherence(x, y, sr=8000, nperseg_ms=0.02, noverlap_ms=0.01, nfft=512, top_n_peaks=5):
    f_corr, Cxy = scipy.signal.coherence(x, y, fs=sr, nperseg=int(sr*nperseg_ms),
                                         noverlap=int(sr*noverlap_ms), nfft=nfft)
    top_n_peaks, _ = get_top_n_frequency_peaks(top_n_peaks, f_corr, Cxy)