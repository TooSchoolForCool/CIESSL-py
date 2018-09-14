import numpy as np
import librosa
from scipy import signal as sig

import global_var


def float32_to_int16(frames):
    convert_float32_to_int16 = lambda x : (x * (0x7FFF + 0.5) - 0.5)
    vectorize_func = np.vectorize(convert_float32_to_int16)

    return vectorize_func(frames)

def convert_type(frames, format_in, format_out):
    if format_in not in global_var.DATA_FORMAT2STR:
        raise Exception("Do NOT support format {}".format(format_in))
    if format_out not in global_var.DATA_FORMAT2STR:
        raise Exception("Do NOT support format {}".format(format_out))

    # convert format type to string
    format_in = global_var.DATA_FORMAT2STR[format_in]
    format_out = global_var.DATA_FORMAT2STR[format_out]

    if format_in == format_out:
        return frames
    elif format_in == "float32" and format_out == "int16":
        return float32_to_int16(frames).astype(global_var.STR2NUMPY_FORMAT[format_out])
    else:
        raise Exception("Do NOT support converting from {} to {}".format(format_in, format_out))


def stft(frames, sample_rate, window=('tukey', .25), segment_size=None, overlap_size=None, nfft=None):
    freqs, time, spec = sig.stft(frames, sample_rate, window=window, nperseg=segment_size,
        noverlap=overlap_size, nfft=nfft)

    # amplitude component of STFT
    amp = np.abs(spec)
    # phase component of STFT
    phase = np.angle(spec)

    return freqs, time, amp, phase


def gcc_phat(sig, refsig, sample_rate, max_tau=None, interp=1):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.

    Args:
        sig ( np.ndarray (n_frames, ) ): the first voice signal
        refsig ( np.ndarray (n_frames, ) ): the second voice signal (reference signal)
        sample_rate (int): sample rate of the signal
        max_tau (float): pre-calculated maximum time delay
        interp (int): interpolation

    Returns:
        tau (float): Time Delay
        inv_cc (np.array (gcc_phat_len, )): cross-correlation between signals
        center (int): the index of the center, where TDOA is 0, of the cross-correlation
            coefficients
    """
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = R / (np.abs(R) + 0.00001)
    inv_cc = np.fft.irfft(cc, n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * sample_rate * max_tau), max_shift)

    inv_cc = np.concatenate((inv_cc[-max_shift:], inv_cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(inv_cc)) - max_shift
    
    # print("shift = %d" % shift)
    # print("max_shift = %d" % max_shift)

    tau = shift / float(interp * sample_rate)
    center = max_shift + shift

    return tau, inv_cc, center


def gccfb(sig, refsig, sample_rate, max_tau=None, interp=1, n_mels=2, f_size=25):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.

    Args:
        sig ( np.ndarray (n_frames, ) ): the first voice signal
        refsig ( np.ndarray (n_frames, ) ): the second voice signal (reference signal)
        sample_rate (int): sample rate of the signal
        max_tau (float): pre-calculated maximum time delay
        interp (int): interpolation
        f_size (int): feature size

    Returns:
        tau (float): Time Delay
        inv_cc (np.array (gcc_phat_len, )): cross-correlation between signals
        center (int): the index of the center, where TDOA is 0, of the cross-correlation
            coefficients
    """
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]
    melW=librosa.filters.mel(sr=sample_rate, n_fft=n, n_mels=n_mels, fmin=100, fmax=8000)

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)
    cc = R / (np.abs(R) + 0.00001)
    mel_cc = melW * cc

    inv_cc = np.fft.irfft(mel_cc, n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * sample_rate * max_tau), max_shift)

    inv_cc = np.concatenate((inv_cc[-max_shift:], inv_cc[:max_shift+1]), axis=1)

    # find max cross correlation index
    shift = np.argmax(np.abs(inv_cc), axis=1) - max_shift

    tau = shift / float(interp * sample_rate)
    center = max_shift + shift
    
    gccfb_feature = []

    for sig, c in zip(inv_cc, center):
        # crop gccfb features
        if c - f_size < 0:
            feat = sig[0 : 2 * f_size]
        elif c + f_size >= sig.shape[0]:
            feat = sig[-2 * f_size : ]
        else:
            feat = sig[c - f_size : c + f_size]
        gccfb_feature.append(feat)

    gccfb_feature = np.asarray(gccfb_feature).flatten()

    return gccfb_feature


################################################################
# Test Cases
################################################################

def test_converter():
    A = np.array([[-1.0, 1.0], [0.5, -0.5], [0, 0]])
    print(A)
    A = convert_type(A, "float32", "int16")
    print(A)


def test_gcc_phat():
    import matplotlib.pyplot as plt

    voice_file = "../data/active_voice/48000-7-1-3-4.pickle"
    voice_frames = np.load(voice_file)

    tau, cc, center = gcc_phat(voice_frames[:, 1], voice_frames[:, 7], 48000, interp=1)

    print(tau)
    print(cc.shape)
    plt.plot(cc[center-15 : center+15])
    # plt.plot(cc)
    plt.show()


if __name__ == '__main__':
    test_gcc_phat()
