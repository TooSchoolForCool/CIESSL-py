import numpy as np
from scipy import signal as sig

import global_var


def float32_to_int16(frames):
    def __float32_to_int16(x):
        return x * (0x7FFF + 0.5) - 0.5

    vectorize_func = np.vectorize(__float32_to_int16)

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


def stft(frames, sample_rate, window=('tukey', .25), segment_size=None, overlap_size=None):
    freqs, time, spec = sig.stft(frames, sample_rate, window, segment_size, overlap_size)

    # amplitude component of STFT
    amp = np.abs(spec)
    # phase component of STFT
    phase = np.angle(spec)

    return freqs, time, amp, phase


def test_converter():
    A = np.array([[-1.0, 1.0], [0.5, -0.5], [0, 0]])
    print(A)
    A = convert_type(A, "float32", "int16")
    print(A)


if __name__ == '__main__':
    test_converter()