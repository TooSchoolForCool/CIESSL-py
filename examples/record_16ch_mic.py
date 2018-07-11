import signal
import time
import threading

import numpy as np

from ciessl.voice_engine.mic_array import MicArray
from ciessl.voice_engine.utils import write2wav

def test_mic_array():
    mic = MicArray(
        sample_rate=48000,
        n_channels=16,
        chunk_size=4096,
        format_in="int16"
    )

    # handle interrupt signal
    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print("\nExit Signal (Ctrl + C) Received")
    signal.signal(signal.SIGINT, signal_handler)

    formed_sequence = None

    mic.start()
    print("Start recording...")

    for formed_frames in mic.read_chunks():
        if is_quit.is_set():
            break

        if formed_sequence is None:
            formed_sequence = formed_frames
        else:
            formed_sequence = np.append(formed_sequence, formed_frames)

    mic.stop()
    print("Stop recording")

    write2wav(formed_sequence, n_channels=mic.get_channels(), 
    	rate=mic.get_sample_rate(), output="16ch_48kHz_int16.wav")
    

if __name__ == '__main__':
    test_mic_array()