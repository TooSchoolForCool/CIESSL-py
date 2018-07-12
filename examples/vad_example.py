import sys
import threading
import signal

from ciessl.voice_engine.mic_array import MicArray
from ciessl.voice_engine.vad import VAD


def test_vad():
    vad_time_interval = 10
    sample_rate = 44100
    sample_rate_out = 32000
    chunk_size = sample_rate * vad_time_interval / 1000

    mic = MicArray(
        sample_rate_in=sample_rate,
        sample_rate_out=sample_rate_out,
        n_channels=16,
        chunk_size=chunk_size,
        format_in="int16"
    )

    vad = VAD(mode=3)

    # handle interrupt signal
    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print("Exit Signal (Ctrl + C) Received")
    signal.signal(signal.SIGINT, signal_handler)

    mic.start()
    print("Start recording...")

    for _, resampled_frames in mic.read_chunks():
        if is_quit.is_set():
            break

        if vad.is_speech(resampled_frames, mic.get_sample_rate_out(), mic.get_channels()):
            sys.stdout.write('1')
        else:
            sys.stdout.write('0')

        sys.stdout.flush()

    mic.stop()
    print("Stop recording")

if __name__ == '__main__':
    test_vad()