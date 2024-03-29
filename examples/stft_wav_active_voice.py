import os
import signal
import threading

import matplotlib.pyplot as plt
import numpy as np

from voice_engine.utils import write2wav
from voice_engine.wave_source import WaveSource
from voice_engine.active_voice_trimmer import ActiveVoiceTrimmer
from voice_engine.signal_process import stft

def save_mic_active_voice():
    current_file_path = os.path.dirname(__file__)
    # wav_file_path = "../data/sample/6_claps_int16.wav"
    wav_file_path = "../data/sample/ch8-raw.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=20,
        format_out="int16"
    )

    voice_cnt = 0

    avt = ActiveVoiceTrimmer(mode=3, audio_source=ws)

    # handle interrupt signal
    is_quit = threading.Event()
    def signal_handler(sig, num):
        print("Exit Signal (Ctrl + C) Received")
        is_quit.set()
    signal.signal(signal.SIGINT, signal_handler)

    print("Start listening ...")
    avt.start()

    for active_frames in avt.read_active_chunks():
        if is_quit.is_set():
            break

        print("active voice {} received".format(voice_cnt))


        f, t, amp, phase = stft(active_frames[:, 0], ws.get_sample_rate_in(), 
            segment_size=256, overlap_size=250)

        print("{} --> {}, {}".format(active_frames[:, 0].shape, amp.shape, phase.shape))

        # log-scale
        amp = np.log10(amp)
        plt.pcolormesh(t, f, amp)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("STFT dB")
        plt.show()

        plt.pcolormesh(t, f, phase)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("STFT Phase")
        plt.show()
        
        voice_cnt += 1
    
    avt.stop()
    print("Stop listening")


if __name__ == '__main__':
    save_mic_active_voice()