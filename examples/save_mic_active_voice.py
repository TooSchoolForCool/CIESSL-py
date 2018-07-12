import signal
import threading

from ciessl.voice_engine.utils import write2wav
from ciessl.voice_engine.mic_array import MicArray
from ciessl.voice_engine.vad import VAD
from ciessl.voice_engine.active_voice_trimmer import ActiveVoiceTrimmer


def save_mic_active_voice():
    sample_rate = 48000
    vad_time_interval = 10
    chunk_size = sample_rate * vad_time_interval / 1000

    mic = MicArray(
        sample_rate=sample_rate,
        n_channels=16,
        chunk_size=chunk_size,
        format_in="int16"
    )

    voice_cnt = 0

    avt = ActiveVoiceTrimmer(mode=2, audio_source=mic)

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

        print("active voice received")
        write2wav(active_frames, mic.get_channels(), mic.get_sample_rate(), 
            output="active_voice_" + str(voice_cnt) + ".wav")
        voice_cnt += 1
    
    avt.stop()
    print("Stop listening")


if __name__ == '__main__':
    save_mic_active_voice()