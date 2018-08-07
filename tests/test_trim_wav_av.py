import os
import signal
import threading

from voice_engine.utils import write2wav
from voice_engine.wave_source import WaveSource
from voice_engine.active_voice_trimmer import ActiveVoiceTrimmer


def test_save_wav_active_voice():
    current_file_path = os.path.dirname(__file__)
    wav_file_path = "../data/sample/6_claps_int16.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=10
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

        print("active voice received")
        write2wav(active_frames, ws.get_channels(), ws.get_sample_rate_in(), 
            output="active_voice_" + str(voice_cnt) + ".wav")
        voice_cnt += 1
    
    avt.stop()
    print("Stop listening")


if __name__ == '__main__':
    test_save_wav_active_voice()