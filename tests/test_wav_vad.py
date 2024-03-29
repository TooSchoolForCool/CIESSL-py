import os
import sys
import threading
import signal

from voice_engine.vad import VAD
from voice_engine.wave_source import WaveSource


def test_wav_vad():
    current_file_path = os.path.dirname(__file__)
    wav_file_path = "../data/sample/6_claps_int16.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=10
    )

    vad = VAD(mode=2)

    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        if vad.is_speech(resampled_frames, ws.get_sample_rate_out(), ws.get_channels()):
            sys.stdout.write('1')
        else:
            sys.stdout.write('0')
        sys.stdout.flush()
    ws.stop()


if __name__ == '__main__':
    test_wav_vad()
