import os
import sys
import threading
import signal

from voice_engine.vad import VAD
from voice_engine.wave_source import WaveSource


def wav_vad():
    current_file_path = os.path.dirname(__file__)
    # wav_file_path = "../data/sample/6_claps_int16.wav"
    wav_file_path = "../data/sample/ch8-raw.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    chunk_time_interval = 20
    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=chunk_time_interval,
        format_out="int16"
    )

    vad = VAD(mode=3)

    time_cnt = 0
    is_active = False
    active_voice = []
    start_time, end_time = 0, 0

    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        if vad.is_speech(resampled_frames, ws.get_sample_rate_out(), ws.get_channels()):
            sys.stdout.write('1')
            if not is_active:
                is_active = True
                start_time = time_cnt
        else:
            sys.stdout.write('0')
            if is_active:
                is_active = False
                end_time = time_cnt
                active_voice.append( (start_time/1000.0, end_time/1000.0) )

        time_cnt += chunk_time_interval
        sys.stdout.flush()

    if is_active:
        end_time = time_cnt
        active_voice.append( (start_time/1000.0, end_time/1000.0) )

    print("\nActive Voice Interval is:")
    for av in active_voice:
        print("{} ~ {} s".format(av[0], av[1]))

    ws.stop()


if __name__ == '__main__':
    wav_vad()
