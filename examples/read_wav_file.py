import os

from ciessl.voice_engine.wave_source import WaveSource


def read_wav_file():
    current_file_path = os.path.dirname(__file__)
    wav_file_path = "../assets/6_claps_int16.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=10
    )

    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        print("raw frame: ", raw_frames.shape)
        print("resampled frame: ", resampled_frames.shape)
    ws.stop()

    print("sample rate in: %r" % ws.get_sample_rate_in())
    print("channels: %r" % ws.get_channels())
    print("data_tpye: %r" % ws.get_dtype())


if __name__ == '__main__':
    read_wav_file()