import os

from voice_engine.wave_source import WaveSource


def test_read_wav_file():
    current_file_path = os.path.dirname(__file__)
    wav_file_path = "../data/sample/6_claps_int16.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path

    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=10
    )

    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        print("raw frames {}, {}".format(raw_frames.shape[0], raw_frames.shape[1]))
        print("resampled frames {}, {}".format(resampled_frames.shape[0], resampled_frames.shape[1]))
    ws.stop()

    print("sample rate in: %r" % ws.get_sample_rate_in())
    print("channels: %r" % ws.get_channels())
    print("data_tpye: %r" % ws.get_dtype())


if __name__ == '__main__':
    test_read_wav_file()