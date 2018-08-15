import Queue
import threading

import scipy.io.wavfile as wf
import numpy as np
import samplerate as sr

from audio_source import AudioSource
from global_var import STR2NUMPY_FORMAT
from signal_process import convert_type


class WaveSource(AudioSource):

    def __init__(self, file_dir, format_out=None, sample_rate_out=None, chunk_time_interval=10):
        self.sample_rate_in_, self.data_ = wf.read(file_dir)
        self.format_in_ = self.data_.dtype
        self.format_out_ = self.format_in_ if format_out is None else STR2NUMPY_FORMAT[format_out]

        try:
            self.n_channels_ = self.data_.shape[1]
        except:
            self.n_channels_ = 1

        # chunk time interval (in second)
        self.chunk_interval_ = chunk_time_interval / 1000.0
        
        self.sample_rate_out_ = self.sample_rate_in_ if sample_rate_out is None else sample_rate_out
        self.chunk_size_ = self.sample_rate_in_ * chunk_time_interval / 1000
        self.fp_ = 0

        self.quit_event_ = threading.Event()

        self.resampler_ = sr.Resampler(converter_type='linear', channels=self.n_channels_)
        self.ratio_ = 1.0 * self.sample_rate_out_ / self.sample_rate_in_


    def start(self):
        self.fp_ = 0
        self.quit_event_.clear()


    def stop(self):
        self.quit_event_.set()
        self.resampler_.reset()


    def read_chunks(self):
        while not self.quit_event_.is_set():
            # reach End of File
            if self.fp_ >= self.data_.shape[0]:
                break

            if self.fp_ + self.chunk_size_ <= self.data_.shape[0]:
                raw_frames = self.data_[self.fp_:self.fp_ + self.chunk_size_, :]
            else:
                raw_frames = self.data_[self.fp_:, :]

                add_on_size = self.fp_ + self.chunk_size_ - self.data_.shape[0]
                add_on = np.zeros((add_on_size, self.n_channels_), dtype=self.format_in_)
                raw_frames = np.append(raw_frames, add_on, axis=0)

            self.fp_ += self.chunk_size_

            if self.sample_rate_in_ == self.sample_rate_out_:
                resampled_frames = raw_frames
            else:
                resampled_frames = self.__resample(raw_frames)

            resampled_frames = convert_type(resampled_frames, self.format_in_, self.format_out_)

            yield raw_frames, resampled_frames


    def get_sample_rate_in(self):
        return self.sample_rate_in_


    def get_sample_rate_out(self):
        return self.sample_rate_out_


    def get_channels(self):
        return self.n_channels_


    def get_dtype(self):
        return self.format_in_


    def get_chunk_interval(self):
        return self.chunk_interval_


    def __resample(self, raw_frames):
        resampled_frames = self.resampler_.process(raw_frames, self.ratio_).astype(self.format_in_)
        return resampled_frames


def test_wav_file():
    import os
    
    current_file_path = os.path.dirname(__file__)
    wav_file_path = "../../data/sample/ch8-raw.wav"
    WAV_FILE_DIR = wav_file_path if not current_file_path else current_file_path + "/" + wav_file_path
    
    ws = WaveSource(
        file_dir=WAV_FILE_DIR,
        sample_rate_out=32000,
        chunk_time_interval=10,
        format_out="int16"
    )

    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        print("raw frame: ", raw_frames.shape)
        print(raw_frames[:10, 0])
        print("resampled frame: ", resampled_frames.shape)
        print(resampled_frames[:10, 0])
    ws.stop()

    print("sample rate in: %r" % ws.get_sample_rate_in())
    print("channels: %r" % ws.get_channels())
    print("data_tpye: %r" % ws.get_dtype())


if __name__ == '__main__':
    test_wav_file()