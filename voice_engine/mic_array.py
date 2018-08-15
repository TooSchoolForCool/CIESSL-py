import collections
import threading
import math

import pyaudio
import numpy as np
import samplerate as sr

from global_var import STR2NUMPY_FORMAT, STR2PYAUDIO_FORMAT
from audio_source import AudioSource
from signal_process import convert_type


# Define some global variables here
MAX_QUEUE_SIZE = 32


class MicArray(AudioSource):
    
    def __init__(
        self, 
        sample_rate_in=44100,
        sample_rate_out=32000,
        n_channels=16, 
        chunk_size=4096, 
        format_in="int16",
        format_out=None
    ):
        self.pyaudio_ins_ = pyaudio.PyAudio()

        self.queue_ = collections.deque(maxlen=MAX_QUEUE_SIZE)

        self.quit_event_ = threading.Event()
        self.queue_cond_ = threading.Condition()
        
        self.n_channels_ = n_channels
        self.sample_rate_in_ = sample_rate_in
        self.sample_rate_out_ = sample_rate_out
        # chunk time interval (in second)
        self.chunk_interval_ = 1.0 * chunk_size / sample_rate_in
        self.chunk_size_ = chunk_size
        self.pyaudio_format_ = STR2PYAUDIO_FORMAT[format_in]
        self.np_format_ = STR2NUMPY_FORMAT[format_in]
        self.format_out_ = self.np_format_ if format_out is None else STR2NUMPY_FORMAT[format_out]

        device_idx = self.__find_device_index()
        
        self.resampler_ = sr.Resampler(converter_type='linear', channels=self.n_channels_)
        self.ratio_ = 1.0 * self.sample_rate_out_ / self.sample_rate_in_

        self.stream_ = self.pyaudio_ins_.open(
            input=True,
            start=False,
            format=self.pyaudio_format_,
            channels=self.n_channels_,
            rate=self.sample_rate_in_,
            frames_per_buffer=self.chunk_size_,
            stream_callback=self.__callback,
            input_device_index=device_idx,
        )

        print("[INFO] Microphone Array Init is done")


    def __del__(self):
        self.stream_.close()
        self.pyaudio_ins_.terminate()


    def start(self):
        if self.stream_.is_active():
            return

        self.queue_.clear()
        self.stream_.start_stream()


    def stop(self):
        if self.stream_.is_stopped():
            return

        self.quit_event_.set()
        self.stream_.stop_stream()
        self.resampler_.reset()

        # let read_chunks out of blocking when 
        # there is no input into queue any more
        self.queue_cond_.acquire()
        self.queue_.append("")
        self.queue_cond_.notify()
        self.queue_cond_.release()


    def read_chunks(self):
        if not self.stream_.is_active():
            raise Exception("streaming is not started yet")

        self.quit_event_.clear()
        while not self.quit_event_.is_set():
            # read next chunk
            self.queue_cond_.acquire()
            if len(self.queue_) == 0:
                self.queue_cond_.wait()
            raw_frames = self.queue_.popleft()
            self.queue_cond_.release()

            # received the stop signal `""`
            if not raw_frames:
                break

            raw_frames = np.fromstring(raw_frames, dtype=self.np_format_)
            raw_frames = np.reshape(raw_frames, (self.chunk_size_, self.n_channels_))

            if self.sample_rate_in_ == self.sample_rate_out_:
                resampled_frames = raw_frames
            else:
                resampled_frames = self.__resample(raw_frames)

            resampled_frames = convert_type(resampled_frames, self.np_format_, self.format_out_)

            yield raw_frames, resampled_frames


    def get_channels(self):
        return self.n_channels_


    def get_sample_rate_in(self):
        return self.sample_rate_in_


    def get_sample_rate_out(self):
        return self.sample_rate_out_


    def get_chunk_interval(self):
        self.chunk_interval_


    def is_active(self):
        return self.stream_.is_active()


    def __callback(self, in_data, frame_count, time_info, status):
        self.queue_cond_.acquire()

        self.queue_.append(in_data)
        
        self.queue_cond_.notify()
        self.queue_cond_.release()

        # since we do NOT want to have any output, set the 
        # 1st return item to `None`
        return None, pyaudio.paContinue


    def __resample(self, raw_frames):
        resampled_frames = self.resampler_.process(raw_frames, self.ratio_).astype(self.np_format_)
        return resampled_frames


    def __find_device_index(self):
        device_idx = None

        for i in range( self.pyaudio_ins_.get_device_count() ):
            dev = self.pyaudio_ins_.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
            if dev['maxInputChannels'] == self.n_channels_:
                print('Use {}'.format(name))
                device_idx = i
                break

        if device_idx is None:
            raise Exception( 'can not find input device with %d channel(s)' % (self.n_channels_) )
        else:
            return device_idx


def test_mic_array():
    import signal
    import time

    from utils import write2wav

    mic = MicArray(
        sample_rate_in=44100,
        sample_rate_out=32000,
        n_channels=16,
        chunk_size=441,
        format_in="float32",
        format_out="int16"
    )

    # handle interrupt signal
    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print("Exit Signal (Ctrl + C) Received")
    signal.signal(signal.SIGINT, signal_handler)

    formed_sequence = None

    mic.start()
    print("Start recording...")

    for raw_frames, _ in mic.read_chunks():
        if is_quit.is_set():
            break

        if formed_sequence is None:
            formed_sequence = raw_frames
        else:
            formed_sequence = np.append(formed_sequence, raw_frames, axis=0)

    mic.stop()
    print("Stop recording")
    print(formed_sequence.shape)
    write2wav(formed_sequence, n_channels=mic.get_channels(), rate=mic.get_sample_rate_in())
    

if __name__ == '__main__':
    test_mic_array()