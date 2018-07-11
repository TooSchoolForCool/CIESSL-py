import collections
import threading
import math

import pyaudio
import numpy as np


# Define some global variables here
MAX_QUEUE_SIZE = 32

PYAUDIO_FORMAT = {
    "int8" : pyaudio.paInt8,
    "int16" : pyaudio.paInt16,
    "int32" : pyaudio.paInt32,
    "float32" : pyaudio.paFloat32
}

NUMPY_FORMAT = {
    "int8" : np.int8,
    "int16" : np.int16,
    "int32" : np.int32,
    "float32" : np.float32
}

class MicArray(object):
    
    def __init__(self, sample_rate=44100, n_channels=16, chunk_size=4096, format_in="int16"):
        self.pyaudio_ins_ = pyaudio.PyAudio()

        self.queue_ = collections.deque(maxlen=MAX_QUEUE_SIZE)

        self.quit_event_ = threading.Event()
        self.queue_cond_ = threading.Condition()
        
        self.n_channels_ = n_channels
        self.sample_rate_ = sample_rate
        self.chunk_size_ = chunk_size
        self.pyaudio_format_ = PYAUDIO_FORMAT[format_in]
        self.np_format_ = NUMPY_FORMAT[format_in]

        device_idx = self.__find_device_index()
        
        self.stream_ = self.pyaudio_ins_.open(
            input=True,
            start=False,
            format=self.pyaudio_format_,
            channels=self.n_channels_,
            rate=self.sample_rate_,
            frames_per_buffer=self.chunk_size_,
            stream_callback=self.__callback,
            input_device_index=device_idx,
        )

        print("[INFO] Microphone Array Init is done")


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
            self.queue_cond_.acquire()

            if len(self.queue_) == 0:
                self.queue_cond_.wait()

            raw_frames = self.queue_.popleft()

            self.queue_cond_.release()

            # received the stop signal `""`
            if not raw_frames:
                break

            formed_frame = np.fromstring(raw_frames, dtype=self.np_format_)
            yield formed_frame


    def get_channels(self):
        return self.n_channels_


    def get_sample_rate(self):
        return self.sample_rate_


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
        sample_rate=44100,
        n_channels=16,
        chunk_size=4096,
        format_in="int32"
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

    for formed_frames in mic.read_chunks():
        if is_quit.is_set():
            break

        if formed_sequence is None:
            formed_sequence = formed_frames
        else:
            formed_sequence = np.append(formed_sequence, formed_frames)

    mic.stop()
    print("Stop recording")

    write2wav(formed_sequence, n_channels=mic.get_channels(), rate=mic.get_sample_rate())
    

if __name__ == '__main__':
    test_mic_array()