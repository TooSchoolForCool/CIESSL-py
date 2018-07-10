import collections
import threading
import math

import pyaudio
import numpy as np


# Define some global variables here
MAX_QUEUE_SIZE = 32


class MicArray(object):
    
    def __init__(self, sample_rate=44100, n_channels=16, chunk_size=4096):
        self.pyaudio_ins_ = pyaudio.PyAudio()

        self.queue_ = collections.deque(maxlen=MAX_QUEUE_SIZE)

        self.quit_event_ = threading.Event()
        self.queue_cond_ = threading.Condition()
        
        self.n_channels_ = n_channels
        self.sample_rate_ = sample_rate
        self.chunk_size_ = chunk_size

        device_idx = self.__find_device_index()
        
        self.stream_ = self.pyaudio_ins_.open(
            input=True,
            start=False,
            format=pyaudio.paInt16,
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


    def read_chunks(self):
        if not self.stream_.is_active():
            raise Exception("streaming is not started yet")

        self.quit_event_.clear()
        while not self.quit_event_.is_set():
            self.queue_cond_.acquire()

            if len(self.queue_) == 0:
                self.queue_cond_.wait()

            frames = self.queue_.popleft()

            self.queue_cond_.release()

            # Cannot acquire signal from device
            if not frames:
                break

            frames = np.fromstring(frames, dtype='int16')
            yield frames


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


def main():
    import signal
    import time

    mic = MicArray(
        sample_rate=44100,
        n_channels=16,
        chunk_size=4096
    )

    # handle interrupt signal
    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print("Quit Signal Received")
    signal.signal(signal.SIGINT, signal_handler)

    print("1st Round")
    mic.start()
    for chunk in mic.read_chunks():
        if is_quit.is_set():
            break
        print(len(chunk))
    mic.stop()

    print("2nd Round")
    is_quit.clear()
    mic.start()
    for chunk in mic.read_chunks():
        if is_quit.is_set():
            break
        print(len(chunk))
    mic.stop()


if __name__ == '__main__':
    main()