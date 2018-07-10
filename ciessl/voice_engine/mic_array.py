import Queue
import threading
import math

import pyaudio
import numpy as np


class MicArray(object):
    
    def __init__(self, sample_rate=44100, n_channels=16, chunk_size=4096):
        self.pyaudio_ins_ = pyaudio.PyAudio()

        self.queue_ = Queue.Queue()

        self.quit_event_ = threading.Event()
        
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
        with self.queue_.mutex:
            self.queue_.queue.clear()

        self.stream_.start_stream()


    def stop(self):
        self.quit_event_.set()
        self.stream_.stop_stream()


    def read_chunks(self):
        self.quit_event_.clear()
        while not self.quit_event_.is_set():
            frames = self.queue_.get()

            # Cannot acquire signal from device
            if not frames:
                break

            frames = np.fromstring(frames, dtype='int16')
            yield frames


    def __callback(self, in_data, frame_count, time_info, status):
        self.queue_.put(in_data)

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