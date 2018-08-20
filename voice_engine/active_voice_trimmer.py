import threading
import Queue
from collections import deque

import numpy as np

from vad import VAD

class ActiveVoiceTrimmer(object):
    def __init__(self, mode, audio_source, prefix_size=None):
        self.vad_ = VAD(mode)
        self.audio_source_ = audio_source
        self.samplerate_out_ = self.audio_source_.get_sample_rate_out()
        self.n_channels_ = self.audio_source_.get_channels()

        self.vad_thread_ = None
        self.quit_event_ = threading.Event()

        self.queue_ = Queue.Queue()

        # save a pre-defined length of audio_chunks, and append it to detected active voice
        if prefix_size is None:
            prefix_size = int(0.08 / audio_source.get_chunk_interval())

        self.prefix_buffer_ = deque(maxlen=prefix_size)

        self.active_chunks_ = None
        

    def start(self):
        # Do NOT start the trimmer when the trimmer is already started
        if self.vad_thread_ is not None:
            return

        if self.queue_.mutex:
            self.queue_.queue.clear()

        self.prefix_buffer_.clear()

        self.quit_event_.clear()
        self.audio_source_.start()

        self.vad_thread_ = threading.Thread(target=self.__voice_detect)
        self.vad_thread_.start()
        

    def stop(self):
        # Do NOT shut down the trimmer when the trimmer is actually shutdown
        if self.vad_thread_ is None:
            return

        self.quit_event_.set()
        self.audio_source_.stop()

        self.vad_thread_.join()
        self.vad_thread_ = None

        # let read_active_chunks out of blocking when 
        # there is no input into queue any more
        self.queue_.put("")
        

    def read_active_chunks(self):
        if self.vad_thread_ is None:
            raise Exception("ActiveVoiceTrimmer is not started yet")

        while not self.quit_event_.is_set():
            frames = self.queue_.get()

            # received 'stop' signal
            if frames is "":
                break

            yield frames


    def __voice_detect(self):
        for raw_frames, resampled_frames in self.audio_source_.read_chunks():
            if self.quit_event_.is_set():
                break

            if self.vad_.is_speech(resampled_frames, self.samplerate_out_, self.n_channels_):
                if self.active_chunks_ is None:
                    self.active_chunks_ = raw_frames
                else:
                    self.active_chunks_ = np.append(self.active_chunks_, raw_frames, axis=0)
            else:
                # save active_chunks and its prefix to data queue
                if self.active_chunks_ is not None:
                    if self.prefix_buffer_: # if prefix buffer is not empty
                        data = self.prefix_buffer_.popleft()

                        while self.prefix_buffer_:
                            data = np.append(data, self.prefix_buffer_.popleft(), axis=0)

                        # concatinate active voice to the prefix data
                        data = np.append(data, self.active_chunks_, axis=0)
                    else: # if prefix buffer is empty
                        data = self.active_chunks_

                    # push trimmed active voice into queue
                    self.queue_.put(data)

                    self.active_chunks_ = None
                    self.prefix_buffer_.clear()

                self.prefix_buffer_.append(raw_frames)

        # when audio source stoped, then send a 'stop' signal to read_active_chunks
        self.queue_.put("")


def test_avt():
    import signal
    import time

    from utils import write2wav
    from mic_array import MicArray

    sample_rate = 44100
    sample_rate_out = 32000
    vad_time_interval = 10
    chunk_size = sample_rate * vad_time_interval / 1000

    mic = MicArray(
        sample_rate_in=sample_rate,
        sample_rate_out=sample_rate_out,
        n_channels=16,
        chunk_size=chunk_size,
        format_in="float32",
        format_out="int16"
    )

    avt = ActiveVoiceTrimmer(mode=2, audio_source=mic)

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

        print( "active voice size", active_frames.shape )
    
    avt.stop()
    print("Stop listening")


if __name__ == '__main__':
    test_avt()