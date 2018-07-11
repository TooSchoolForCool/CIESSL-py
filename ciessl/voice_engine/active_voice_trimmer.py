import threading
import Queue

import numpy as np

from vad import VAD

class ActiveVoiceTrimmer(object):
    def __init__(self, mode, audio_source):
        self.vad_ = VAD(mode)
        self.audio_source_ = audio_source

        self.vad_thread_ = None
        self.quit_event_ = threading.Event()

        self.queue_ = Queue.Queue()

        self.active_chunks_ = []
        

    def start(self):
        # Do NOT start the trimmer when the trimmer is already started
        if self.vad_thread_ is not None:
            return

        if self.queue_.mutex:
            self.queue_.queue.clear()

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
            chunks = self.queue_.get()

            # received stop signal
            if not chunks:
                break

            frames = np.append(chunks[0], chunks[1:])

            yield frames


    def __voice_detect(self):
        for chunk in self.audio_source_.read_chunks():
            if self.quit_event_.is_set():
                break

            if self.vad_.is_speech(chunk, self.audio_source_.get_sample_rate(), self.audio_source_.get_channels()):
                self.active_chunks_.append(chunk)
            else:
                if self.active_chunks_:
                    self.queue_.put(self.active_chunks_)
                    self.active_chunks_ = []


def test_avt():
    import signal
    import time

    from utils import write2wav
    from mic_array import MicArray

    sample_rate = 48000
    vad_time_interval = 10
    chunk_size = sample_rate * vad_time_interval / 1000

    mic = MicArray(
        sample_rate=sample_rate,
        n_channels=16,
        chunk_size=chunk_size,
        format_in="int16"
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