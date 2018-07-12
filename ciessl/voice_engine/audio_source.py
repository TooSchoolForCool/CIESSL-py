from abc import ABCMeta, abstractmethod

class AudioSource(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def read_chunks(self):
        pass

    @abstractmethod
    def get_sample_rate_in():
        pass

    @abstractmethod
    def get_sample_rate_out():
        pass

    @abstractmethod
    def get_channels():
        pass