# Complex Indoor Environment Sound Source Localization (CIESSL)

[![Build Status](https://travis-ci.com/TooSchoolForCool/CIESSL-py.svg?token=pTSTf8Kr3MZ8RE9G5srX&branch=master)](https://travis-ci.com/TooSchoolForCool/CIESSL-py) ![python-2.7](https://img.shields.io/badge/python-2.7-blue.svg)

This repo implements a python application which performs sound source localization in complex indoor environment with the help of a microphone array and the knowledge of the room structure. To support processing audio sources (i.e., microphone array, wave file ...), a python package [voice_engine](#21-voice-engine) has been developed.

*Note: This repo is developed under **ubuntu 16.04** with **python 2.7**, other ubuntu distribution may also be compatible.*

## 1. Installation

Before installing this package, several system dependencies need to be installed

```bash
sudo apt-get install python-pyaudio libsamplerate0 portaudio19-dev
```

Then use following command to install this package

```bash
python setup.py bdist_wheel
sudo pip install dist/*.whl
```

## 2. Package Information

To support our sound source localization solution, several python packages have been implemented. Below is a brief introduction to every package we developed. 

### 2.1 Voice Engine

[voice_engine](voice_engine) is a python package that handling audio sources processing. It includes acquiring data from a microphone array, reading/writing data from/to a .wav file, resampling signal data, Voice Activity Detection (VAD) and Short Time Fourier Transform (STFT).

## 3. CIESSL

### 3.1 Voice Signal Processing

