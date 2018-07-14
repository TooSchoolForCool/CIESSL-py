# Complex Indoor Environment Sound Source Localization (CIESSL)

[![Build Status](https://travis-ci.com/TooSchoolForCool/CIESSL-py.svg?token=pTSTf8Kr3MZ8RE9G5srX&branch=master)](https://travis-ci.com/TooSchoolForCool/CIESSL-py) ![python-2.7](https://img.shields.io/badge/python-2.7-blue.svg)



This repo implements a python application which performs sound source localization in complex indoor environment with the help of a microphone array and the knowledge of the room structure. The application is developed based on a python package named `ciessl` in which a [voice_engine](ciessl/voice_engine) is implemented.



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



## 2. Introduction

### 2.1 Voice Engine



### 2.2 CIESSL

