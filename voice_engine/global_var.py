import numpy as np
import pyaudio

################################################
# Define some global variable here
################################################

STR2PYAUDIO_FORMAT = {
    "int8" : pyaudio.paInt8,
    "int16" : pyaudio.paInt16,
    "int32" : pyaudio.paInt32,
    "float32" : pyaudio.paFloat32
}

STR2NUMPY_FORMAT = {
    "int8" : np.int8,
    "int16" : np.int16,
    "int32" : np.int32,
    "float32" : np.float32
}

DATA_FORMAT2STR = {
    "int8"              :   "int8",
    "int16"             :   "int16",
    "int32"             :   "int32",
    "float32"           :   "float32",

    pyaudio.paInt8      :   "int8",
    pyaudio.paInt16     :   "int16",
    pyaudio.paInt32     :   "int32",
    pyaudio.paFloat32   :   "float32",

    np.int8             :   "int8",
    np.int16            :   "int16",
    np.int32            :   "int32",
    np.float32          :   "float32",

    np.dtype('int8')    :   "int8",
    np.dtype('int16')   :   "int16",
    np.dtype('int32')   :   "int32",
    np.dtype('float32') :   "float32"
}