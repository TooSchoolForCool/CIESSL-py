import os
import argparse

import numpy as np

from voice_engine.wave_source import WaveSource


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Waveform to pickle')

    parser.add_argument(
        "--data_in",
        dest="data_in",
        type=str,
        required=True,
        help="Input wave data directory"
    )
    parser.add_argument(
        "--data_out",
        dest="data_out",
        type=str,
        required=True,
        help="Output data directory"
    )
    
    args = parser.parse_args()

    return args


def wav2pickle(file_path, output_dir):
    ws = WaveSource(
        file_dir=file_path,
    )

    frames = None
    ws.start()
    for raw_frames, resampled_frames in ws.read_chunks():
        if frames is None:
            frames = raw_frames
        else:
            frames = np.append(frames, raw_frames, axis=0)
    ws.stop()

    file_name = file_path.split('.')[0].split('/')[-1]
    output_file_path = output_dir + '/' + file_name + ".pickle"

    print(frames.shape)

    frames.dump(output_file_path)
    print("[INFO] Active voice is saved at: %s" % (output_file_path))


def main():
    args = arg_parser()

    # load all files' directories in voice_data_dir
    voice_dataset = args.data_in
    file_dirs = []
    
    # check output directory if exists, if not then create one
    if not os.path.exists(args.data_out):
        os.makedirs(args.data_out)

    # acquire all input data files
    for file in os.listdir(voice_dataset):
        if file.endswith(".wav"):
            file_dirs.append(os.path.join(voice_dataset, file))

    # trim active voice in each file
    for file in file_dirs:
        wav2pickle(file, args.data_out)


if __name__ == '__main__':
    main()