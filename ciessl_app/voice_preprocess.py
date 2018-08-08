import os
import argparse

from voice_engine.utils import write2wav
from voice_engine.wave_source import WaveSource
from voice_engine.active_voice_trimmer import ActiveVoiceTrimmer

def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='voice preprocessing')

    parser.add_argument(
        "--data_in",
        dest="data_in",
        type=str,
        required=True,
        help="Input voice data directory"
    )
    parser.add_argument(
        "--data_out",
        dest="data_out",
        type=str,
        required=True,
        help="Output data directory"
    )
    parser.add_argument(
        "--chunk_interval",
        dest="chunk_interval",
        type=int,
        default=20,
        help="Chunk time interval for VAD"
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        type=int,
        default=3,
        help="VAD detection mode [1 - 3]"
    )
    
    args = parser.parse_args()

    return args


def trim_active_voice(voice_file, outpath, sample_rate_out=32000,
    chunk_time_interval=20, format_out="int16", mode=3):
    """
    Trim active voice from a voice file

    Args:
        voice_file (string): voice file directory
        outpath (string): output file directory
    """
    ws = WaveSource(
        file_dir=voice_file,
        sample_rate_out=sample_rate_out,
        chunk_time_interval=chunk_time_interval,
        format_out=format_out
    )

    voice_cnt = 0
    file_name = voice_file.split('.')[0].split('/')[-1]
    sample_rate = ws.get_sample_rate_in()

    avt = ActiveVoiceTrimmer(mode=mode, audio_source=ws)
    avt.start()

    print("%s sample rate in: %d" % (file_name, sample_rate))

    for active_frames in avt.read_active_chunks():
        output_path = outpath + '/' + str(sample_rate) + '-' + file_name \
            + '-' + str(voice_cnt) + ".pickle"
        active_frames.dump(output_path)
        # write2wav(active_frames, ws.get_channels(), ws.get_sample_rate_in(), output=output_path)

        print("[INFO] Active voice is saved at: %s" % (output_path))
        voice_cnt += 1
    
    avt.stop()


def main():
    args = arg_parser()

    # load all files' directories in voice_data_dir
    voice_dataset = args.data_in
    voice_file_dirs = []
    
    # check output directory if exists
    if not os.path.exists(args.data_out):
        os.makedirs(args.data_out)

    # acquire all input data files
    for file in os.listdir(voice_dataset):
        voice_file_dirs.append(os.path.join(voice_dataset, file))

    # trim active voice in each file
    for voice_file in voice_file_dirs:
        trim_active_voice(voice_file, args.data_out, sample_rate_out=48000, 
            chunk_time_interval=args.chunk_interval, format_out="int16", mode=args.mode)


if __name__ == '__main__':
    main()