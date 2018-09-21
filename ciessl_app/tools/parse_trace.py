import os
import sys
import argparse
import random
import json

import numpy as np
import matplotlib.pyplot as plt

# import DataLoader
sys.path.append(os.path.dirname(__file__) + "/../")
from model.batch_loader import BatchLoader


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Training Voice AutoEncoder')

    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        required=True,
        help="Input data directory"
    )

    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        required=True,
        help="output directory"
    )
    
    args = parser.parse_args()

    return args


def plot_histogram(hit_cnt, mistakes_cnt, out_dir):
    idx = np.arange(2)
    x_title = ["Number of\nFirst hits", "Number of\nMistakes"]

    plt.bar(idx, height=[hit_cnt, mistakes_cnt], color=["blue", "red"], width=[0.8, 0.8])
    
    plt.xticks(idx, x_title, fontsize=15)
    plt.yticks(np.arange(0, 121, 20), fontsize=15)

    plt.ylabel("Count", fontsize=15)
    # plt.legend(loc="upper right")

    plt.savefig(out_dir, dpi=300)
    print("Save plot at {}".format(out_dir))



def parse_traces(voice_data_dir, out_path):    
    json_file=open(voice_data_dir).read()
    trace_history = json.loads(json_file)

    mistakes_cnt = 0
    first_hit = 0
    for i, trace in enumerate(trace_history):
        visits_cnt = len( trace["trace"] )

        mistakes_cnt += visits_cnt - 1
        first_hit += 1 if visits_cnt == 1 else 0

        out_dir = out_path + "/" + str(i) + ".png"
        plot_histogram(first_hit, mistakes_cnt, out_dir)
        # print(first_hit, mistakes_cnt)




def main():
    args = arg_parser()

    # check output directory if exists
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    parse_traces(args.data, args.out)

if __name__ == '__main__':
    main()