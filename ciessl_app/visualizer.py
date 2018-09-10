import sys
import os
import argparse

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Result Visulizer')

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

    plot_type = ["acc_variance"]
    parser.add_argument(
        "--plot",
        dest="plot",
        type=str,
        required=True,
        help="choose what type of encoder to train: %r" % plot_type
    )
    
    args = parser.parse_args()

    # check validation
    if args.plot not in plot_type:
        print("[ERROR] only support plot type {}".format(plot_type))
        raise

    return args


def load_file_path(target_dir, ext):
    file_dirs = []

    for file in os.listdir(target_dir):
        if file.endswith("." + ext):
            file_dirs.append(os.path.join(target_dir, file))

    return file_dirs


def plot_acc_with_variance(data_path, output):
    file_dirs = load_file_path(data_path, "csv")

    """
    In order to use seaborn we need to convert our results into pandas DataFrame, as following

                      Event  Number of Samples  Succese Rate
    0     Explore 1 time(s)                  1      0.000000
    1     Explore 2 time(s)                  1      1.000000
    2     Explore 1 time(s)                  2      0.000000
    3     Explore 2 time(s)                  2      0.500000
    4     Explore 1 time(s)                  3      0.000000
    5     Explore 2 time(s)                  3      0.333333
    ...
    """
    data_dict = {
        "Number of Samples" : [],
        "Succese Rate" : [],
        "Event" : []
    }
    for file in file_dirs:
        data = np.genfromtxt(file, delimiter=',')
        event_list = ["Explore %d time(s)" % (i) for i in range(1, data.shape[1] + 1)]

        for i, row in enumerate(data):
            for e, item in enumerate(row[:-1]):  
                data_dict["Number of Samples"].append(i + 1)
                data_dict["Event"].append(event_list[e])
                data_dict["Succese Rate"].append(item)
    df = pd.DataFrame(data_dict)
    
    # plot accuracy with erroebands
    sns.set(style="darkgrid")
    sns.lineplot(x="Number of Samples", y="Succese Rate", hue="Event", data=df)
    plt.show()


def main():
    args = arg_parser()

    if args.plot == "acc_variance":
        plot_acc_with_variance(args.data, args.out)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()