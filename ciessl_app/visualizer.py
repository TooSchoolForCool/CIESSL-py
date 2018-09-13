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
        nargs='+',
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

    parser.add_argument(
        "--name_tag",
        dest="name_tag",
        type=str,
        nargs='+',
        help="Name tag for each dataset"
    )

    plot_type = ["acc_variance", "err_hist", "cmp_acc"]
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

    if "cmp" in args.plot and not args.name_tag:
        print("[ERROR] must specify reference data directory {}".format(plot_type))
        raise

    return args


def load_file_path(target_dir, ext):
    file_dirs = []

    for file in os.listdir(target_dir):
        if file.endswith("." + ext):
            file_dirs.append(os.path.join(target_dir, file))

    return file_dirs


def plot_err_histogram(data_path, output):
    data_path = data_path[0]
    file_dirs = load_file_path(data_path, "csv")

    # get data dimension
    tmp = np.genfromtxt(file_dirs[0], delimiter=',')
    n_samples = len(tmp)
    n_bins = 10

    # create title for x-axis
    # ["1 ~ 20", "21 ~ 40", ...]
    x_title = []
    for i in range(n_bins, n_samples + 1, n_bins):
        x_title.append("%d ~ %d" % (i-n_bins+1, i))
    if n_samples % n_bins != 0:
        x_title.append("%d ~ %d" % (n_samples - (n_samples%n_bins) + 1, n_samples))

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
        "Error Counts" : []
    }
    for file in file_dirs:
        data = np.genfromtxt(file, delimiter=',')
        for i in range(n_bins, n_samples + 1, n_bins):
            cnt = data[i-1] - data[i - n_bins]
            data_dict["Error Counts"].append(cnt)
            data_dict["Number of Samples"].append(x_title[i / n_bins - 1])
        if n_samples % n_bins != 0:
            cnt = data[-1] - data[-(len(data)%n_bins)]
            data_dict["Error Counts"].append(cnt)
            data_dict["Number of Samples"].append(x_title[-1])

    df = pd.DataFrame(data_dict)
    
    # plot error histogram
    sns.set(style="whitegrid")
    sns.catplot(x="Number of Samples", y="Error Counts", kind="bar", data=df, color="b")
    plt.show()


def plot_acc_cmp(data_path, output, name_tag):
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid', {"axes.spines.right": True, "axes.spines.top": True})

    # data_dict = {
    #     "n_samples" : [],
    #     "acc" : [],
    #     "Model" : [],
    #     "Event" : []
    # }
    # event_list = ["Explore %d time(s)" % (i) for i in range(1, 3 + 1)]

    # for data_dir, tag in zip(data_path, name_tag):
    #     file_dirs = load_file_path(data_dir, "csv")
    #     for file in file_dirs:
    #         data = np.genfromtxt(file, delimiter=',')
    #         for i, row in enumerate(data):
    #             for j, item in enumerate(row[:2]):
    #                 data_dict["n_samples"].append(i + 1)
    #                 data_dict["Model"].append(tag)
    #                 data_dict["acc"].append(item)
    #                 data_dict["Event"].append(event_list[j])

    # df = pd.DataFrame(data_dict)

    # sns.relplot(x="n_samples", y="acc", hue="Model", row="Event", kind="line", data=df)
    # plt.ylabel("Success Rate [%]")
    # plt.xlabel("Number of Samples")
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 114)

    data_dict = {
        "n_samples" : [],
        "acc" : [],
        "Model" : [],
    }

    for data_dir, tag in zip(data_path, name_tag):
        file_dirs = load_file_path(data_dir, "csv")
        for file in file_dirs:
            data = np.genfromtxt(file, delimiter=',')
            for i, row in enumerate(data):
                data_dict["n_samples"].append(i + 1)
                data_dict["Model"].append(tag)
                data_dict["acc"].append(row[0])

    df = pd.DataFrame(data_dict)

    plt.subplot(2, 1, 1)
    sns.lineplot(x="n_samples", y="acc", hue="Model", data=df)
    plt.ylabel("Success Rate [%]")
    plt.xlabel("")
    plt.ylim(0, 1.0)
    plt.xlim(0, 114)

    data_dict = {
        "n_samples" : [],
        "acc" : [],
        "Model" : [],
    }

    for data_dir, tag in zip(data_path, name_tag):
        file_dirs = load_file_path(data_dir, "csv")
        for file in file_dirs:
            data = np.genfromtxt(file, delimiter=',')
            for i, row in enumerate(data):
                data_dict["n_samples"].append(i + 1)
                data_dict["Model"].append(tag)
                data_dict["acc"].append(row[1])

    df = pd.DataFrame(data_dict)

    plt.subplot(2, 1, 2)
    sns.lineplot(x="n_samples", y="acc", hue="Model", data=df)
    plt.ylabel("Success Rate [%]")
    plt.xlabel("Number of Samples")
    plt.ylim(0, 1.0)
    plt.xlim(0, 114)
    
    # plt.rcParams["axes.spines.right"] = True
    # plt.rcParams["axes.spines.top"] = True
    plt.show()


def plot_acc_with_variance(data_path, output):
    data_path = data_path[0]
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
    elif args.plot == "err_hist":
        plot_err_histogram(args.data, args.out)
    elif args.plot == "cmp_acc":
        plot_acc_cmp(args.data, args.out, args.name_tag)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()
