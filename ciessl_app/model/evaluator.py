import os
import sys

import numpy as np
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, n_rooms, verbose=False):
        self.n_rooms_ = n_rooms

        # scoreboard_[i] indicates the total times that find the 
        # target within `i+1` trials
        self.scoreboard_ = [0 for _ in range(n_rooms)]

        # verbose: print out ranking information
        self.verbose_ = verbose

        # scoreboard accuracy history (scoreboard / total_exp)
        # A sequence of accuracy, each time when a new sample come in, the 
        # evaluator will generate a new accuracy record
        self.acc_history_ = []

        # A sequence of error history
        # If the rebot explore the room 3 times, then there are 2 error tries
        self.error_history_ = []

        # number of experiments
        self.total_exp_ = 0


    def evaluate(self, y, predicted_y):
        try:
            assert(len(y) == len(predicted_y))
        except:
            print("y: {}".format(y))
            print("predicted_y: {}".format(predicted_y))
            assert(len(y) == len(predicted_y))

        for target, predicted_prob in zip(y, predicted_y):
            try:
                assert(len(predicted_prob) == self.n_rooms_)
            except:
                print("predicted_prob: {}".format(predicted_prob))
                print("n_rooms: {}".format(self.n_rooms_))
                assert(len(predicted_prob) == self.n_rooms_)

            ranking = self.calc_room_ranking_(predicted_prob)
            for rank, room_idx in enumerate(ranking):
                if room_idx == target:
                    for i in range(rank, self.n_rooms_):
                        self.scoreboard_[i] += 1
        
        self.total_exp_ += len(y)
        self.acc_history_.append([1.0 * score / self.total_exp_ for score in self.scoreboard_])
        self.error_history_.append(self.n_rooms_ * self.total_exp_ - sum(self.scoreboard_))

        if self.verbose_:
            self.print_log_(y, predicted_y)


    def get_eval_result(self):
        """
        Calculate the probability that find the target within `i` trials
        """
        return self.acc_history_[-1]


    def plot_acc_history(self, n_lines=3):
        palette = "bgrcmykw"

        data = np.asarray(self.acc_history_)

        plt.title("Accuracy Trendline")
        plt.xlabel("Number of Sound Events")
        plt.ylabel("Accuracy")

        x = [xi for xi in range(1, data.shape[0] + 1)]
        for i in range(0, n_lines):  
            y = data[:, i]
            color = palette[i % len(palette)]

            plt.plot(x, y, color, label="Goal within " + str(i + 1) + " trails")

        y_axis = [0.2 * i for i in range(0, 6)]
        plt.yticks(y_axis, y_axis, rotation=0)
        # plt.xticks(x, x, rotation=0)
        plt.legend(loc=0)
        plt.grid()
        plt.show()


    def plot_error_bar(self, n_bins=10):
        errors = self.error_history_

        # calculate number of errors for every n_bins samples
        accum = []
        x_title = []
        for i in range(n_bins, len(errors), n_bins):
            accum.append( errors[i] - errors[i - n_bins] )
            x_title.append("%d ~ %d" % (i-n_bins+1, i))
        if len(errors) % n_bins != 0:
            accum.append( errors[-1] - errors[-(len(errors)%n_bins)] )
            x_title.append("%d ~ %d" % (len(errors) - (len(errors)%n_bins) + 1, len(errors)))

        idx = np.arange( len(accum) )
        plt.bar(idx, height=accum)
        plt.xticks(idx, x_title)

        plt.ylabel("Number of Errors")
        plt.title("Number of Errors Trendline")
        plt.show()


    def save_history(self, out_dir, file_prefix, type="csv"):
        if type == "csv":
            self.dump2csv(out_dir, file_prefix)


    def dump2csv(self, out_dir, file_prefix):
        if not os.path.exists(out_dir + "/acc"):
            os.makedirs(out_dir+"/acc")

        if not os.path.exists(out_dir + "/error"):
            os.makedirs(out_dir+"/error")

        # accuracy
        file_out = out_dir + "/acc/" + file_prefix + "_acc.csv"
        np.savetxt(file_out, self.acc_history_, delimiter=",")
        print("[Evaluator]: Save Accuracy history to file {}".format(file_out))

        # errors counts
        file_out = out_dir + "/error/" + file_prefix + "_err.csv"
        np.savetxt(file_out, self.error_history_, delimiter=",")
        print("[Evaluator]: Save Error history to file {}".format(file_out))


    def find_target_room_(self, y):
        for i in range(0, self.n_rooms_):
            if y[i] == 1:
                return i + 1


    def calc_room_ranking_(self, predicted_y):
        """
        Calculate the room ranking given the predicted result

        Args:
            predicted_y ( np.ndarray (n_samples,) ): predicted label

        Returns:
            ranking (a list of index): room ranking sequence,
                ranking[0] indicates the highest priority
        """
        # calculate normalized probability of each room
        # total = np.sum(predicted_y)
        # predicted_y /= total

        ranking_tuple = [(i + 1, predicted_y[i]) for i in range(0, len(predicted_y))]
        ranking_tuple.sort(key=lambda v : v[1], reverse=True)

        ranking = [t[0] for t in ranking_tuple]

        return ranking


    def print_log_(self, y, predicted_y):
        print("-" * 80)
        print("| * Sample %d *" % self.total_exp_)
        print("| y:\t%r" % (y))
        print("| pred:\t {}".format(predicted_y))
        print("| acc: %r" % (self.get_eval_result()))
        print("| errors: {}".format(self.error_history_[-1]))
        print("-" * 80)


def test_evaluator():
    evaluator = Evaluator(n_rooms=4)

    for _ in range(0, 10):
        y = np.array([0, 0, 1, 0])
        py = np.array([0.3, 0.8, 0.6, 0.4])
        evaluator.evaluate(y, py)
        print(evaluator.get_eval_result())

        y = np.array([0, 1, 0, 0])
        py = np.array([0.3, 0.8, 0.6, 0.4])
        evaluator.evaluate(y, py)
        print(evaluator.get_eval_result())

        y = np.array([0, 0, 0, 1])
        py = np.array([0.3, 0.8, 0.6, 0.4])
        evaluator.evaluate(y, py)
        print(evaluator.get_eval_result())

        y = np.array([1, 0, 0, 0])
        py = np.array([0.3, 0.8, 0.6, 0.4])
        evaluator.evaluate(y, py)
        print(evaluator.get_eval_result())

        y = np.array([0, 0, 1, 0])
        py = np.array([0.3, 0.8, 0.6, 0.4])
        evaluator.evaluate(y, py)
        print(evaluator.get_eval_result())

    evaluator.plot_acc_history(1)


if __name__ == '__main__':
    test_evaluator()

