import numpy as np


class Evaluator(object):
    def __init__(self, n_rooms):
        self.n_rooms_ = n_rooms

        # scoreboard_[i] indicates the total times that find the 
        # target within `i+1` trials
        self.scoreboard_ = [0 for _ in range(n_rooms)]

        # number of experiments
        self.total_exp_ = 0


    def evaluate(self, y, predicted_y):
        assert(len(y) == self.n_rooms_)
        assert(len(predicted_y) == self.n_rooms_)

        # room index start from 1
        target_room = self.__find_target_room(y)
        room_ranking = self.__calc_room_ranking(predicted_y)

        for rank, room_idx in enumerate(room_ranking):
            if room_idx == target_room:
                for i in range(rank, self.n_rooms_):
                    self.scoreboard_[i] += 1

        self.total_exp_ += 1


    def get_eval_result(self):
        """
        Calculate the probability that find the target within `i` trials
        """
        return [1.0 * score / self.total_exp_ for score in self.scoreboard_]


    def __find_target_room(self, y):
        for i in range(0, self.n_rooms_):
            if y[i] == 1:
                return i + 1


    def __calc_room_ranking(self, predicted_y):
        """
        Calculate the room ranking given the predicted result

        Args:
            predicted_y ( np.ndarray (n_samples,) ): predicted label

        Returns:
            ranking (a list of index): room ranking sequence,
                ranking[0] indicates the highest priority
        """
        # calculate normalized probability of each room
        total = np.sum(predicted_y)
        predicted_y /= total

        ranking_tuple = [(i + 1, predicted_y[i]) for i in range(0, len(predicted_y))]
        ranking_tuple.sort(key=lambda v : v[1], reverse=True)

        ranking = [t[0] for t in ranking_tuple]

        return ranking


def test_evaluator():
    evaluator = Evaluator(n_rooms=4)

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


if __name__ == '__main__':
    test_evaluator()

