import numpy as np
import json


class TraceTracker(object):
    def __init__(self, verbose=False):
        self.verbose_ = verbose
        self.trace_ = []


    def append(self, rank_score, target, origin):
        ranking_tuple = [(i + 1, rank_score[i]) for i in range(0, len(rank_score))]
        ranking_tuple.sort(key=lambda v : v[1], reverse=True)
        ranking = [t[0] for t in ranking_tuple]

        print(ranking)
        trace = []
        for room_id in ranking:
            trace.append(room_id)
            if room_id == target:
                break

        trail = {
            "trace" : trace,
            "target" : target,
            "origin" : origin
        }

        if self.verbose_:
            print("[TraceTracker]: Save trace {}".format(trail))

        self.trace_.append(trail)


    def dump(self, out_dir):
        with open(out_dir, 'w') as outfile:
            json.dump(self.trace_, outfile)

        print("[TraceTracker]: Save trace to json file {}".format(out_dir))