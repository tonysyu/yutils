#!/usr/bin/env python
import argparse
import numpy as np
import motmot.FlyMovieFormat.FlyMovieFormat as fmf


parser = argparse.ArgumentParser()
parser.add_argument('fmf_file')
args = parser.parse_args()
mov = fmf.FlyMovie(args.fmf_file)
times = mov.get_all_timestamps()
monotonic = all(np.diff(times) > 0)
if monotonic:
    print "Time stamps are sequential"
else:
    print "WARNING: time stamps out of order in %s" % args.fmf_file
