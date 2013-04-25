#!/usr/bin/env python
import argparse
import numpy as np
import motmot.FlyMovieFormat.FlyMovieFormat as fmf


parser = argparse.ArgumentParser()
parser.add_argument('fmf_file')
args = parser.parse_args()
if args.fmf_file.endswith('.fmf'):
    mov = fmf.FlyMovie(args.fmf_file)
    times = mov.get_all_timestamps()
elif args.fmf_file.endswith('.npy'):
    times = np.load(args.fmf_file)
else:
    print "Unsupported file type:", args.fmf_file
dts = np.diff(times)
monotonic = all(dts > 0)
if monotonic:
    print "Time stamps are sequential"
    print "Median dt = ", np.median(dts)
    print "Max difference in dt = ", np.max(np.diff(dts))
else:
    print "WARNING: time stamps out of order in %s" % args.fmf_file

