#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser
import numpy as np

import motmot.FlyMovieFormat.FlyMovieFormat as FMF


def get_parser():
    parser = ArgumentParser(description="Tool to convert fmf movie to images")
    parser.add_argument('input', type=str,
                        help='Input fmf filename')
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Filename for trimmed fmf file.')
    parser.add_argument('--postfix', default='_fixed',
                        help='Postfix added to input file if output not given.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    base, ext = os.path.splitext(args.input)
    if not ext == '.fmf':
        print 'Input filename does not end in .fmf'
        sys.exit()

    if args.output is None:
        args.output = base + args.postfix + ext

    fly_movie = FMF.FlyMovie(args.input)
    times = fly_movie.get_all_timestamps()
    monotonic = all(np.diff(times) > 0)
    if monotonic:
        print "Timestamps are sequential. Quitting..."
    fmf_saver = FMF.FlyMovieSaver(args.output)

    # cast frame numbers to int64, otherwise more than 4100 frames from
    # a mega pixel camera will overflow int32.
    fixed_frame_order = np.int64(np.argsort(times))
    for frame_number in fixed_frame_order:
        frame, timestamp = fly_movie.get_frame(frame_number)
        fmf_saver.add_frame(frame, timestamp)
    fmf_saver.close()


if __name__=='__main__':
    main()

