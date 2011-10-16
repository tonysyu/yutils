#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser

import motmot.FlyMovieFormat.FlyMovieFormat as FMF


def get_args():
    parser = ArgumentParser(description="Tool to convert fmf movie to images")
    parser.add_argument('input', type=str,
                        help='Input fmf filename')
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Filename for trimmed fmf file.')
    parser.add_argument('--start', type=int, default=0,
                        help='first frame to save')
    parser.add_argument('--stop', type=int, default=None,
                        help='last frame to save')
    parser.add_argument('--interval', type=int, default=1,
                        help='save every Nth frame')
    parser.add_argument('--postfix', default='_trimmed',
                        help='Postfix added to input file if output not given.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    assert args.interval >= 1
    if args.start == 0 and args.stop is None and args.interval == 1:
        print "No changes to fmf file. Specify 'start', 'stop', or 'interval'"
        sys.exit()

    base, ext = os.path.splitext(args.input)
    if not ext == '.fmf':
        print 'Input filename does not end in .fmf'
        sys.exit()

    if args.output is None:
        args.output = base + args.postfix + ext

    fly_movie = FMF.FlyMovie(args.input)

    n_frames = fly_movie.get_n_frames()
    if args.stop < 0 or args.stop >= n_frames:
        args.stop = n_frames - 1
    frames = range(args.start, args.stop+1, args.interval)

    fmf_saver = FMF.FlyMovieSaver(args.output)
    for count, frame_number in enumerate(frames):
        frame, timestamp = fly_movie.get_frame(frame_number)
        fmf_saver.add_frame(frame, timestamp)
    fmf_saver.close()

    print "%s images saved to %s" % (len(frames), args.output)


if __name__=='__main__':
    main()

