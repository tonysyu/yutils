#!/usr/bin/env python
import os
import sys
import warnings
from argparse import ArgumentParser
import Image

from motmot.FlyMovieFormat import FlyMovieFormat
import motmot.imops.imops as imops


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('fmf_file', type=str)
    parser.add_argument('--start', type=int, default=0,
                        help='first frame to save')
    parser.add_argument('--stop', type=int, default=-1,
                        help='last frame to save')
    parser.add_argument('--interval', type=int, default=1,
                        help='save every Nth frame')
    parser.add_argument('--extension', type=str, default='bmp',
                        help='image extension (default: bmp)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='directory to save images (default: same as fmf)')
    parser.add_argument('--progress', action='store_true', default=False,
                        help='show progress bar')
    parser.add_argument('--prefix', default=None, type=str,
                        help='prefix for image filenames')
    return parser


def get_progressbar(n_frames):
    import progressbar
    widgets = ['fmf2img ', progressbar.Percentage(), ' ',
               progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=n_frames)
    return pbar


class DummyProgressBar(object):
    def __init__(self): pass
    def start(self): pass
    def update(self, *args): pass
    def finish(self): pass


def rgb_convert(fmf_format, frame):
    save_frame = imops.to_rgb8(fmf_format, frame)
    h, w = save_frame.shape[:2]
    im = Image.fromstring('RGB', (w,h), save_frame.tostring())
    return im


def mono_convert(fmf_format, frame):
    save_frame = imops.to_mono8(fmf_format, frame)
    h, w = save_frame.shape[:2]
    im = Image.fromstring('L', (w,h), save_frame.tostring())
    return im


def is_mono(fmf_format):
    if (fmf_format in ['RGB8','ARGB8','YUV411','YUV422'] or
        fmf_format.startswith('MONO8:') or fmf_format.startswith('MONO32f:')):
        return False
    else:
        if fmf_format not in ['MONO8','MONO16']:
            msg = 'converting unknown fmf format %s to mono' % fmf_format
            warnings.warn(msg)
        return True


def main():
    parser = get_parser()
    args = parser.parse_args()

    filename = args.fmf_file
    imgformat = args.extension
    startframe = args.start
    endframe = args.stop
    interval = args.interval
    assert interval >= 1

    base,ext = os.path.splitext(filename)
    if ext != '.fmf':
        print 'fmf_filename does not end in .fmf'
        sys.exit()

    path,base = os.path.split(base)
    if args.prefix is not None:
        base = args.prefix

    if args.outdir is None:
        outdir = path
    else:
        outdir = args.outdir

    fly_movie = FlyMovieFormat.FlyMovie(filename)
    fmf_format = fly_movie.get_format()
    n_frames = fly_movie.get_n_frames()
    if endframe < 0 or endframe >= n_frames:
        endframe = n_frames - 1

    if is_mono(fmf_format):
        convert = mono_convert
    else:
        convert = rgb_convert

    fly_movie.seek(startframe)
    frames = range(startframe, endframe+1, interval)
    n_frames = len(frames)

    if args.progress:
        pbar = get_progressbar(n_frames)
    else:
        pbar = DummyProgressBar()
    pbar.start()

    for count,frame_number in enumerate(frames):
        pbar.update(count)

        basename = os.path.join(outdir, base)
        f = '%s_%08d.%s'%(basename, frame_number, imgformat)

        frame, timestamp = fly_movie.get_frame(frame_number)
        im = convert(fmf_format, frame)
        im.save(f)

    pbar.finish()
    print "%s images saved to %s" % (frame_number + 1, outdir)


if __name__=='__main__':
    main()

