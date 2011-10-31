#!/usr/bin/env python
import os
import argparse

import numpy as np
import motmot.FlyMovieFormat.FlyMovieFormat as FMF

import imgseq


Y4M_MAGIC = 'YUV4MPEG2'
ASPECT_RATIO = '1:1' # *pixel* aspect ratio


if 1:
    import signal
    # http://mail.python.org/pipermail/python-list/2004-June/268512.html
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# TODO: see if this blog post
# http://derrickpetzold.com/index.php/capturing-output-from-ffmpeg-python/
# contains informations about how to avoid the strange blocking problem that
# happens when attempting to pipe directly to ffmpeg through stdin.

def fmf2y4m(fmf, fps, out='fmf2vid_temp.y4m', rotate_180=False):

    if fmf.get_format() not in ['MONO8','RAW8']:
        msg = 'Only MONO8 and RAW8 formats are currently supported.'
        raise NotImplementedError(msg)

    width = fmf.get_width()//(fmf.get_bits_per_pixel()//8)
    height = fmf.get_height()


    y4m_opts = dict(y4mspec=Y4M_MAGIC, width=width, height=height, fps=fps,
                    aspect_ratio=ASPECT_RATIO)

    with open(out, 'w') as y4mfile:

        y4mfile.write('%(y4mspec)s W%(width)d H%(height)d F%(fps)d:1 '
                      'Ip A%(aspect_ratio)s Cmono\n' % y4m_opts)
        while 1:
            try:
                frame,timestamp = fmf.get_next_frame()
            except FMF.NoMoreFramesException, err:
                break

            y4mfile.write('FRAME\n')

            if rotate_180:
                frame = np.rot90(np.rot90(frame))

            for i in range(height):
                y4mfile.write(frame[i,:].tostring())
            y4mfile.flush()
    return out


def get_fps(fmf):
    times = fmf.get_all_timestamps()
    fmf.seek(0) # get_all_timestamps leaves the fmf file at end
    dt = np.median(np.diff(times))
    fps = int(round(1./dt))
    return fps


def fmf2mov(filename, fps=None, rotate_180=False, crf=10):
    fmf = FMF.FlyMovie(filename)
    if fps is None:
        fps = get_fps(fmf)
        print "No fps specified. Using detected frame rate of %i fps" % fps
    fpath, fext = os.path.splitext(filename)
    movfile = fpath + '.mov'
    y4mfile = fpath + '_temp.y4m'

    try:
        fmf2y4m(fmf, fps, rotate_180=rotate_180, out=y4mfile)

        output_opts = '-an -r %d -crf %d -vcodec h264' % (fps, crf)
        imgseq.convert_video(y4mfile, movfile, output_opts=output_opts)
    finally:
        # make sure we remove y4m file so it doesn't hog up disk space
        os.remove(y4mfile)


def main():
    usage = """%prog FILENAME [options]

Pipe the contents of an .fmf file to stdout in the yuv4mpegpipe
format. This allows an .fmf file to be converted to any format that
ffmpeg supports. For example, to convert the file x.fmf to x.avi using
the MPEG4 codec:

%prog x.fmf > x.y4m
ffmpeg -vcodec msmpeg4v2 -i x.y4m x.avi
"""

    parser = argparse.ArgumentParser(usage)

    parser.add_argument('filename', help='fmf file')
    parser.add_argument('--fps', type=int, default=None,
                        help=("Frames per second of output. Currently, this "
                              "must be an integer. If none, use median input "
                              "frame rate (rounded to the nearest integer)"))
    parser.add_argument('--rotate-180', action='store_true')
    parser.add_argument('--crf', type=int, default=10,
                        help="Constant rate factor: a lower crf produces "
                             "higher quality/bitrate movies. (default 10)")
    args = parser.parse_args()

    fmf2mov(filename=args.filename, fps=args.fps, rotate_180=args.rotate_180,
            crf=args.crf)

if __name__=='__main__':
    main()

