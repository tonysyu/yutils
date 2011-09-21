#!/usr/bin/env python
import os, sys
from motmot.FlyMovieFormat import FlyMovieFormat
import Image
from argparse import ArgumentParser
import motmot.imops.imops as imops
import warnings

def main():

    parser = ArgumentParser()
    parser.add_argument('fmf_file', type=str)
    parser.add_argument('--start',type=int,default=0,help='first frame to save')
    parser.add_argument('--stop',type=int,default=-1,help='last frame to save')
    parser.add_argument('--interval',type=int,default=1,help='save every Nth frame')
    parser.add_argument('--extension',type=str,default='bmp',
                      help='image extension (default: bmp)')
    parser.add_argument('--outdir',type=str,default=None,
                      help='directory to save images (default: same as fmf)')
    parser.add_argument('--progress',action='store_true',default=False,
                      help='show progress bar')
    parser.add_argument('--prefix',default=None,type=str,
                      help='prefix for image filenames')
    args = parser.parse_args()

    filename = args.fmf_file
    startframe = args.start
    endframe = args.stop
    interval = args.interval
    assert interval >= 1
    imgformat = args.extension

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

    fly_movie.seek(startframe)
    frames = range(startframe,endframe+1,interval)
    n_frames = len(frames)
    if args.progress:
        import progressbar
        widgets=['fmf2bmps', progressbar.Percentage(), ' ',
                 progressbar.Bar(), ' ', progressbar.ETA()]
        pbar=progressbar.ProgressBar(widgets=widgets,
                                     maxval=n_frames).start()
    else:
        pbar = None

    for count,frame_number in enumerate(frames):
        if pbar is not None:
            pbar.update(count)
        frame,timestamp = fly_movie.get_frame(frame_number)

        mono=False
        if (fmf_format in ['RGB8','ARGB8','YUV411','YUV422'] or
            fmf_format.startswith('MONO8:') or
            fmf_format.startswith('MONO32f:')):
            save_frame = imops.to_rgb8(fmf_format,frame)
        else:
            if fmf_format not in ['MONO8','MONO16']:
                warnings.warn('converting unknown fmf format %s to mono'%(
                    fmf_format,))
            save_frame = imops.to_mono8(fmf_format,frame)
            mono=True
        h,w=save_frame.shape[:2]
        if mono:
            im = Image.fromstring('L',(w,h),save_frame.tostring())
        else:
            im = Image.fromstring('RGB',(w,h),save_frame.tostring())
        f='%s_%08d.%s'%(os.path.join(outdir,base),frame_number,imgformat)
        im.save(f)
    if pbar is not None:
        pbar.finish()

if __name__=='__main__':
    main()
