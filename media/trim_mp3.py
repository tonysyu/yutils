#!/usr/bin/env python
"""Trim mp3 using ffmpeg"""
import os
import argparse
import yutils.path


class Time(object):
    """
    Parameters
    ----------
    time_str : str
        Time specified as sec, min:sec, or hr:min:sec
    """
    def __init__(self, time_str=None, hour=0, minute=0, second=0):
        if time_str is not None:
            time_tuple = time_str.split(':')
            time_tuple = [int(i) for i in time_tuple]
            number_of_missing_zeros = 3 - len(time_tuple)
            prepend = [0] * number_of_missing_zeros
            self.hour, self.minute, self.second = prepend + time_tuple
        else:
            self.hour = hour
            self.minute = minute
            self.second = second
        assert self.hour >= 0
        assert 0 <= self.minute < 60
        assert 0 <= self.second < 60

    def __add__(self, time):
        hour = self.hour
        minute = self.minute
        second = self.second + time.second
        if second > 60:
            second -= 60
            minute += 1
        minute += time.minute
        if minute > 60:
            minute -= 60
            hour += 1
        hour += time.hour
        return Time(hour=hour, minute=minute, second=second)

    def __sub__(self, time):
        hour = self.hour
        minute = self.minute
        second = self.second - time.second
        if second < 0:
            second += 60
            minute -= 1
            if minute < 0:
                minute += 60
                hour -= 1
        minute -= time.minute
        if minute > 60:
            minute -= 60
            hour -= 1
        assert hour >= 0, "Subtraction gives negative time"
        return Time(hour=hour, minute=minute, second=second)

    def __str__(self):
        return '%s:%s:%s'  % (self.hour, self.minute, self.second)


parser = argparse.ArgumentParser()
parser.add_argument('--start', '-s', type=str, default='0',
                    help=('Start time of mp3 to record. '
                          'Start time assumed to be 0 if not specified. '
                          'Time format: sec, min:sec, or hr:min:sec'))
parser.add_argument('--end', '-e', type=str,
                    help=('End time of mp3 to record. '
                          'Time format: sec, min:sec, or hr:min:sec'))
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', help='output file')


def get_command_string(args):

    assert args.input is not None, "Input file required (-i input_file)"
    assert args.end is not None, "End time required (--end hr:min:sec)"

    if not yutils.path.hasext(args.input):
        args.input = args.input + '.mp3'
    if args.output is None:
        path, ext = os.path.splitext(args.input)
        args.output = path + '_trimmed' + ext
    start = Time(args.start)
    end = Time(args.end)

    args.start = '-ss %s' % start
    duration = '-t %s' % (end - start)

    inputs = (args.start, duration, args.input, args.output)
    cmdstr = 'ffmpeg -acodec copy %s %s -i %s %s' % inputs
    return cmdstr

def main(args):
    cmdstr = get_command_string(args)
    os.system(cmdstr)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

