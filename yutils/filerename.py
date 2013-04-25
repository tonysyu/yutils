#!/usr/bin/env python
import glob
import shutil
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('search', help="search string")
    parser.add_argument('replace', help="replacement string")
    parser.add_argument('-f', '--files', nargs='+',
                        help="Files to filter. This is easily specified using "
                             "wild card characters. For example '*.jpg' would "
                             "limit search and replace to jpeg images.")
    parser.add_argument('--run', action='store_true',
                        help="If True, display new names; don't move files")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.files is None:
        args.files = glob.glob('*')

    for old in args.files:
        new = old.replace(args.search, args.replace)
        if new == old:
            continue
        print "%s\n\t-> %s" % (old, new)
        if args.run:
            shutil.move(old, new)

    if not args.run:
        print '~' * 60
        print "Dry run! Add --run flag to actually execute command"
        print '~' * 60

if __name__ == '__main__':
    main()
