"""
Testing and profile related helper functions.
"""
import os
import pstats
import cProfile


def profile(cmd, numstats=20):
    stat_file = 'profile_run_TRASH_ME.stats'
    cProfile.run(cmd, stat_file)
    stats = pstats.Stats(stat_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative').print_stats(numstats)
    os.remove(stat_file)
