"""
Mercurial-related helper functions.
"""

import mercurial.ui
import mercurial.hg

def get_revision(repo_path):
    r = mercurial.hg.repository(mercurial.ui.ui(), path=repo_path)
    parents = r.parents()
    rev_num = parents[0].rev()
    return rev_num

if __name__ == '__main__':
    import os
    print 'Current Revision:', get_revision(os.path.dirname(__file__))