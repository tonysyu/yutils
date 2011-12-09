#!/usr/bin/env python
"""
Collection of commonly used plotstyles.

The functions in this module change the global plotting parameters.
"""
import matplotlib.pyplot as plt
import numpy as np


THESIS_WIDTH_INCHES = 6.5 # textwidth of page with 1-inch margins.
TWO_COLUMN_WIDTH_INCHES = 3.375 # 3 3/8 (from APS website)
JFM_WIDTH_INCHES = 5.11 # 13 cm (from JFM website)
PRESENTATION_WIDTH_100DPI = 9.7 # at 100 dpi, fits 1024x768 display w/ 54 px pad
PRESENTATION_WIDTH_72DPI = 13.5 # at 72 dpi, fits 1024x768 display w/ 54 px pad


def thesis(width=4, figaspect=1.35):
    """Change plot settings for thesis.

    Parameters
    ----------
    figaspect : float
        aspect ratio of figure
    """
    plt.rc('figure', figsize=figsize(width, figaspect))
    plt.rc('font', size=10)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', linewidth=0.5)
    plt.rc('patch', linewidth=0.5)
    plt.rc('savefig', extension='pdf')

def thesis_full(figaspect=1.75):
    """Change plot settings for thesis.

    Parameters
    ----------
    figaspect : float
        aspect ratio of figure
    """
    thesis(THESIS_WIDTH_INCHES, figaspect=figaspect)

def publication(width=4, figaspect=1.35):
    """Change plot settings for publication.

    Parameters
    ----------
    figaspect : float
        aspect ratio of figure
    """
    plt.rc('figure', figsize=figsize(width, figaspect))
    plt.rc('font', size=9)
    plt.rc('legend', fontsize=9)
    plt.rc('lines', linewidth=0.5)
    plt.rc('axes', linewidth=0.5)
    plt.rc('patch', linewidth=0.5)
    plt.rc('savefig', extension='pdf')
    plt.rc('figure.subplot', left=0.17, bottom=0.16, right=0.93, top=0.93)


def twocolumn(figaspect=1.35):
    """Change plot settings for publication in two-column format.

    Parameters
    ----------
    figaspect : float
        aspect ratio of figure
    """
    publication(TWO_COLUMN_WIDTH_INCHES, figaspect=figaspect)


def jfm(figaspect=1.35, frac_page_width=0.7):
    """Change plot settings for Journal of Fluid Mechanics."""
    width = JFM_WIDTH_INCHES * frac_page_width
    publication(width, figaspect=figaspect)


def presentation_figwidth(frac_page_width=1, pdf=None):
    """Return figure width calculated from fraction of presentation page.

    Parameters
    ----------
    frac_page_width : 0 <= float < 1
        width of figure as a fraction of the full page width. `width` argument
        is ignored of if frac_page_width is set.
    """
    if pdf is not None:
        print DeprecationWarning('"pdf" argument no longer needed.')
    return frac_page_width * PRESENTATION_WIDTH_72DPI


def presentation(width=PRESENTATION_WIDTH_72DPI/2., frac_page_width=None,
                 figaspect=1.35, pdf=None):
    """Change plot settings for presentation.

    Parameters
    ----------
    figaspect : float
        aspect ratio of figure
    width : float
        width of figure in inches
    frac_page_width : 0 <= float < 1
        width of figure as a fraction of the full page width. `width` argument
        is ignored of if frac_page_width is set.
    """
    if pdf is not None:
        print DeprecationWarning('"pdf" argument no longer needed.')
    if frac_page_width is not None:
        width = presentation_figwidth(frac_page_width)
    plt.rc('figure', figsize=figsize(width, figaspect), dpi=72)
    plt.rc('font', size=18)
    plt.rc('legend', fontsize=18)
    plt.rc('lines', linewidth=2, markersize=6)
    plt.rc('savefig', dpi=72)
    plt.rc('figure.subplot', left=0.16, bottom=0.13, wspace=0.35)


def thumbnail():
    plt.rc('font', size='28')
    plt.rc('lines', linewidth=6, markersize=24, markeredgewidth=4)
    plt.rc('savefig', dpi=150)
    plt.rc('legend', fontsize=24)
    plt.rc('figure.subplot', left=0.24, bottom=0.2)

if __name__ == '__main__':
    presentation(frac_page_width=1)

    n_lines = 10
    cycle_cmap(n_lines, 'Oranges')
    x = np.linspace(0, 10)
    f, ax1 = plt.subplots()
    for shift in np.linspace(0, np.pi, n_lines):
        ax1.plot(x, np.sin(x - shift))
    ax1.set_title('cycle colormap')

    cm = make_color_manager((5, 10), start=100)
    f, ax2 = plt.subplots()
    ax2.plot([0, 1], color=cm(5))
    ax2.plot([0.5, 0.5], color=cm(7.5))
    ax2.plot([1, 0], color=cm(10))
    ax2.legend(('val = 5', 'val = 7.5', 'val = 10'))
    ax2.set_title('color based on parameter value')

    plt.show()

