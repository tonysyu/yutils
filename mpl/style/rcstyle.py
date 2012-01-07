"""
The functions in this module change the global plotting parameters.
"""
import matplotlib.pyplot as plt


__all__ = ['figsize', 'thesis', 'thesis_full', 'publication', 'twocolumn',
           'pof', 'jfm', 'presentation', 'thumbnail']


THESIS_WIDTH_INCHES = 6.5 # textwidth of page with 1-inch margins.
APS_TWO_COLUMN_WIDTH_INCHES = 3.375 # 3 3/8 (from APS website)
JFM_WIDTH_INCHES = 5.11 # 13 cm (from JFM website)
KEYNOTE_WIDTH_72DPI = 13.5 # at 72 dpi, fits 1024x768 display w/ 54 px pad


def figsize(width, figaspect):
    """Return figsize (width, height) from figure width and aspect ratio"""
    return (width, width/figaspect)


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
    plt.rc('lines', linewidth=0.5, markersize=3)
    plt.rc('axes', linewidth=0.5, titlesize=9)
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
    publication(APS_TWO_COLUMN_WIDTH_INCHES, figaspect=figaspect)


def pof(figaspect=1.35, fullwidth=False):
    width = APS_TWO_COLUMN_WIDTH_INCHES
    if fullwidth:
        raise NotImplementedError("Todo")
    publication(width=width, figaspect=figaspect)
    plt.rc('font', family='serif')


def jfm(figaspect=1.35, frac_page_width=0.7):
    """Change plot settings for Journal of Fluid Mechanics."""
    width = JFM_WIDTH_INCHES * frac_page_width
    publication(width, figaspect=figaspect)
    plt.rc('lines', linewidth=1)


def presentation(width=KEYNOTE_WIDTH_72DPI/2., frac_page_width=None,
                 figaspect=1.35):
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
    if frac_page_width is not None:
        width = frac_page_width * KEYNOTE_WIDTH_72DPI
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
    from yutils.mpl.core import demo_plot

    # Note you should not run multiple rc configurations.
    # rc parameters can affect *some* values in *existing* plots.

    #presentation()
    #f, ax = plt.subplots()
    #demo_plot(ax)

    publication()
    f, ax = plt.subplots()
    demo_plot(ax)

    plt.show()
