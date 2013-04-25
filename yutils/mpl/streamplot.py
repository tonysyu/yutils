from yutils import deprecated

__all__ = ['streamplot']


@deprecated('plt.streamplot')
def streamplot(*args, **kwargs):
    ax = kwargs.get('ax', plt.gca())
    ax.streamplot(*args, **kwargs)

