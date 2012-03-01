import matplotlib.pyplot as plt

from yutils import deprecate


__all__ = ['tight_layout', 'tight']


@deprecate('plt.tight_layout')
def tight_layout(*args, **kwargs):
    return plt.tight_layout(*args, **kwargs)


@deprecate('plt.tight_layout')
def tight(*args, **kwargs):
    return plt.tight_layout(*args, **kwargs)

