import matplotlib.pyplot as plt

from yutils import deprecated


__all__ = ['tight_layout', 'tight']


@deprecated('plt.tight_layout')
def tight_layout(*args, **kwargs):
    return plt.tight_layout(*args, **kwargs)


@deprecated('plt.tight_layout')
def tight(*args, **kwargs):
    return plt.tight_layout(*args, **kwargs)

