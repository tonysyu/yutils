import numpy as np
import matplotlib as mpl
from matplotlib import colors


class _AlphaColormap(colors.LinearSegmentedColormap):
    max_alpha = 0.5
    def __call__(self, X, alpha=None, bytes=False):
        rgba = colors.LinearSegmentedColormap.__call__(self, X, alpha=alpha,
                                                       bytes=bytes)
        intensity = np.asarray(X)
        if np.issubdtype(intensity.dtype, np.integer):
            intensity = intensity / 255.
        intensity = np.clip(intensity, 0, 1)
        alpha = self.max_alpha * self._intensity2alpha(intensity)
        if np.issubdtype(intensity.dtype, np.integer):
            alpha = np.uint8(np.round(255 * alpha))
        if len(rgba) == 4:
            return rgba[:3] + (alpha,)
        else:
            rgba.flat[3::4] = alpha.flat
        return rgba

    def _intensity2alpha(self, intensity):
        raise NotImplementedError()


class _DivergingAlphaColormap(_AlphaColormap):

    def _intensity2alpha(self, intensity):
        """Scale alpha based on intensity values (0 <= intensity <= 1)"""
        # scale alpha so 0 and 1 intensity maps to 1, while 0.5 maps to 0
        return np.abs(2 * (intensity - 0.5))

class _SequentialAlphaColormap(_AlphaColormap):

    def _intensity2alpha(self, intensity):
        """Scale alpha based on intensity values (0 <= intensity <= 1)"""
        return intensity


_lutsize = mpl.rcParams['image.lut']
_spec = {'blue': [(0.0, 0.3803921639919281,  0.3803921639919281),
                  (0.5, 0.3803921639919281,  0.3803921639919281),
                  (0.5, 0.12156862765550613, 0.12156862765550613),
                  (1.0, 0.12156862765550613, 0.12156862765550613)],

        'green': [(0.0, 0.18823529779911041, 0.18823529779911041),
                  (0.5, 0.18823529779911041, 0.18823529779911041),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)],

        'red': [(0.0, 0.019607843831181526, 0.019607843831181526),
                (0.5, 0.019607843831181526, 0.019607843831181526),
                (0.5, 0.40392157435417175, 0.40392157435417175),
                (1.0, 0.40392157435417175, 0.40392157435417175)]}
blue_white_red = _DivergingAlphaColormap(_spec, _spec, _lutsize)

_spec_seq = {'blue': [(0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],

            'green': [(0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],

            'red': [(0.0, 0.40392157435417175, 0.40392157435417175),
                    (1.0, 0.40392157435417175, 0.40392157435417175)]}
white_red = _SequentialAlphaColormap(_spec_seq, _spec_seq, _lutsize)

wo_speq = {'blue': [(0., 0.386, 0.386),
              (1., 0.386, 0.386)],
     'green': [(0., 0.714, 0.714),
               (1., 0.714, 0.714)],
     'red': [(0., 0.979, 0.979),
             (1., 0.979, 0.979)]}
white_orange = _SequentialAlphaColormap(wo_speq, wo_speq, _lutsize)
white_orange.max_alpha = 1

