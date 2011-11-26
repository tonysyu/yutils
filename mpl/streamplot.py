"""
Streamline plotting like Mathematica.
Copyright (c) 2011 Tom Flannaghan.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

version = '4'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpp


__all__ = ['streamplot']


def value_at(a, xi, yi):
    ## Linear interpolation - nice and quick because we are
    ## working in grid-index coordinates.
    if type(xi) == np.ndarray:
        x = xi.astype(np.int)
        y = yi.astype(np.int)
    else:
        x = np.int(xi)
        y = np.int(yi)
    a00 = a[y, x]
    a01 = a[y, x + 1]
    a10 = a[y + 1, x]
    a11 = a[y + 1, x + 1]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    return a0 * (1 - yt) + a1 * yt


class Grid(object):
    def __init__(self, x, y):

        if len(x.shape) == 2:
            x_row = x[0]
            assert np.allclose(x_row, x)
            x = x_row
        else:
            assert len(x.shape) == 1

        if len(y.shape) == 2:
            y_col = y[:, 0]
            assert np.allclose(y_col, y.T)
            y = y_col
        else:
            assert len(y.shape) == 1

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

    @property
    def shape(self):
        return self.ny, self.nx


class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    Streamlines are only allowed to pass through zeroed regions. The resolution
    of this grid determines the approximate spacing between trajectories.

    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g. if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, density):
        if type(density) == float or type(density) == int:
            assert density > 0
            self.nx = self.ny = int(30 * density)
        else:
            assert len(density) > 0
            self.nx = int(30 * density[0])
            self.ny = int(30 * density[1])
        self._mask = np.zeros((self.ny, self.nx))
        self.size = max(self.ny, self.nx)

    def __setitem__(self, *args):
        idx, value = args
        self._traj.append(idx)
        self._mask.__setitem__(*args)

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def start_trajectory(self):
        # clear any previous trajectories
        self._traj = []

    def undo_trajectory(self):
        for t in self._traj:
            self._mask.__setitem__(t, 0)


def streamplot(x, y, u, v, density=1, linewidth=1, color='k', cmap=None,
               norm=None, vmax=None, vmin=None, arrowsize=1, INTEGRATOR='RK4',
               ax=None):
    """Draws streamlines of a vector flow.

    Parameters
    ----------
    x, y : 1d arrays
        an *evenly spaced* grid.
    u, v : 2d arrays
        x and y-velocities. Number of rows should match length of y, and
        the number of columns should match x.
    density : numeric
        controls the closeness of the streamlines. For different densities in
        each direction, use a tuple or list [densityx, densityy].
    linewidth : numeric or 2d array
        vary linewidth when given a 2d array with the same shape as velocities.
    color : matplotlib color code, or 2d array
      then transformed into color by the 
      A value of None gives the default for each.
        Streamline color. When given an array with the same shape as
        velocities, values are converted to color using cmap, norm, vmin and
        vmax args.

    INTEGRATOR is experimental. Currently, RK4 should be used.
    """
    ax = ax if ax is not None else plt.gca()

    grid = Grid(x, y)

    ## Sanity checks.
    assert u.shape == grid.shape
    assert v.shape == grid.shape
    if type(linewidth) == np.ndarray:
        assert linewidth.shape == grid.shape
    if type(color) == np.ndarray:
        assert color.shape == grid.shape


    ## Now rescale velocity onto axes-coordinates
    u = u / grid.width
    v = v / grid.height
    speed = np.sqrt(u*u+v*v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= grid.nx
    v *= grid.ny
    ## Now u and v in grid-coordinates.

    mask = StreamMask(density)
    ## Constants for conversion between grid-index space and
    ## mask-index space
    bx_spacing = grid.nx / float(mask.nx - 1)
    by_spacing = grid.ny / float(mask.ny - 1)

    def mask_pos(xi, yi):
        ## Takes grid space coords and returns nearest space in
        ## the mask array.
        return int((xi / bx_spacing) + 0.5), \
               int((yi / by_spacing) + 0.5)

    def forward_time(xi, yi):
        ds_dt = value_at(speed, xi, yi)
        dt_ds = 0 if ds_dt == 0 else 1. / ds_dt
        ui = value_at(u, xi, yi)
        vi = value_at(v, xi, yi)
        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def within_index_grid(xi, yi):
        return xi >= 0 and xi < (grid.nx - 1) and yi >= 0 and yi < (grid.ny - 1)

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'mask array'
        ## termination conditions. TODO tidy the integration loops.
        mask.start_trajectory()

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01 #min(1./grid.ny, 1./grid.ny, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = mask_pos(xi, yi)
            xf_traj = []
            yf_traj = []

            while within_index_grid(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + .5*ds*k1x, yi + .5*ds*k1y)
                    k3x, k3y = f(xi + .5*ds*k2x, yi + .5*ds*k2y)
                    k4x, k4y = f(xi + ds*k3x, yi + ds*k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds*(k1x+2*k2x+2*k3x+k4x) / 6.
                yi += ds*(k1y+2*k2y+2*k3y+k4y) / 6.
                # Final position might be out of the domain

                if not within_index_grid(xi, yi):
                    break

                stotal += ds
                # Next, if s gets to thres, check mask.
                new_xb, new_yb = mask_pos(xi, yi)

                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if mask[new_yb,new_xb] == 0:
                        mask[new_yb,new_xb] = 1
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        ## Alternative Integrator function

        ## RK45 does not really help in it's current state. The
        ## resulting trajectories are accurate but low-resolution in
        ## regions of high curvature and thus fairly ugly. Maybe a
        ## curvature based cap on the maximum ds permitted is the way
        ## forward.

        def rk45(x0, y0, f):
            maxerror = 0.001
            maxds = 0.03
            ds = 0.03
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = mask_pos(xi, yi)
            xf_traj = []
            yf_traj = []

            while within_index_grid(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK45
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + .25*ds*k1x,
                                 yi + .25*ds*k1y)
                    k3x, k3y = f(xi + 3./32*ds*k1x + 9./32*ds*k2x,
                                 yi + 3./32*ds*k1y + 9./32*ds*k2y)
                    k4x, k4y = f(xi + 1932./2197*ds*k1x - 7200./2197*ds*k2x
                                    + 7296./2197*ds*k3x,
                                 yi + 1932./2197*ds*k1y - 7200./2197*ds*k2y
                                    + 7296./2197*ds*k3y)
                    k5x, k5y = f(xi + 439./216*ds*k1x - 8*ds*k2x
                                    + 3680./513*ds*k3x - 845./4104*ds*k4x,
                                 yi + 439./216*ds*k1y - 8*ds*k2y
                                    + 3680./513*ds*k3y - 845./4104*ds*k4y)
                    k6x, k6y = f(xi - 8./27*ds*k1x + 2*ds*k2x
                                    - 3544./2565*ds*k3x + 1859./4104*ds*k4x
                                    - 11./40*ds*k5x,
                                 yi - 8./27*ds*k1y + 2*ds*k2y
                                    - 3544./2565*ds*k3y + 1859./4104*ds*k4y
                                    - 11./40*ds*k5y)

                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                dx4 = ds*(25./216*k1x + 1408./2565*k3x
                          + 2197./4104*k4x - 1./5*k5x)
                dy4 = ds*(25./216*k1y + 1408./2565*k3y
                          + 2197./4104*k4y - 1./5*k5y)
                dx5 = ds*(16./135*k1x + 6656./12825*k3x
                          + 28561./56430*k4x - 9./50*k5x + 2./55*k6x)
                dy5 = ds*(16./135*k1y + 6656./12825*k3y
                          + 28561./56430*k4y - 9./50*k5y + 2./55*k6y)

                ## Error is normalized to the axes coordinates (it's a distance)
                error = np.sqrt(((dx5-dx4)/grid.nx)**2 + ((dy5-dy4)/grid.ny)**2)
                if error < maxerror:
                    # Step is within tolerance so continue
                    xi += dx5
                    yi += dy5
                    # Final position might be out of the domain
                    if not within_index_grid(xi, yi):
                        break
                    stotal += ds
                    # Next, if s gets to thres, check mask.
                    new_xb, new_yb = mask_pos(xi, yi)
                    if new_xb != xb or new_yb != yb:
                        # New square, so check and colour. Quit if required.
                        if mask[new_yb,new_xb] == 0:
                            mask[new_yb,new_xb] = 1
                            xb = new_xb
                            yb = new_yb
                        else:
                            break
                    if stotal > 2:
                        break
                # Modify ds for the next iteration.
                if len(xf_traj) > 2:
                    ## hacky curvature dependance:
                    v1 = np.array((xf_traj[-1]-xf_traj[-2],
                                   yf_traj[-1]-yf_traj[-2]))
                    v2 = np.array((xf_traj[-2]-xf_traj[-3],
                                   yf_traj[-2]-yf_traj[-3]))
                    costheta = (v1/np.sqrt((v1**2).sum()) *
                                v2/np.sqrt((v2**2).sum())).sum()
                    if costheta < .8:
                        ds = .01
                        continue
                ds = min(maxds, 0.85*ds*(maxerror/error)**.2)
            return stotal, xf_traj, yf_traj

        if INTEGRATOR == 'RK4':
            integrator = rk4
        elif INTEGRATOR == 'RK45':
            integrator = rk45

        ## Forward and backward trajectories
        sf, xf_traj, yf_traj = integrator(x0, y0, forward_time)
        sb, xb_traj, yb_traj = integrator(x0, y0, backward_time)

        # combine forward and backward trajectories
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1:
            return None

        if stotal > .2:
            initxb, inityb = mask_pos(x0, y0)
            mask[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            mask.undo_trajectory()
            return None

    ## A quick function for integrating trajectories if mask==0.
    trajectories = []
    def traj(xb, yb):
        if xb < 0 or xb >= mask.nx or yb < 0 or yb >= mask.ny:
            return
        if mask[yb, xb] == 0:
            t = rk4_integrate(xb*bx_spacing, yb*by_spacing)
            if t != None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for mask==0 along the edges first, and work inwards.
    for indent in range(mask.size / 2):
        for xi in range(mask.size - 2*indent):
            traj(xi+indent, indent)
            traj(xi+indent, mask.ny - 1 - indent)
            traj(indent, xi+indent)
            traj(mask.nx - 1 - indent, xi + indent)

    ## PLOTTING HERE.
    #plt.pcolormesh(np.linspace(x.min(), x.max(), mask.nx+1),
    #                 np.linspace(y.min(), y.max(), mask.ny+1), mask)

    # Load up the defaults - needed to get the color right.
    if type(color) == np.ndarray:
        if vmin == None: vmin = color.min()
        if vmax == None: vmax = color.max()
        if norm == None: norm = matplotlib.colors.normalize
        if cmap == None: cmap = matplotlib.cm.get_cmap(
            matplotlib.rcParams['image.cmap'])

    for t in trajectories:
        # Finally apply the rescale to adjust back to user-coords from
        # grid-index coordinates.
        tx = np.array(t[0]) * grid.dx + grid.x_origin
        ty = np.array(t[1]) * grid.dy + grid.y_origin

        tgx = np.array(t[0])
        tgy = np.array(t[1])

        points = np.array([tx, ty]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        args = {}
        if type(linewidth) == np.ndarray:
            args['linewidth'] = value_at(linewidth, tgx, tgy)[:-1]
            arrowlinewidth = args['linewidth'][len(tgx) / 2]
        else:
            args['linewidth'] = linewidth
            arrowlinewidth = linewidth

        if type(color) == np.ndarray:
            args['color'] = cmap(norm(vmin=vmin,vmax=vmax)
                                 (value_at(color, tgx, tgy)[:-1]))
            arrowcolor = args['color'][len(tgx) / 2]
        else:
            args['color'] = color
            arrowcolor = color

        lc = matplotlib.collections.LineCollection(segments, **args)
        ax.add_collection(lc)

        ## Add arrows half way along each trajectory.
        n = len(tx) / 2
        p = mpp.FancyArrowPatch((tx[n], ty[n]), (tx[n+1], ty[n+1]),
                                arrowstyle='->', lw=arrowlinewidth,
                                mutation_scale=20*arrowsize, color=arrowcolor)
        ax.add_patch(p)

    ax.update_datalim(((x.min(), y.min()), (x.max(), y.max())))
    ax.autoscale_view(tight=True)
    return

def test():
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    u = -1 - x**2 + y[:,np.newaxis]
    v = 1 + x - y[:,np.newaxis]**2
    speed = np.sqrt(u*u + v*v)

    f, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.ravel()
    streamplot(x, y, u, v, density=1, color='b', ax=axes[0])
    lw = 5*speed/speed.max()
    streamplot(x, y, u, v, density=(1,1), color=u, linewidth=lw, ax=axes[1])
    streamplot(x, y, u, v, density=(1,1), INTEGRATOR='RK45', ax=axes[2])
    # test x, y matrices
    X = np.repeat(x.reshape(1, 100), 100, axis=0)
    Y = np.repeat(y.reshape(100, 1), 100, axis=1)
    streamplot(X, Y, u, v, density=(0.5,1), color=u, linewidth=lw, ax=axes[3])

    plt.show()

if __name__ == '__main__':
    test()

