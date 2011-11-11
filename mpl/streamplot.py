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
    a00 = a[y,x]
    a01 = a[y,x+1]
    a10 = a[y+1,x]
    a11 = a[y+1,x+1]
    xt = xi - x
    yt = yi - y
    a0 = a00*(1-xt) + a01*xt
    a1 = a10*(1-xt) + a11*xt
    return a0*(1-yt) + a1*yt


class Grid(object):
    def __init__(self, x, y):
        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]


def streamplot(x, y, u, v, density=1, linewidth=1,
               color='k', cmap=None, norm=None, vmax=None, vmin=None,
               arrowsize=1, INTEGRATOR='RK4', ax=None):
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

    ## Sanity checks.
    assert len(x.shape)==1
    assert len(y.shape)==1
    assert u.shape == (len(y), len(x))
    assert v.shape == (len(y), len(x))
    if type(linewidth) == np.ndarray:
        assert linewidth.shape == (len(y), len(x))
    if type(color) == np.ndarray:
        assert color.shape == (len(y), len(x))

    grid = Grid(x, y)

    ## Now rescale velocity onto axes-coordinates
    u = u / grid.width
    v = v / grid.height
    speed = np.sqrt(u*u+v*v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= grid.nx
    v *= grid.ny
    ## Now u and v in grid-coordinates.

    ## Blank array: This is the heart of the algorithm. It begins life
    ## zeroed, but is set to one when a streamline passes through each
    ## box. Then streamlines are only allowed to pass through zeroed
    ## boxes. The lower resolution of this grid determines the
    ## approximate spacing between trajectories.
    if type(density) == float or type(density) == int:
        assert density > 0
        NBX = int(30*density)
        NBY = int(30*density)
    else:
        assert len(density) > 0
        NBX = int(30*density[0])
        NBY = int(30*density[1])
    blank = np.zeros((NBY,NBX))

    ## Constants for conversion between grid-index space and
    ## blank-index space
    bx_spacing = grid.nx/float(NBX-1)
    by_spacing = grid.ny/float(NBY-1)

    def blank_pos(xi, yi):
        ## Takes grid space coords and returns nearest space in
        ## the blank array.
        return int((xi / bx_spacing) + 0.5), \
               int((yi / by_spacing) + 0.5)

    def forward_time(xi, yi):
        dt_ds = 1./value_at(speed, xi, yi)
        ui = value_at(u, xi, yi)
        vi = value_at(v, xi, yi)
        return ui*dt_ds, vi*dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def within_index_grid(xi, yi):
        return xi >= 0 and xi < grid.nx-1 and yi >= 0 and yi < grid.ny-1

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01 #min(1./grid.ny, 1./grid.ny, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
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
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)

                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb,new_xb] == 0:
                        blank[new_yb,new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
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
            xb, yb = blank_pos(xi, yi)
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
                    # Next, if s gets to thres, check blank.
                    new_xb, new_yb = blank_pos(xi, yi)
                    if new_xb != xb or new_yb != yb:
                        # New square, so check and colour. Quit if required.
                        if blank[new_yb,new_xb] == 0:
                            blank[new_yb,new_xb] = 1
                            bx_changes.append(new_xb)
                            by_changes.append(new_yb)
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
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1:
            return None

        if stotal > .2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []
    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb*bx_spacing, yb*by_spacing)
            if t != None:
                trajectories.append(t)

    blank_grid_size = max(NBX,NBY)
    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range(blank_grid_size/2):
        for xi in range(blank_grid_size-2*indent):
            traj(xi+indent, indent)
            traj(xi+indent, NBY-1-indent)
            traj(indent, xi+indent)
            traj(NBX-1-indent, xi+indent)

    ## PLOTTING HERE.
    #plt.pcolormesh(np.linspace(x.min(), x.max(), NBX+1),
    #                 np.linspace(y.min(), y.max(), NBY+1), blank)

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

        points = np.array([tx, ty]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        args = {}
        if type(linewidth) == np.ndarray:
            args['linewidth'] = value_at(linewidth, tgx, tgy)[:-1]
            arrowlinewidth = args['linewidth'][len(tgx)/2]
        else:
            args['linewidth'] = linewidth
            arrowlinewidth = linewidth

        if type(color) == np.ndarray:
            args['color'] = cmap(norm(vmin=vmin,vmax=vmax)
                                 (value_at(color, tgx, tgy)[:-1]))
            arrowcolor = args['color'][len(tgx)/2]
        else:
            args['color'] = color
            arrowcolor = color

        lc = matplotlib.collections.LineCollection\
             (segments, **args)
        ax.add_collection(lc)

        ## Add arrows half way along each trajectory.
        n = len(tx)/2
        p = mpp.FancyArrowPatch((tx[n],ty[n]), (tx[n+1],ty[n+1]),
                                arrowstyle='->', lw=arrowlinewidth,
                                mutation_scale=20*arrowsize, color=arrowcolor)
        ax.add_patch(p)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return

def test():
    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)
    u = -1-x**2+y[:,np.newaxis]
    v = 1+x-y[:,np.newaxis]**2
    speed = np.sqrt(u*u + v*v)

    f, axes = plt.subplots(ncols=3)
    streamplot(x, y, u, v, density=1, color='b', ax=axes[0])
    lw = 5*speed/speed.max()
    streamplot(x, y, u, v, density=(1,1), color=u, linewidth=lw, ax=axes[1])
    streamplot(x, y, u, v, density=(1,1), INTEGRATOR='RK45', ax=axes[2])

    plt.show()

if __name__ == '__main__':
    test()

