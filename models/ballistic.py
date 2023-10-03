import copy
from dataclasses import dataclass, field, make_dataclass
from functools import cached_property
import random
import warnings

from calib3d import Point3D, Point2D
import numpy as np
import scipy.optimize
from tasks.ballsize import compute_projection_error

from deepsport_utilities.court import BALL_DIAMETER
from deepsport_utilities.ds.instants_dataset import BallState, Ball


g = 9.81 * 100 /(1000*1000) # m/s² => cm/ms²




import scipy.integrate
import scipy.optimize
from deepsport_utilities.court import BALL_DIAMETER


class ParabolaBallistic:
    def __init__(self, calib, points2D, timestamps):
        self.calib = calib
        self.points2D = points2D
        self.timestamps = timestamps
        self.T0 = timestamps[0]
        P = calib.P
        ti = timestamps - self.T0
        A1 = np.vstack(np.dstack(np.broadcast_arrays(P[None,0:2,0:3], P[None,0:2,0:3]*ti[...,None,None])))
        A2 = np.tile(points2D.T.reshape(-1, 1)*(P[2,0:3]), [1, 2])*np.repeat(np.repeat(np.vstack(np.broadcast_arrays(1, ti)).T, 2, axis=0), 3, axis=1)
        A = A1 - A2
        b1 = (np.repeat(P[2,3]+P[2,2]*g*ti**2/2, 2)*points2D.T.reshape(1,-1)).T
        b2 = (P[0:2,2:3]*g*ti**2/2).T.reshape(-1, 1)
        b3 = np.tile(P[0:2,3], len(ti))[None].T
        b = b1 - b2 - b3

        initial_guess = (np.linalg.inv(A.T@A)@A.T@b).flatten()

        self.p0 = Point3D(initial_guess[0:3])
        self.v0 = Point3D(initial_guess[3:6])
        self.a0 = Point3D(0, 0, g)

    def __call__(self, t):
        t = t-self.T0
        return self.p0 + self.v0*t + self.a0*t**2/2

class DragBallistic:
    def __init__(self, calib, points2D, timestamps):
        self.calib = calib
        self.timestamps = timestamps
        self.T0 = timestamps[0]

        _fm=100  # length units per meters
        _fs=1000 # duration units per seconds

        g = 9.81*_fm/(_fs**2)  # [m/ms²]
        max_speed = 14*_fm/_fs # [m/s]
        max_height = 6*_fm     # [m]
        margin = 2*_fm         # [m]
        d = BALL_DIAMETER/_fm  # [m]

        t_eval = (timestamps - self.T0)*_fs/1000

        rho = 1.268/(_fm**3)  # [kg/m³]
        A = 0.045*_fm**2      # [m²]
        Cd = 0.2              # []
        m = 0.62              # [kg]
        k0 = rho*A*Cd/2/m

        n0 = 1                # []
        self.points3D = None

        def error(parameters):
            x0, y0, z0, vx0, vy0, vz0, k, n = parameters

            def fun(t, X, *p):
                if isinstance(p[0], tuple):
                    p = p[0]
                x, y, z, vx, vy, vz = X
                k, n = p
                v2 = (vx**2+vy**2+vz**2)
                dXdt = np.vstack([
                    vx,
                    vy,
                    vz,
                    -k*vx*v2**((n-1)/2),
                    -k*vy*v2**((n-1)/2),
                    g+k*vz*v2**((n-1)/2)
                ])
                return dXdt

            X0 = np.array([x0, y0, z0, vx0, vy0, vz0])
            result = scipy.integrate.solve_ivp(fun, (t_eval[0], t_eval[-1]), X0, t_eval=t_eval, dense_output=True, vectorized=True, args=(k, n))
            self.points3D = Point3D(result['y'][0:3])
            return np.mean(np.linalg.norm(points2D - calib.project_3D_to_2D(self.points3D), axis=0))

        model = ParabolaBallistic(calib, points2D, timestamps)
        p0 = model.p0
        v0 = model.v0
        initial_guess =   (     p0.x     ,      p0.y     ,     p0.z    ,     v0.x   ,     v0.y   ,     v0.z   ,   k0  ,  n0  )
        bounds = list(zip(( 0*_fm-margin ,  0*_fm-margin , -max_height , -max_speed , -max_speed , -max_speed ,   0   ,   0  ),
                          (28*_fm+margin , 15*_fm+margin ,    -d/2     ,  max_speed ,  max_speed ,  max_speed , 10*k0 , 2*n0 )))
        result = scipy.optimize.minimize(error, initial_guess, bounds=bounds)
        self.p0 = result['x'][0:3]
        self.v0 = result['x'][3:6]
        self.k = result['x'][6]
        self.n = result['x'][7]

    def __call__(self, t):
        if t == self.timestamps:
            return self.points3D
        raise
