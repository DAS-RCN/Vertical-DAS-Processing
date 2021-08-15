import numpy as np
from numba import njit, prange
import math


@njit(parallel=True)
def pred_vertical_tt(velocity, dz, angles):
    """" Calculates travel time delays for a given velocity and different incidence angles
    Inputs:
        velocity - 1D numpy array of velocities, starting at the top of the array
        dz - distance between adjacent DAS channels (m)
        angles - 1D numpy array of incidence angles, measured at the bottom of the array.
            0 - vertical propagation
            90 - horizontal propagation
    Output:
        2D numpy array with travel-times at each array point as a function of incidence angle.
    """

    if np.amax(np.abs(angles)) > math.pi:
        angles = angles / 180.0 * math.pi

    nz = velocity.size
    nang = angles.size
    vel_ratio = np.zeros(shape=(nz-1,))
    delays = np.zeros(shape=(nang, nz))

    for i in prange(nz-1):
        vel_ratio[i] = velocity[nz-1-i]/velocity[nz-2-i]

    avg_vel = 0.5*(velocity[0:nz-1]+velocity[1:])
    avg_vel = np.flip(avg_vel)

    for i in prange(nang):
        inc_angs = np.zeros(shape=(nz,))
        next_ang = angles[i]
        inc_angs[0] = next_ang
        for j in range(nz-1):
            next_ang = np.arcsin(np.sin(next_ang)/vel_ratio[j])
            inc_angs[j+1] = next_ang

        delays[i, 1:] = dz*np.cumsum(1.0/avg_vel*np.cos(inc_angs[1:]))

    return delays[:, ::-1]
