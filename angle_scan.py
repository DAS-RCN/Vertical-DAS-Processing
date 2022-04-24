import numpy as np
from numba import njit, prange


@njit()
def calc_coherence(data, semb_win, mode='semb'):
    """" Calculate coherency over 2-D array data
    Inputs:
        data - 2D numpy array, of dimensions [channels, samples]
        semb_win - window size, in samples
        mode - type of coherency measure
            semb - Windowed semblance (result between 0 and 1)
            stack - mean over channels
            sembstack - semblance multiplied by absolute value of stack (result > 0)
    Output:
        1D numpy array of sample-by-sample coherency
    """
    nchan, nt = data.shape
    semblance_res = np.zeros(shape=(nt,), dtype=np.float32)
    sum_of_sqr = np.zeros(shape=(nt,), dtype=np.float32)
    stack_res = np.zeros(shape=(nt,), dtype=np.float32)
    half_win = int(np.floor(semb_win/2))

    sum_of_sqr[:] = 0.0
    stack_res = np.sum(data, axis=0)
    sum_of_sqr = np.sum(np.power(data, 2.0), axis=0)

    for w in range(half_win, nt-half_win):
        if sum_of_sqr[w] == 0 or n_live_ch[w]==0:
            semblance_res[w] = 0
        else:
            semblance_res[w] = np.divide(np.sum(np.power(stack_res[w-half_win:w+half_win+1], 2.0)),
                                         np.sum(sum_of_sqr[w-half_win:w+half_win+1])) / nchan

    semblance_res[0:half_win] = semblance_res[half_win]
    semblance_res[nt-half_win:nt] = semblance_res[nt-half_win-1]
    if mode == 'semb':
        return semblance_res
    elif mode == 'stack':
        return stack_res/nchan
    elif mode == 'sembstack':
        return semblance_res*np.abs(stack_res)/nchan
    else:
        raise RuntimeError


@njit()
def apply_moveout(data, moveout_vec, dt=1):
    """ Applies time shift to a 2-D array
    Inputs:
        dat - 2D numpy array, of dimensions [channels, samples]
        moveout_vec - 1D numpy array, of size [channels], containing non-negative time shifts for each channel.
                      Data are always shifted "back" in time, i.e. to earlier times
                      Shifts are relative, and the channel with lowest value is unchanged.
        dt - time sampling, in seconds. If empty, assumed to be 1 and uses moveout_vec as number of samples
    Output:
        2D array after temporal shifts
    """

    # TODO apply in the frequency domain for higher accuracy

    nch, nt = data.shape

    if len(moveout_vec) != nch:
        raise RuntimeError('Number of data channels does not match number of channels in moveout vector')

    if np.any(moveout_vec < 0):
        raise RuntimeError('Time shifts must be positive')

    mov_vec = moveout_vec - np.amin(moveout_vec)
    mov_vec = mov_vec/dt
    mov_data = np.zeros(shape=(nch, nt), dtype=np.float32)
    samp_bef = np.float32(0.0)
    int_fac = np.float32(0.0)

    for ch in range(nch):
        samp_bef = np.rint((np.floor(mov_vec[ch])))
        int_fac = mov_vec[ch]-np.floor(mov_vec[ch])
        if int_fac == 0.0:
            int_fac = np.float32(1.0)
        mov_data[ch, 0:nt - samp_bef - 1] = (1.0-int_fac)*data[ch, samp_bef:-1] + int_fac*data[ch, samp_bef+1:]

    return mov_data


@njit(parallel=True)
def scan_angles(data, delay_tables, semb_win,  dt=1, mode='semb'):
    """ Scans for optimal angle of incidence for a given data array
        Inputs:
        data - 2D numpy array, of dimensions [channels, samples]
        delay_tables - table of calculated delays (see TT_prediction: pred_vertical_tt), dimensions [angles, channels]
        semb_win - window size, in seconds
        dt - time sampling, in seconds. If empty, assumed to be 1 and uses moveout_vec as number of samples
        mode - type of coherency measure
            semb - Windowed semblance (result between 0 and 1)
            stack - mean over channels
            sembstack - semblance multiplied by absolute value of stack (result > 0)

    Output:
        ang_inds - vector containing optimal angle estimated at each time sample
        coherence - vector containing coherence value for the matching angle at each time sample
    """
    nch, nt = data.shape
    nang, nch2 = delay_tables.shape

    if nch != nch2:
        raise RuntimeError('Number of data channels does not match number of channels in delay table')

    semb_res = np.zeros(shape=(nang, nt), dtype=np.float32)
    int_semb_win = np.rint(semb_win/dt)
    for i in prange(nang):
        semb_res[i, :] = calc_coherence(apply_moveout(data, delay_tables[i, :], dt), int_semb_win, mode)

    ang_inds = np.zeros(shape=(nt,), dtype=np.int32)
    coherence = np.zeros(shape=(nt,), dtype=np.float32)

    for t in prange(nt):
        ang_inds[t] = np.argmax(semb_res[:, t])
        coherence[t] = semb_res[ang_inds[t], t]

    return (ang_inds, coherence)

