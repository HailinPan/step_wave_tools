from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import copy
from .find_level import find_levels



def transitions_to_stair_signal(
    signal: np.array,
    transitions: np.array,
) -> np.array:
    stair = np.zeros(len(signal))
    for i in range(len(transitions)-1):
        start, end = transitions[i], transitions[i+1]
        stair[start:end] = [np.median(signal[start:end])] * (end - start)
    return stair

def _find_stairs(
    signal: np.array,
    sensitivity: float = 1.0,
    min_level_length: int = 50,
) -> Union[np.array, np.array]:
    transitions, features, errors, stiffnesses = find_levels(
        signal, sensitivity=sensitivity, min_level_length=min_level_length
        )
    stair_signal = transitions_to_stair_signal(signal=signal, transitions=transitions)

    return stair_signal, transitions

def find_stairs_for_an_obj(
    obj: dict,
    sensitivity: float = 1.0,
    min_level_length: int = 50,
    truncate_to_raw_window: bool = True,
    in_place: bool = False,
) -> Union[dict, None]:
    if not in_place:
        new_obj = copy.deepcopy(obj)
    else:
        new_obj = obj

    for read_id, read_obj in new_obj.items():
        window_s, window_e = read_obj['window'][0], read_obj['window'][1]
        signal = read_obj['signal'][window_s:window_e]
        if truncate_to_raw_window:
            read_obj['signal'] = signal
            read_obj['window'] = [0, len(read_obj['signal'])-1]
            del read_obj['leading_platform_index']
            del read_obj['staircasing_for_finding_window']

        stair_signal, transitions = _find_stairs(
            signal=signal,
            sensitivity=sensitivity,
            min_level_length=min_level_length,
            )
        
        read_obj['stair_signal'] = stair_signal
        read_obj['transitions'] = transitions

    if not in_place:
        return new_obj
    else:
        return None






def _find_change_point_by_lrt(x, n_point=10, flank_len=100):
    #log probability ratio l = -flank_len*log(sigma1)-flank_len*log(sigma2) + 2*flank_len*log(sigma0)
    #sigma1:前flank_len的std
    #sigma2:后flank_len的std
    #sigma0:前后各flank_len一起的std
    #D(X) = E(X^2)-(E(X))^2
    x2 = x**2
    
    # pre
    k = np.zeros(flank_len*2+1)
    k[0:flank_len] = 1/flank_len
    e_x2 = conv1d_torch(x2, k)
    ex_2 = conv1d_torch(x, k) ** 2
    pre_dx = e_x2 - ex_2

    # pos
    k = np.zeros(flank_len*2+1)
    k[-flank_len:] = 1/flank_len
    e_x2 = conv1d_torch(x2, k)
    ex_2 = conv1d_torch(x, k) ** 2
    pos_dx = e_x2 - ex_2

    #com
    k = np.zeros(flank_len*2+1)
    k[:] = 1/(flank_len*2)
    k[flank_len] = 0
    e_x2 = conv1d_torch(x2, k)
    ex_2 = conv1d_torch(x, k) ** 2
    com_dx = e_x2 - ex_2

    pre_dx = pre_dx.astype(np.float64)
    pos_dx = pos_dx.astype(np.float64)
    com_dx = com_dx.astype(np.float64)

    pre_dx[pre_dx<=0] = 1e-5
    pos_dx[pos_dx<=0] = 1e-5
    com_dx[com_dx<=0] = 1e-5

    ls = - flank_len*np.log(np.sqrt(pre_dx)) - flank_len*np.log(np.sqrt(pos_dx)) + 2*flank_len*np.log(np.sqrt(com_dx))

    ls[0:flank_len] = 0
    ls[-flank_len:] = 0
    ls[ls<0] = 0
    ls = np.nan_to_num(ls)
    return ls



def get_topk_points(ls, n_point=10, min_dis=10):
    change_point_indexs = []
    orders = np.argsort(ls*(-1))
    for one_order in orders:
        if len(change_point_indexs) == n_point:
            break
        if len(change_point_indexs) == 0:
            change_point_indexs.append(one_order)
        if can_add(change_point_indexs, one_order, min_dis):
            change_point_indexs.append(one_order)
    return change_point_indexs

def get_higher_points(ls, l_cutoff:float=1.0, min_dis=10):
    change_point_indexs = []
    orders = np.argsort(ls*(-1))
    for one_order in orders:
        if ls[one_order]<l_cutoff:
            continue
        if len(change_point_indexs) == 0:
            change_point_indexs.append(one_order)
        if can_add(change_point_indexs, one_order, min_dis):
            change_point_indexs.append(one_order)
    return change_point_indexs

def can_add(change_point_indexs, one_order, min_dis):
    flag = True
    for i in change_point_indexs:
        if np.abs(one_order-i)<min_dis:
            flag = False
            break
    return flag
            

def conv1d_torch(x, k):
    import torch
    from torch import nn
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(k, list):
        k = np.array(k)
    kernel_size = len(k)
    padding = kernel_size//2

    x = x.astype(np.float32)
    x = torch.Tensor(x[None,:])

    k = k.astype(np.float32)
    k = torch.Tensor(k[None, None, :])
    
    m = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
    m.weight = torch.nn.Parameter(k)

    return m(x).detach().numpy()[0]