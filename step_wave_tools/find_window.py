import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Literal, List
import pywt
from scipy.signal import butter, lfilter, find_peaks
import copy

def butter_lowpass(cutoff, fs, order=2):
    #这里假设采样频率为fs=5000hz,要滤除1000hz以上频率成分，即截至频率为1000hz,则wn=2*1000/5000=0.4
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=1000, fs=5000, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def terracing(data):

    wavelet = 'db1'  # 选择小波基函数
    levels = 9  # 设置分解级别

    coeffs = pywt.wavedec(data, wavelet, level=levels)
    # coeffs 是一个列表，包含近似系数和细节系数; coeffs[0] 是最高层次的近似系数; coeffs[1:] 是细节系数，从低频到高频

    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    cD1.fill(0)
    cD2.fill(0)
    cD3.fill(0)
    cD4.fill(0)
    # cD5.fill(0)
    # cD6.fill(0)
    # cD7.fill(0)
    # cD8.fill(0)
    # cD9.fill(0)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)

    rdata_diff = np.diff(rdata)
    peaks_idx, _ = find_peaks(np.abs(rdata_diff))

    step_width = int(np.diff(peaks_idx).mean())

    return rdata, peaks_idx, step_width


def DWT_transform(data):

    wavelet = 'db1'  # 选择小波基函数
    levels = 7  # 设置分解级别

    coeffs = pywt.wavedec(data, wavelet, level=levels)
    # coeffs 是一个列表，包含近似系数和细节系数; coeffs[0] 是最高层次的近似系数; coeffs[1:] 是细节系数，从低频到高频

    cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    cD1.fill(0)
    cD2.fill(0)
    # cD3.fill(0)
    # cD4.fill(0)
    # cD5.fill(0)
    # cD6.fill(0)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


def _find_start_end(
    data: np.array, # data after low pass and DWT
    raw_data: np.array, # raw data
    revise_start: bool = False,
    revise_end: bool = False,
) -> Tuple[float, float, np.array]:
    
    denoised_data = DWT_transform(data)
    staircasing, _, step_width = terracing(denoised_data)  # 台阶化; staircasing: 小波去燥之后的信号
    if len(data) %2 == 1:
        staircasing = staircasing[1:]

    first_diff = np.diff(staircasing)

    # 找起点
    min_diff_idx_start = np.argmin(first_diff[:(len(first_diff)//3)])

    if revise_start:
        slice_data = first_diff[min_diff_idx_start: min_diff_idx_start + step_width * 4 + 1]
        if np.all(slice_data <= 0):
            start = min_diff_idx_start + step_width * 4
        else:
            for idx, diff in enumerate(slice_data):
                if diff > 0:
                    start = idx + min_diff_idx_start
                    break
    else:
        start = min_diff_idx_start
        for i in range(min_diff_idx_start, len(raw_data)):
            if raw_data[i] <= raw_data[start]:
                start = i
            else:
                break

    # 找终点
    min_diff_idx_end = np.argmin(first_diff[int(0.35 * (len(first_diff))): int(0.85 * (len(first_diff)))]) + int(0.35 * len(first_diff))

    if revise_end:
        slice_data = first_diff[min_diff_idx_end - step_width * 4: min_diff_idx_end + 1]

        if slice_data[-step_width-1] < 0:
            end = min_diff_idx_end - step_width
        else:
            end = min_diff_idx_end
    else:
        end = min_diff_idx_end
        for i in range(min_diff_idx_end, start, -1):
            if raw_data[i] >= raw_data[end]:
                end = i
            else:
                break

    return start, end, staircasing

def find_start_end_for_an_obj(
    obj: dict,
    butter_low_pass_trunk: int = 5,
    in_place: bool = False,
    truncate_to_raw_window: bool = True,
    **kwargs,
) -> Union[dict, None]:
    """find start and end by Butterworth lowpass filter and DWT
    1. read有给定的raw window
    2. 根据raw window提取信号
    3. Butterworth lowpass filter
    4. 两次DWT
    5. 根据差分定义start和end
    6. 用新的start和end修改raw window
    7. 如果truncate_to_raw_window=True，则信号会截断成只保留raw window区域的信号，且新的window坐标是根据截断后的来算的。

    Args:
        obj (dict): dict
        butter_low_pass_trunk (int, optional): remove points in this value from the head of result of Butterworth lowpass filter. Defaults to 5.
        in_place (bool, optional): in place or not. Defaults to False.
        truncate_to_raw_window (bool, optional): whether to trancate signal to raw window. Defaults to False.
        **kwargs: Keyword arguments to pass to :func:`_find_start_end`.


    Returns:
        Union[dict, None]: in_place=True: None, in_place=False: dict
    """
    if not in_place:
        new_obj = copy.deepcopy(obj)
    else:
        new_obj = obj

    for read_id, read_obj in new_obj.items():
        signal = read_obj['signal']
        raw_start, raw_end = read_obj['window']
        raw_x = signal[raw_start:raw_end]
        x = raw_x / read_obj['OpenPore']
        x = butter_lowpass_filter(x, cutoff=1000, fs=5000, order=3)
        x_trunk = x[butter_low_pass_trunk:]
        raw_x_trunk = raw_x[butter_low_pass_trunk:]
        start, end, staircasing = _find_start_end(x_trunk, raw_x_trunk, **kwargs)
        start, end = start + butter_low_pass_trunk, end + butter_low_pass_trunk
        if truncate_to_raw_window:
            read_obj['signal'] = read_obj['signal'][raw_start:raw_end]
        else:
            start, end = start + raw_start, end + raw_start
        read_obj['window'] = [start, end]

        # add trunked signal to head of staircasing
        staircasing = np.concatenate([x[0:butter_low_pass_trunk], staircasing])
        read_obj['staircasing_for_finding_window'] = staircasing

    if in_place:
        return None
    else:
        return new_obj
