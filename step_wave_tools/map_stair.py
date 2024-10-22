from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import copy
from dtaidistance import dtw
import matplotlib.pyplot as plt
import matplotlib as mpl




def random_select_one_read(
    obj: dict,
    stair_num: int,
    seed: int = 0,
) -> str:
    np.random.seed(seed)
    read_ids_with_stair_num = get_read_ids_with_specific_stairs(obj, stair_num)
    np.random.shuffle(read_ids_with_stair_num)

    return read_ids_with_stair_num[0]


def get_read_ids_with_specific_stairs(
    obj: dict,
    stair_num: int,
) -> np.array:
    read_ids = []
    for read_id, read_obj in obj.items():
        if 'reset_transitions' in read_obj:
            if len(read_obj['reset_transitions']) == stair_num:
                read_ids.append(read_id)
        elif 'transitions' in read_obj:
            if len(read_obj['transitions']) - 1 == stair_num:
                read_ids.append(read_id)

    return np.array(read_ids)


def dtw_rectificate(data, reference):

    path = dtw.warping_path(data, reference)  # path中记录了一系列索引元组tuple，反映了从 tuple[0] 到 tuple[1] 的映射关系
    new_idx_list = []
    new_vlu_list = []
    for idx_tuple in path:
        from_p = idx_tuple[0]
        to_p = idx_tuple[1]
        new_idx_list.append(to_p)
        new_vlu_list.append(data[from_p])

    new_idx_list = np.array(new_idx_list)  # 索引数组中包含有重复的索引 ，需要删掉
    new_vlu_list = np.array(new_vlu_list)

    _, unique_indices = np.unique(new_idx_list, return_index=True)  # 获取唯一的索引及其第一个出现的位置
    sorted_unique_indices = np.sort(unique_indices)
    new_idx_list = new_idx_list[sorted_unique_indices]  # 根据保留的索引更新 index_array 和 data_array
    new_vlu_list = new_vlu_list[sorted_unique_indices]
    return new_idx_list, new_vlu_list, path


def map_reset_stair_to_ref(
    obj: dict,
    ref_read_id: str,
    in_place: bool = False,
):
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    reference = obj[ref_read_id]['reset_stair_signal']/obj[ref_read_id]['OpenPore']
    for read_id, read_obj in new_obj.items():
        if read_id != ref_read_id:
            data = read_obj['reset_stair_signal']/read_obj['OpenPore']
            new_idx_list, new_vlu_list, path = dtw_rectificate(data, reference)
            read_obj['map_stair_signal'] = new_vlu_list
            read_obj['path'] = path
        else:
            read_obj['map_stair_signal'] = reference
    
    if in_place:
        return None
    else:
        return new_obj
    

def draw_map_relationship(
    obj: dict,
    ref_read_id: str,
    read_id: str,
):
    fig, ax = plt.subplots()
    reference = obj[ref_read_id]['map_stair_signal']
    data = obj[read_id]['reset_stair_signal'] / obj[read_id]['OpenPore']
    map_dict = {}
    for i in obj[read_id]['path']:
        if i[0] not in map_dict:
            map_dict[i[0]] = i[1]
    colors = [mpl.colors.to_hex(i) for i in mpl.colormaps['Paired'].colors]

    ax.plot([0,1], [reference[0], reference[0]], color='black', label='ref')
    for i in range(len(reference)):
        ax.plot([0+i,1+i], [reference[i], reference[i]], color=colors[i])
    
    ax.plot([0,1], [data[0], data[0]], color='black', label='read_to_map', linestyle='dotted')
    for i in range(len(data)):
        ax.plot([0+i,1+i], [data[i], data[i]], color=colors[map_dict[i]], linestyle='dotted')
    
    ax.set_ylabel('I/I0')
    ax.set_xlabel('Stair')
    plt.legend()


