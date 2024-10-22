from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import copy

def _reset_stair_width(
    stair_signal: np.array,
    transitions: np.array,
    reset_stair_width: int = 1,
) -> Tuple[np.array, np.array]:
    reset_stair_signal = []
    for point in transitions[:-1]:
        reset_stair_signal.append(stair_signal[point])

    reset_stair_signal = np.repeat(reset_stair_signal, repeats=reset_stair_width)
    reset_transitions = [i*reset_stair_width for i in range(len(reset_stair_signal)//reset_stair_width)]
    return reset_stair_signal, reset_transitions


def reset_stair_width_for_an_obj(
    obj: dict,
    reset_stair_width: int = 1,
    in_place: bool = False,
):
    if in_place:
        new_obj = obj
    else:
        new_obj = copy.deepcopy(obj)

    for read_id, read_obj in new_obj.items():
        reset_stair_signal, reset_transitions = _reset_stair_width(
            stair_signal=read_obj['stair_signal'],
            transitions=read_obj['transitions'],
            reset_stair_width=reset_stair_width,
        )
        read_obj['reset_stair_signal'] = reset_stair_signal
        read_obj['reset_transitions'] = reset_transitions
    
    if in_place:
        return None
    else:
        return new_obj