import numpy as np

def low_n_cpic(x):
    a = 0.5903  #(0.5415, 0.639)
    b = 0.9174  #(0.8761, 0.9586)
    c = 0.7691  #(0.7491, 0.7892)
    p3 = 3.937e-10  #(2.924e-10, 4.95e-10)
    p4 = -1.328e-06 #(-1.589e-06, -1.067e-06)
    p5 = 0.003363   #(0.002932, 0.003795)
    ph = -0.1562    #(-0.1703, -0.1421)
    return a * np.log(np.log(x))+b * np.log(x) + p3 * x**3 + p4 * x**2 + p5 * x + ph * np.abs(x)**0.5 + c

def high_n_cpic(x):
    a = 1.575   #(1.324, 1.826)
    b = 2.168   #(1.601, 2.734)
    c = 1.264   #(1.153, 1.376)

    return  a*np.log(np.log(x))+b*np.log(np.log(np.log(x)))+c
def find_levels(data, sensitivity=1, min_level_length=2, second_pass=False, cpic_multiplier_final=1):
    # Load CPIC fits
    # cpic_fits = scipy.io.loadmat('cpic_fits_1D.mat')
    # print( cpic_fits )
    # low_n_cpic = cpic_fits['lessthan1000']
    # high_n_cpic = cpic_fits['greaterthan1000']

    #define the number  determining level finder sensitivity
    # cpic_multiplier = 1
    # cpic_multiplier_final = 1

    # Remove NaNs from data
    original_data_mapping = np.arange(len(data))
    data = data[~np.isnan(data)]
    original_data_mapping = original_data_mapping[~np.isnan(data)]

    # Cumulative sum of data and squared data
    x = np.concatenate(([0], np.cumsum(data)))
    xsq = np.concatenate(([0], np.cumsum(data ** 2)))

    # Recursive level finding function
    transitions = find_transitions(2, len(data), x, xsq, min_level_length, sensitivity)
    transitions = np.sort(transitions) - 1

    # Delete transitions at the beginning and end which are too short
    transitions = transitions[(transitions > min_level_length) & (transitions <= len(data) - min_level_length)]

    # Append 0 and the end
    transitions_with_ends = np.concatenate(([0], transitions, [len(data)]))

    some_change = True

    if second_pass:
        while some_change:
            some_change = False
            transition_cpic = -np.inf * np.ones(len(transitions_with_ends))

            for ii in range(1, len(transitions_with_ends) - 1):
                left = max(transitions_with_ends[ii - 1], 2)
                right = transitions_with_ends[ii + 1]

                n_t = right - left + 1
                n_l = transitions_with_ends[ii] - left + 1
                n_r = right - transitions_with_ends[ii]

                if n_l.size == 0:
                    return

                x_mean_l = (x[left + n_l - 1] - x[left - 1]) / n_l
                x_mean_r = (x[right] - x[right - n_r]) / n_r
                x_mean_t = (x[right] - x[left - 1]) / n_t

                xsq_mean_l = (xsq[left + n_l - 1] - xsq[left - 1]) / n_l
                xsq_mean_r = (xsq[right] - xsq[right - n_r]) / n_r
                xsq_mean_t = (xsq[right] - xsq[left - 1]) / n_t

                var_l = max(xsq_mean_l - x_mean_l ** 2, 0.0003)
                var_r = max(xsq_mean_r - x_mean_r ** 2, 0.0003)
                var_t = max(xsq_mean_t - x_mean_t ** 2, 0.0003)

                if n_t >= 1e6:
                    p_cpic = high_n_cpic( 1e6 )
                elif 1000 < n_t < 1e6:
                    p_cpic = high_n_cpic( n_t )
                else:
                    p_cpic = low_n_cpic( n_t )

                transition_cpic[ii] = 0.5 * (n_l * np.log(var_l) + n_r * np.log(var_r) - n_t * np.log(var_t)) + 1 + cpic_multiplier_final * p_cpic

            min_cpic, where_max = np.min(transition_cpic), np.argmin(transition_cpic)

            if min_cpic > 0:
                transitions_with_ends = np.delete(transitions_with_ends, where_max)
                some_change = True
    transitions_with_ends = transitions_with_ends.astype(np.int16)
    features = np.zeros((2, len(transitions_with_ends) - 1))
    errors = np.zeros((2, len(transitions_with_ends) - 1))
    stiffnesses = [None] * (len(transitions_with_ends) - 1)

    for ct in range(1, len(transitions_with_ends)):
        features[:, ct - 1] = [np.median(data[transitions_with_ends[ct - 1]:transitions_with_ends[ct]]),
                               np.std(data[transitions_with_ends[ct - 1]:transitions_with_ends[ct]])]

        errors[:, ct - 1] = [features[1, ct - 1] / np.sqrt(transitions_with_ends[ct] - transitions_with_ends[ct - 1] - 1 + 1e-10),
                             features[1, ct - 1] / np.sqrt(2 * (transitions_with_ends[ct] - transitions_with_ends[ct - 1] - 1) + 1e-10)]

        stiffnesses[ct - 1] = np.diag(errors[:, ct - 1] ** -2)

    transitions = np.concatenate(([0], original_data_mapping[transitions_with_ends[1:-1]], [len(data)]))
    transitions = np.sort(transitions)

    return transitions, features, errors, stiffnesses


def find_transitions(left, right, x, xsq, min_level_length, sensitivity):
    # global x, xsq,  min_level_length, cpic_multiplier
    cpic_multiplier = sensitivity
    # 默认情况下，没有转变点
    transitions = []

    # 计算区间长度和候选点
    N_T = right - left + 1
    N_L = np.arange(min_level_length, N_T - min_level_length + 1)
    N_R = N_T - N_L

    # 如果候选点为空，直接返回
    if len(N_L) == 0:
        return transitions

    # 计算均值和方差
    x_mean_L = (x[left + N_L - 1] - x[left - 1]) / N_L
    x_mean_R = (x[right] - x[right - N_R]) / N_R
    x_mean_T = (x[right] - x[left - 1]) / N_T

    xsq_mean_L = (xsq[left + N_L - 1] - xsq[left - 1]) / N_L
    xsq_mean_R = (xsq[right] - xsq[right - N_R]) / N_R
    xsq_mean_T = (xsq[right] - xsq[left - 1]) / N_T

    var_L = np.maximum(xsq_mean_L - x_mean_L**2, 0.00002)
    var_R = np.maximum(xsq_mean_R - x_mean_R**2, 0.00002)
    var_T = np.maximum(xsq_mean_T - x_mean_T**2, 0.00002)

    # 计算 CPIC 惩罚
    if N_T >= 1e6:
        p_CPIC = high_n_cpic(1e6)
    elif 1000 < N_T < 1e6:
        p_CPIC = high_n_cpic(N_T)
    else:
        p_CPIC = low_n_cpic(N_T)

    # 计算 CPIC 总值
    CPIC = 0.5 * (N_L * np.log(var_L) + N_R * np.log(var_R) - N_T * np.log(var_T)) + 1 + cpic_multiplier * p_CPIC

    # 找到最佳转变点
    minCPIC = np.min(CPIC)
    wheremin = np.argmin(CPIC)

    # 修正索引
    min_index = wheremin + min_level_length + left - 2

    # CPIC < 0 表示编码更好
    if minCPIC < 0:
        transitions = ([min_index] + find_transitions(left, min_index,x, xsq, min_level_length, sensitivity) +
                       find_transitions(min_index + 1, right, x, xsq, min_level_length, sensitivity))

    return transitions