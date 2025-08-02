import numpy as np
import cmath

def fft_240(data):
    #输入数据重排
    reordered_data = data_reorder(data)

    #重排后输入数据分为15组，每组16点
    slice_size1 = 16
    num_slices1 = 15
    output1_components = []

    #循环15次，对每组16个点做16点radix2_fft
    for i in range(num_slices1):
        start_index = i * slice_size1
        end_index = start_index + slice_size1
        segment = reordered_data[start_index:end_index]  # 截取当前分段
        processed_segment = radix2_16points_computation(segment)  # 处理分段
        output1_components.append(processed_segment)  # 存储结果

    #15组16点radix2_fft结果存入output1
    output1 = sum(output1_components, [])

    #output1分为5组，每组48点
    slice_size2 = 48
    num_slices2 = 5
    output2_components = []

    #循环5次，每次对一组的48点做48点radix3_fft(16×3)
    for i in range(num_slices2):
        start_index = i * slice_size2
        end_index = start_index + slice_size2
        segment = output1[start_index:end_index]  # 截取当前分段
        processed_segment = radix3_48points_computation(segment)  # 处理分段
        output2_components.append(processed_segment)  # 存储结果

    #5组48点(16×3)radix3_fft结果存入output2
    output2 = sum(output2_components, [])

    #做240点(48×5)radix5_fft
    output3 = radix5_240points_computation(output2)

    return output3

#输入数据重排
def data_reorder(input):
    output1 = data_reorder_radix5(input)
    output2_components = []  # 存储处理后的分段结果
    slice_size1 = 48  # 每个分段的长度
    num_slices1 = 5  # 分段数量

    for i in range(num_slices1):
        start_index = i * slice_size1
        end_index = start_index + slice_size1
        segment = output1[start_index:end_index]  # 截取当前分段
        processed_segment = data_reorder_radix3(segment)  # 处理分段
        output2_components.append(processed_segment)  # 存储结果

    output2 = sum(output2_components, [])  # 拼接所有分段结果

    output3_components = []
    slice_size2 = 16
    num_slices2 = 15

    for j in range(num_slices2):
        start_index = j * slice_size2
        end_index = start_index + slice_size2
        segment = output2[start_index:end_index]  # 截取当前分段
        processed_segment = data_reorder_radix2(segment)  # 处理分段
        output3_components.append(processed_segment)  # 存储结果

    output3 = sum(output3_components, [])

    return output3

def data_reorder_radix5(input):
    idx1 = []
    idx2 = []
    idx3 = []
    idx4 = []
    idx5 = []

    for i in range(48):
        idx1.append(5 * i)
        idx2.append(5 * i + 1)
        idx3.append(5 * i + 2)
        idx4.append(5 * i + 3)
        idx5.append(5 * i + 4)

    idx = idx1 + idx2 + idx3 + idx4 + idx5

    output = [input[i] for i in idx]
    return output

def data_reorder_radix3(input):
    idx1 = []
    idx2 = []
    idx3 = []

    for i in range(16):
        idx1.append(3 * i)
        idx2.append(3 * i + 1)
        idx3.append(3 * i + 2)

    idx = idx1 + idx2 + idx3

    output = [input[i] for i in idx]
    return output

def data_reorder_radix2(input):
    bit_reverse_table = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    output = [input[i] for i in bit_reverse_table]
    return output

def radix2_16points_computation(data):
    N = 16
    stages = 4

    # 旋转因子 tf(twiddle_factor)
    tf = [cmath.exp(-1j * 2 * cmath.pi * k / N) for k in range(N//2)]

    for s in range(stages):  # N = 16，总共分为四级计算
        span = 2 ** s  # 蝶形运算跨度(相隔多远的两个数据做蝶形运算)，第0级为1，第1级为2，第2级为4，第3级为8
        step = 2 * span  # 步长，控制每个stage中，共有多少次蝶形运算
        for i in range(0, N, step):
            for j in range(span):
                # 旋转因子索引
                tf_idx = j * (2 ** (3 - s))

                # 得到旋转因子
                W = tf[tf_idx]

                # 输入数据索引
                data_idx_1 = i + j
                data_idx_2 = data_idx_1 + span

                a = data[data_idx_1]
                b = data[data_idx_2]


                data[data_idx_1] = a + W * b
                data[data_idx_2] = a - W * b

    return data

#将3组16点DFT结果组合成48点
def radix3_48points_computation(data):
    N = 48

    #拆分成三组
    X0 = data[0:16]
    X1 = data[16:32]
    X2 = data[32:48]

    for k in range(16):
        data[k] = X0[k] + cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k]
        data[k+16] = X0[k] + cmath.exp(-1j * 2 * cmath.pi / 3) * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[
            k] + cmath.exp(-1j * 4 * cmath.pi / 3) * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * \
                  X2[k]
        data[k+32] = X0[k] + cmath.exp(-1j * 4 * cmath.pi / 3) * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[
            k] + cmath.exp(-1j * 2 * cmath.pi / 3) * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * \
                       X2[k]

    return data

#将五组48点DFT结果组合成240点
def radix5_240points_computation(data):
    N = 240

    #拆分成五组
    X0 = data[0:48]
    X1 = data[48:96]
    X2 = data[96:144]
    X3 = data[144:192]
    X4 = data[192:240]

    W5 = []
    for m in range(1,5):
        W5.append(cmath.exp(-1j * 2 * cmath.pi * m / 5))

    for k in range(48):
        data[k] = (X0[k] + cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k] +
                   cmath.exp(-1j * 2 * cmath.pi * 3 * k / N) * X3[k] + cmath.exp(-1j * 2 * cmath.pi * 4 * k / N) * X4[k])
        data[k + 48] = (X0[k] + W5[0] * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + W5[1] * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k] +
                   W5[2] * cmath.exp(-1j * 2 * cmath.pi * 3 * k / N) * X3[k] + W5[3] * cmath.exp(-1j * 2 * cmath.pi * 4 * k / N) * X4[k])
        data[k + 96] = (X0[k] + W5[1] * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + W5[3] * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k] +
                   W5[0] * cmath.exp(-1j * 2 * cmath.pi * 3 * k / N) * X3[k] + W5[2] * cmath.exp(-1j * 2 * cmath.pi * 4 * k / N) * X4[k])
        data[k + 144] = (X0[k] + W5[2] * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + W5[0] * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k] +
                   W5[3] * cmath.exp(-1j * 2 * cmath.pi * 3 * k / N) * X3[k] + W5[1] * cmath.exp(-1j * 2 * cmath.pi * 4 * k / N) * X4[k])
        data[k + 192] = (X0[k] + W5[3] * cmath.exp(-1j * 2 * cmath.pi * k / N) * X1[k] + W5[2] * cmath.exp(-1j * 2 * cmath.pi * 2 * k / N) * X2[k] +
                   W5[1] * cmath.exp(-1j * 2 * cmath.pi * 3 * k / N) * X3[k] + W5[0] * cmath.exp(-1j * 2 * cmath.pi * 4 * k / N) * X4[k])

    return data

if __name__ == "__main__":

    input = [i for i in range(240)]
    fft_result_custom = fft_240(input)
    fft_result_function = np.fft.fft(input)
    # print(input)
    print(fft_result_custom)
    print(fft_result_function)