import numpy as np
import cmath

def data_reorder(input):
    #按照radix5重排
    output1 = data_reorder_radix5(input)

    #按照radix3重排
    output2_components = []  # 存储处理后的分段结果
    slice_size1 = 48  # 每个分段的长度
    num_slices1 = 5  # 分段数量

    for i in range(num_slices1):
        start_index = i * slice_size1 #起始索引
        end_index = start_index + slice_size1
        segment = output1[start_index:end_index]  # 截取当前分段
        processed_segment = data_reorder_radix3(segment)  # 处理分段
        output2_components.append(processed_segment)  # 存储结果

    output2 = sum(output2_components, [])  # 拼接所有分段结果

    #按照radix2重排
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


if __name__ == "__main__":
    input = [i for i in range(240)]
    output3 = data_reorder(input)
    print(input)
    print(output3)