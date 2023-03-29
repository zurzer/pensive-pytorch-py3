import os


# TRAIN_TRACES = './data/cooked_traces/'  # 原tensorflow版本Pensive工程的网速文件
# TRAIN_TRACES = './dataset/video_trace/'
# TEST_TRACES = './dataset/network_trace/'
COOKED_TRACE_FOLDER = './network_trace/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []

    for cooked_file in cooked_files:  # 便利网速文件夹
        sub_cooked_file = os.listdir(cooked_trace_folder + cooked_file + '/')
        for subfile in sub_cooked_file:  # 便利网速文件
            file_path = cooked_trace_folder + cooked_file + '/' + subfile
            cooked_time = []
            cooked_bw = []
            # print file_path
            with open(file_path, 'rb') as f:  # 添加： 时刻 网速
                for line in f:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))  # s
                    cooked_bw.append(float(parse[1]))  # mbps
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)  # 文件（夹）名，未使用

    return all_cooked_time, all_cooked_bw, all_file_names
