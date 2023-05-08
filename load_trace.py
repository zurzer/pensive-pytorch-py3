import os

# avgspeed mbps ： 1.5-2.5(5), 2.5-4(6), 4-6(7), 6-8(8), 8-16(9), 16-32(10), 32-64(11), 64-128(12), 128-256(13)
# TEST_TRACES = './qdata/network_trace/test/'
TRAIN_TRACES = './qdata/network_trace/train/'

# 1629328	-1	1140736	5601	0	0	5629
# bandwidth = -1 没有此过程，down_time = -1 异常值或者没有cdn下载过程，P2P起始速度慢不影响
# {cdn_bandwidth}\t{p2p_bandwidth}\t{cdn_size}\t{cdn_time}\t{p2p_size}\t{p2p_time}\t{down_time}
def load_trace(cooked_trace_folder=TRAIN_TRACES):
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
            with open(file_path) as f:  # 添加： 时刻 网速
                fix_client_time = 0
                last_client_time = 0
                for line in f:
                    client_time, cdn_bandwidth, p2p_bandwidth, cdn_size, cdn_time, p2p_size, p2p_time, down_time = line.strip().split('\t')
                    client_time = int(client_time)
                    cdn_bandwidth = int(cdn_bandwidth)
                    cdn_size = int(cdn_size)
                    cdn_time = int(cdn_time)
                    down_time = int(down_time)
                    if cdn_bandwidth > 0:
                        if cdn_time > 0 and down_time > cdn_time:
                            cdn_bandwidth = int(8000 * cdn_size / cdn_time)
                        cooked_time.append(fix_client_time / 1000.0)  # ms to (s)
                        cooked_bw.append(cdn_bandwidth / 1000000.0)  # bps to (mbps)
                        fix_client_time += min(10000 if last_client_time == 0 else 30000, client_time - last_client_time)
                    last_client_time = client_time
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)  # 文件（夹）名，未使用

    return all_cooked_time, all_cooked_bw, all_file_names
