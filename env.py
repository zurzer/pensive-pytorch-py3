import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer  固定每个分段时长
BITRATE_LEVELS = 6  # 可选码率个数
TOTAL_VIDEO_CHUNCK = 48  # 总分段个数
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit  触发下载休眠的最大缓存量
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec  每次休眠时长
PACKET_PAYLOAD_PORTION = 0.95  # 网速有效载荷
LINK_RTT = 80  # millisec  连接耗时
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9  # 下载耗时波动率下限
NOISE_HIGH = 1.1  # 下载耗时波动率上限
VIDEO_SIZE_FILE = './data/video_size_'  # 分段大小文件


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):  # 初始化网速列表和各码率的分段大小列表，随机初始化网速起始idx
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        # 多个网速文件list
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0  # 视频段起点
        self.buffer_size = 0  # 缓存

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))  # 随机选一个网速文件
        self.cooked_time = self.all_cooked_time[self.trace_idx]  # 网速对应时刻列表
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]  # 网速列表

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))  # 随机算一个网速列表开始时刻idx
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]  # 上一个时刻，算时间间隔

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):  # 第几档码率，INT
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))  # 码率对应的分段文件大小，每个分段固定长度VIDEO_CHUNCK_LEN=4s

    def get_video_chunk(self, quality):  # quality：INT第几档码率，下载完当前chunk就返回，TODO：码率档位可能不够

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]  # 分段大小

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms  下载耗时
        video_chunk_counter_sent = 0  # in bytes  已经发给播放器的大小

        while True:  # download video chunk over mahimahi
            # 速度 和 当前速度区间的间隔时长，TODO：当前没有下载速度怎么处理
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # 下载大小

            if video_chunk_counter_sent + packet_payload > video_chunk_size:  # 当前分段会在本速度区间下载完毕
                # 实际下载耗时
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time  # 下载耗时
                self.last_mahimahi_time += fractional_time  # 结束时刻
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload  # 下载的数据发给播放器
            delay += duration  # 下载耗时
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]  # 结束时刻
            self.mahimahi_ptr += 1  # 进入下一个速度区间

            if self.mahimahi_ptr >= len(self.cooked_bw):  # 循环速度文件
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1  # idx为0的话，获取不到此刻速度区间的耗时，速度列表文件第一个速度值需要丢弃，保留第一个时刻值
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT  # 下载耗时加连接耗时

    # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)  # 下载耗时叠加波动率
        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)  # 卡顿时长，当前分段下载耗时 减去 已有缓存时长，TODO：认为一个分段下载完毕才能去播放，导致开播的时候一定会卡顿

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)  # 已有缓存时长，减去 下载耗时，余留缓存时长

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN  # 缓存时长，加上 分段时长，TODO：固定值改为变动值

        # sleep if buffer gets too large
        sleep_time = 0  # 下载休眠时长
        if self.buffer_size > BUFFER_THRESH:  # 缓存过大触发下载休眠，TODO：进入和退出休眠用不同的值
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH  # 多出的过大缓存量
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME  # 下载休眠时长
            self.buffer_size -= sleep_time  # 下载休眠期间消耗的缓存时长

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:  # 跳过休眠期间的网速列表
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND  # 修改上次网速结束时间点 匹配 休眠结束时间点
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND  # 休眠期时间减去网速区间时长
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1  # chunk idx 加1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter  # 余留chunk个数

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:  # chunk播放完毕，参数初始化回归
            end_of_video = True  # 播放结束标志置位

            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])  # 下一个分段的各码率的大小

        # 下载耗时，休眠时长，缓存时长，卡顿时长，当前播放chunk大小，下一个chunk的各码率的大小，播放是否结束，余留chunk个数
        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain

    def setEnvironmentPtr(self, ptrTraceIndex, ptrIndex):
        # 选择某个网速文件
        self.trace_idx = ptrTraceIndex
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.video_chunk_counter = 0  # 起始chunk idx
        self.buffer_size = 0  # 缓存时长

        # randomize the start point of the trace
        # note: trace file starts with time 0
        # 选择网速列表的起始idx，TODO：ptrIndex需要>0
        self.mahimahi_ptr = ptrIndex % len(self.cooked_bw)
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
