import sys
import torch
import load_trace
import numpy as np
import fixed_env as env
from Network import ActorNetwork
from torch.distributions import Categorical


STATE_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
STATE_LEN = 8  # take how many frames in the past
ACTION_DIM = 6
ACTOR_LR_RATE = 0.00001
CRITIC_LR_RATE = 0.0001
# VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  
VIDEO_BIT_RATE = [200, 380, 600, 900, 1600, 4000]  # kbps
BUFFER_NORM_FACTOR = 600.0
BANDWIDTH_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 720.0
M_IN_K = 1000.0
REBUF_PENALTY = 3.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 0.2
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
# RAND_RANGE = 1000
# log_sim_rl_N  {time_stamp / M_IN_K}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size}\t{rebuf}\t{video_chunk_size}\t{delay}\t{reward}\n
LOG_FILE = './results/test/log_test_network'
TEST_TRACES = './qdata/network_trace/test/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# frame_time_len = 0.04
ACTOR_MODEL=sys.argv[1]

def main():
    torch.set_num_threads(1)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == ACTION_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    # all models have same actor network
    # so model_type can be anything
    net=ActorNetwork([STATE_INFO, STATE_LEN], ACTION_DIM)

    # restore neural net parameters
    net.load_state_dict(torch.load(ACTOR_MODEL))
    print("Testing model restored.")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    video_count = 0
    state = torch.zeros((STATE_INFO,STATE_LEN))

    while True:  # serve video forever
        # the action is from the last decision, this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(f'{time_stamp / M_IN_K}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size}\t{rebuf}\t{video_chunk_size}\t{delay}\t{reward}\n')
        log_file.flush()

        # retrieve previous state
        state = torch.roll(state,-1,dims=-1)

        # this should be STATE_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / BANDWIDTH_NORM_FACTOR  # MB/s
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # sec
        state[4, :ACTION_DIM] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            probability = net.forward(state.unsqueeze(0))
            m = Categorical(probability)
            bit_rate = m.sample().item()
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            state=torch.zeros((STATE_INFO,STATE_LEN))

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
