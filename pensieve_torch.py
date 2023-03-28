# from multiprocessing import Barrier
import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import env
import load_trace
import torch
from A3C import A3C
from datetime import datetime


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
STATE_INFO = 6  # 状态值的个数
STATE_LEN = 8  # take how many frames in the past  用于算法判断的历史状态个数
ACTION_DIM = 6  # 可选码率个数
ACTOR_LR_RATE = 0.0001  # 演员训练学习率
CRITIC_LR_RATE = 0.001  # 观察家训练学习率
NUM_AGENTS = 4  # 演员个数
TRAIN_SEQ_LEN = 100  # take as a train batch  最多一次进行100个chunk的播放训练
MODEL_SAVE_INTERVAL = 100  # 每100次保存一下模型参数
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps  可选码率列表
# VIDEO_BIT_RATE = [500, 850, 1200, 1850, 0, 0]
# HD_REWARD = [1, 2, 3, 12, 15, 20]  # 激励函数，码率转化为梯度的激励值
BUFFER_NORM_FACTOR = 10.0  # 缓存时长归一化分母
CHUNK_TIL_VIDEO_END_CAP = 48.0  # chunk位置idx归一化分母
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps  卡顿惩罚系数
SMOOTH_PENALTY = 1  # 码率平滑度，码率切换惩罚系数
DEFAULT_QUALITY = 1  # default video quality without agent  默认起始码率
RANDOM_SEED = 42  # 随机数初始化种子
# RAND_RANGE = 1000

SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
# TRAIN_TRACES = './data/cooked_traces/'
TRAIN_TRACES = './dataset/video_trace/'

CRITIC_MODEL = None  # 不保存中间结果的观察家训练参数
# CRITIC_MODEL= './results/critic.pt'
ACTOR_MODEL = './results/actor.pt'
TOTALEPOCH = 30000  # 总训练轮数

IS_CENTRAL = True  # 观察家模式
NO_CENTRAL = False  # 演员模式


def testing(epoch, actor_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)

    # run test script  系统层面调用rl_test.py
    os.system(f'python rl_test.py {actor_model}')

    # append test performance to the log  获取rl_test.py的测试运行结果日志文件获取奖励系数
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.strip().split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards) # 最小值
    rewards_5per = np.percentile(rewards, 5)  # 5%分位值
    rewards_mean = np.mean(rewards)  # 均值
    rewards_median = np.percentile(rewards, 50)  # 50%分位值，中值
    rewards_95per = np.percentile(rewards, 95)  # 95%分位值
    rewards_max = np.max(rewards)  # 最大值

    log_file.write(f'{epoch}\t{rewards_min}\t{rewards_5per}\t{rewards_mean}\t{rewards_median}\t{rewards_95per}\t{rewards_max}\n')
    log_file.flush()


def central_agent(net_params_queues, exp_queues, model_type, barrier):
    # torch.set_num_threads(1)
    # barrier.wait()

    timenow = datetime.now()
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central', filemode='w', level=logging.INFO)

    # print("Initial network")
    net = A3C(IS_CENTRAL, model_type, [STATE_INFO, STATE_LEN], ACTION_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)
    # print("Network inited.")

    test_log_file = open(LOG_FILE+'_test', 'w')

    # 加载训练好或训练中途的模型的参数列表
    if CRITIC_MODEL is not None and os.path.exists(ACTOR_MODEL):
        net.actorNetwork.load_state_dict(torch.load(ACTOR_MODEL))
        net.criticNetwork.load_state_dict(torch.load(CRITIC_MODEL))

    for epoch in range(TOTALEPOCH):
        print("Run epoch", epoch)
        # synchronize the network parameters of work agent
        actor_net_params = net.getActorParam()
        # critic_net_params=net.getCriticParam()
        for i in range(NUM_AGENTS):
            # net_params_queues[i].put([actor_net_params, critic_net_params])
            net_params_queues[i].put(actor_net_params)  # 由观察者来控制下发所有演员的参数
            # Note: this is synchronous version of the parallel training, which is easier to understand and probe.
            # The framework can be fairly easily modified to support asynchronous training.
            # Some practices of asynchronous training (lock-free SGD at its core) are nicely explained in the following two papers:
            # https://arxiv.org/abs/1602.01783
            # https://arxiv.org/abs/1106.5730

        # record average reward and td loss change in the experiences from the agents
        total_batch_len = 0.0
        total_reward = 0.0  # 奖励值
        # total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0  # 总演员个数

        # assemble experiences from the agents
        # actor_gradient_batch = []
        # critic_gradient_batch = []

        for i in range(NUM_AGENTS):
            state_batch, bitrate_batch, reward_batch, end_of_video, info = exp_queues[i].get()  # 依次从各演员处获取训练结果

            net.getNetworkGradient(state_batch, bitrate_batch, reward_batch, terminal=end_of_video)  # 进行观察家评判演员训练结果，修正演员参数

            total_reward += np.sum(reward_batch)  # 所有chunk的总奖励系数
            total_batch_len += len(reward_batch)  # chunk训练个数
            total_agents += 1.0  # 演员个数
            total_entropy += np.sum(info['entropy'])

        # log training information  进行下一步
        net.updateNetwork()
        # epoch += 1
        avg_reward = total_reward / total_agents  # 单个演员平均，所有chunk的总奖励系数
        avg_entropy = total_entropy / total_batch_len

        logging.info(f'Epoch: {epoch}, Avg_reward: {avg_reward}, Avg_entropy: {avg_entropy}')

        if (epoch+1) % MODEL_SAVE_INTERVAL == 0:
            # Save the neural net parameters to disk.
            print(f"\nTrain ep:{epoch+1}, time use :{(datetime.now()-timenow).seconds}s\n")
            timenow = datetime.now()
            torch.save(net.actorNetwork.state_dict(), SUMMARY_DIR+"/actor.pt")
            if model_type < 2:
                torch.save(net.criticNetwork.state_dict(), SUMMARY_DIR+"/critic.pt")
            
            testing(epoch+1, SUMMARY_DIR+"/actor.pt", test_log_file)  # 测试模型
        
        # Barrier thead after all thread finish  每轮训练都需要等所有演员训练结束
        if epoch == TOTALEPOCH - 1:
            barrier.wait()


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, model_type, barrier):
    torch.set_num_threads(1)  # 每个演员只能单核
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=agent_id)

    with open(LOG_FILE+'_agent_'+str(agent_id), 'w') as log_file:

        net = A3C(NO_CENTRAL, model_type, [STATE_INFO, STATE_LEN], ACTION_DIM, ACTOR_LR_RATE, CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator

        time_stamp = 0  # 模拟 运行时刻
        for epoch in range(TOTALEPOCH):
        # while True:
            actor_net_params = net_params_queue.get()  # 从观察家获取训练后的参数
            net.hardUpdateActorNetwork(actor_net_params)  # 强制更新net的actorNetwork.parameters()
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state_batch = []
            bitrate_batch = []
            reward_batch = []
            entropy_record = []
            state = torch.zeros((1, STATE_INFO, STATE_LEN))  # 状态列表

            # the action is from the last decision
            # this is to make the framework similar to the real
            # 下载完当前chunk就返回： 下载耗时，休眠时长，缓存时长，卡顿时长，当前播放chunk大小，下一个chunk的各码率的大小，播放是否结束，余留chunk个数
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms  运行时刻 加 下载耗时
            time_stamp += sleep_time  # in ms  运行时刻 加 休眠时长

            while not end_of_video and len(state_batch) < TRAIN_SEQ_LEN:  # chunk结束，或者播放训练大于100个chunk
                last_bit_rate = bit_rate
                state = state.clone().detach()  # 深拷贝复制状态，新建内存存储，然后脱离计算图，不提供梯度回溯计算
                state = torch.roll(state, -1, dims=-1)  # 行方向，横向左移一位

                # 行方向，各行的横向最后一列赋值
                state[0, 0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality  归一化，当前码率/最高码率
                state[0, 1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec  缓存时长/10秒
                # kilo byte / ms  下载速度 = 当前播放chunk大小/下载耗时
                state[0, 2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
                state[0, 3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec  下载耗时/10秒
                state[0, 4, :ACTION_DIM] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte  下一个chunk的各码率的大小。MB
                state[0, 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)  # 归一化，余留chunk个数/总chunk个数

                bit_rate = net.actionSelect(state)  # 选择下一个码率 INT

                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states
                # 下载下一个选择好的码率
                delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)
                # 计算激励函数，码率 减去 卡顿惩罚系数×卡顿时长 减去 平滑惩罚系数×码率变动量
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                state_batch.append(state)
                bitrate_batch.append(bit_rate)
                reward_batch.append(reward)
                entropy_record.append(3)  # TODO: why

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(f'{time_stamp}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size}\t{rebuf}\t{video_chunk_size}\t{delay}\t{reward}\n')
                log_file.flush()

            # ignore the first chuck since we don't have the control over it, TODO: do something
            exp_queue.put([state_batch, bitrate_batch, reward_batch, end_of_video, {'entropy': entropy_record}])
            log_file.write('\n')  # so that in the log we know where video ends

            # Barrier thead after all thread finish  每轮训练都需要等所有演员训练结束
            if epoch == TOTALEPOCH - 1:
                barrier.wait()


def parse_args():
    parser = argparse.ArgumentParser("Pensieve")
    # 仅解析model_type参数， 0 mean original， 1 mean critic_td， 2 mean only actor
    parser.add_argument("--model_type", type=int, default=0, help="Refer to README for the meaning of this parameter")
    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    start = datetime.now()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == ACTION_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []  # actor_net_params
    exp_queues = []  # state_batch, bitrate_batch, reward_batch, end_of_video, info
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue())
        exp_queues.append(mp.Queue())
        print("append in exp_queue: ", i)

    # 屏障，threading.Barrier 控制线程的作用，控制下限，即满足到一定数量才能进行
    # 每个线程通过调用 barrier.wait() 尝试通过 Barrier，如果不能通过，则阻塞，直到阻塞的线程数量达到了 parties 时候，阻塞的线程被全部释放。
    barrier = mp.Barrier(NUM_AGENTS+1)  # 调用wait后会等待所有线程都启动后，各线程才往wait以下运行

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues, arglist.model_type, barrier))
    coordinator.start()  # 先启动一个评论家进程

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(
        TRAIN_TRACES)  # 加载网速列表

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent, args=(i, all_cooked_time, all_cooked_bw, net_params_queues[i], exp_queues[i], arglist.model_type, barrier)))
        agents[i].start()  # 启动多个演员进程

    for i in range(NUM_AGENTS):
        agents[i].join()

    coordinator.join()  # 等待进程结束后再继续往下运行，用于进程间的同步

    print(str(datetime.now() - start))  # 输出总耗时
