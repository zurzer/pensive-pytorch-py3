import torch
import torch.nn as nn
from torch.distributions import Categorical
from Network import (ActorNetwork, CriticNetwork)

PATH = './results/'


class A3C(object):
    def __init__(self, is_central, model_type, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3):
        self.state_dim = state_dim  # [STATE_INFO, STATE_LEN]
        self.action_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        # self.entropy_eps = 1e-6
        self.model_type = model_type
        self.is_central = is_central
        self.device = 'cpu'
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cuda:0')

        print("Load module")
        self.actorNetwork = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        # self.actorNetwork=ActorNetwork(self.state_dim,self.action_dim).cuda()

        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            # https://blog.csdn.net/weixin_39228381/article/details/108511882
            self.actorOptim = torch.optim.RMSprop(self.actorNetwork.parameters(), lr=actor_lr, alpha=0.9, eps=1e-10)
            self.actorOptim.zero_grad()
            if model_type < 2:
                '''
                model==0 mean original
                model==1 mean critic_td
                model==2 mean only actor
                '''
                self.criticNetwork = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
                # self.criticNetwork=CriticNetwork(self.state_dim,self.action_dim).cuda()
                self.criticOptim = torch.optim.RMSprop(self.criticNetwork.parameters(), lr=critic_lr, alpha=0.9, eps=1e-10)
                self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()  # evaluation mode

        self.loss_function = nn.MSELoss()

    def getNetworkGradient(self, state_batch, bitrate_batch, reward_batch, terminal):
        state_batch = torch.cat(state_batch).to(self.device)
        bitrate_batch = torch.LongTensor(bitrate_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).to(self.device)
        reward_batch = torch.zeros(reward_batch.shape).to(self.device)
        # state_batch=torch.cat(state_batch).cuda()
        # bitrate_batch=torch.LongTensor(bitrate_batch).cuda()
        # reward_batch=torch.tensor(reward_batch).cuda()
        # reward_batch=torch.zeros(reward_batch.shape).cuda()

        reward_batch[-1] = reward_batch[-1]
        for t in reversed(range(reward_batch.shape[0]-1)):
            reward_batch[t] = reward_batch[t] + self.discount*reward_batch[t+1]

        if self.model_type < 2:
            with torch.no_grad():
                v_batch = self.criticNetwork.forward(state_batch).squeeze().to(self.device)
                # v_batch=self.criticNetwork.forward(state_batch).squeeze().cuda()
            td_batch = reward_batch-v_batch
        else:
            td_batch = reward_batch

        probability = self.actorNetwork.forward(state_batch)
        m_probs = Categorical(probability)
        log_probs = m_probs.log_prob(bitrate_batch)
        actor_loss = torch.sum(log_probs*(-td_batch))
        entropy_loss = -self.entropy_weight*torch.sum(m_probs.entropy())
        actor_loss = actor_loss+entropy_loss
        actor_loss.backward()

        if self.model_type < 2:
            if self.model_type == 0:
                # original
                critic_loss = self.loss_function(reward_batch, self.criticNetwork.forward(state_batch).squeeze())
            else:
                # cricit_td
                v_batch = self.criticNetwork.forward(state_batch[:-1]).squeeze()
                next_v_batch = self.criticNetwork.forward(state_batch[1:]).squeeze().detach()
                critic_loss = self.loss_function(reward_batch[:-1]+self.discount*next_v_batch, v_batch)

            critic_loss.backward()

        # use the feature of accumulating gradient in pytorch

    def actionSelect(self, stateInputs):
        if not self.is_central:
            with torch.no_grad():  # 关闭梯度计算，确认不会调用Tensor.backward()时开启，可以减少计算所用内存消耗。
                stateInputs = stateInputs.to(self.device)
                # BW : model use here
                probability = self.actorNetwork.forward(stateInputs)
                # 创建以参数probs为标准的类别分布，样本是来自 “0 … K-1” 的整数，其中 K 是probs参数的长度。也就是说，按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。
                m = Categorical(probability)
                action = m.sample().item()
                return action

    def hardUpdateActorNetwork(self, actor_net_params):
        for target_param, source_param in zip(self.actorNetwork.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def updateNetwork(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            if self.model_type < 2:
                self.criticOptim.step()
                self.criticOptim.zero_grad()

    def getActorParam(self):
        return list(self.actorNetwork.parameters())

    def getCriticParam(self):
        return list(self.criticNetwork.parameters())


if __name__ == '__main__':
    # test maddpg in convid,ok
    SINGLE_STATE_LEN = 19
    AGENT_NUM = 1
    BATCH_SIZE = 200
    STATE_INFO = 6
    STATE_LEN = 8
    ACTION_DIM = 6
    discount = 0.9

    obj = A3C(False, 0, [STATE_INFO, STATE_LEN], ACTION_DIM)
    print("finish modelu")

    episode = 3000
    for i in range(episode):
        state = torch.randn(AGENT_NUM, STATE_INFO, STATE_LEN)
        action = torch.randint(0, 5, (AGENT_NUM,), dtype=torch.long)
        reward = torch.randn(AGENT_NUM)
        probability = obj.actionSelect(state)
        print(probability)
