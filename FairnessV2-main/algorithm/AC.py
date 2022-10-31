import torch
import numpy as np
import torch.nn.functional as F


device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim_, action_dim_):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim_   # 骑手当前的状态
        self.action_dim = action_dim_  # 骑手收完订单后的状态
        self.S = torch.nn.Linear(state_dim_, 32).to(device)
        self.A = torch.nn.Linear(action_dim_, 8).to(device)
        self.l1 = torch.nn.Linear(32 + 8, 16).to(device)
        self.l2 = torch.nn.Linear(16, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s1 = X[:, :self.state_dim]
        a1 = X[:, -self.action_dim:]
        S1 = F.relu(self.S(s1))
        A1 = F.relu(self.A(a1))
        Y1 = torch.cat((S1, A1), dim=1)
        l1 = F.relu(self.l1(Y1))
        l2 = F.relu(self.l2(l1))
        return F.relu(self.f(l2)) + 1


# critic
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim_,action_dim_):
        super(ValueNet, self).__init__()
        self.state_dim = state_dim_  # 骑手当前的状态
        self.action_dim = action_dim_  # 骑手收完订单后的状态
        self.S = torch.nn.Linear(state_dim_, 32).to(device)
        self.A = torch.nn.Linear(action_dim_, 8).to(device)
        self.l1 = torch.nn.Linear(32 + 8, 16).to(device)
        self.l2 = torch.nn.Linear(16, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s1 = X[:, :self.state_dim]
        a1 = X[:, -self.action_dim:]
        S1 = F.relu(self.S(s1))
        A1 = F.relu(self.A(a1))
        Y1 = torch.cat((S1, A1), dim=1)
        l1 = F.relu(self.l1(Y1))
        l2 = F.relu(self.l2(l1))
        return F.relu(self.f(l2)) + 1

class ActorCritic:
    def __init__(self, state_dim, action_dim_, actor_lr_, critic_lr_, gamma_, device_):
        self.actor = PolicyNet(state_dim, action_dim_).to(device_)
        self.critic = ValueNet(state_dim,action_dim_).to(device_)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr_)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr_)
        self.gamma = gamma_

    # actor:采取动作
    def take_action(self, _state):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        v_output = v_output.reshape(-1)  # 将二维张量变为一维张量
        action_prob = torch.softmax(v_output, dim=0)
        action_dist = torch.distributions.Categorical(action_prob)
        c_id_ = action_dist.sample().cpu()
        # c_id_ = action_dist.sample()
        # print(c_id_)
        return c_id_.item(), np.array(action_prob.detach())  # 对softmax函数求导时的用法


    def take_next_action(self, _state):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        v_output = v_output.reshape(-1)  # 将二维张量变为一维张量
        action_prob = torch.softmax(v_output, dim=0)
        c_id_ = torch.argmax(action_prob)
        # c_id_ = action_dist.sample()
        # print(c_id_)
        return c_id_.item()  # 对softmax函数求导时的用法

    def take_action_epsilon(self, _state, epsilon):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        softmax_V = torch.softmax(v_output, dim=0)
        c_id = np.argmax(np.array(v_output.cpu().detach().numpy()))
        # epsilon-greedy 策略
        action_prob = np.ones(len(_state))
        action_prob = action_prob * (1 - epsilon) / (len(_state))
        action_prob[c_id] += epsilon
        c_id_ = np.argmax(np.random.multinomial(1, action_prob))
        return c_id_.item(), softmax_V.detach()

    def update(self, state,reward,next_state, policy):
        state = torch.tensor(state, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        policy = torch.tensor(policy, dtype=torch.float, requires_grad=True).to(device)  # 对应的softmax_V_output

        td_target = torch.unsqueeze(reward, 1) + self.gamma * self.critic(next_state).to(device)
        V = self.critic(state)
        critic_loss = torch.mean(F.mse_loss(V, td_target.detach())).to(device)  # TD error
        td_delta = td_target - V  # 时序差分误差 TD_error
        log_prob = torch.log(policy)  # softmax后对应的选的action那一值组成的tensor, requires_grad=True
        actor_loss = torch.mean(-log_prob * td_delta.detach()).to(device)

        self.critic_optimizer.zero_grad()
        #self.actor_optimizer.zero_grad()
        critic_loss.backward()  # 计算critic网络的梯度
        #actor_loss.backward()  # 计算actor网络的梯度
        self.critic_optimizer.step()  # 更新critic网络参数
        #self.actor_optimizer.step()  # 更新actor网络参数

