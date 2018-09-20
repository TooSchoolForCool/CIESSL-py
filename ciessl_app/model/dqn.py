# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import gym


target_replace_iter = 5
memory_capacity = 5
lr = 0.01
batch_size = 4
epsilon = 0.9
gamma = 0.9

class q_net(nn.Module):
    def __init__(self, n_states, n_actions, hidden=50):
        super(q_net, self).__init__()
        if hidden < n_states:
            hidden = 2 * n_states
        
        self.fc = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_actions)
        )
        
        nn.init.normal(self.fc[0].weight, std=0.1)
        nn.init.normal(self.fc[2].weight, std=0.1)
        
    def forward(self, x):
        actions_value = self.fc(x)
        return actions_value


class DQN(object):
    def __init__(self, n_states, n_actions):
        self.eval_net = q_net(n_states=n_states, n_actions=n_actions)
        self.target_net = q_net(n_states=n_states, n_actions=n_actions)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # 当前的状态和动作，之后的状态和动作
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.n_states_ = n_states
        self.n_actions_ = n_actions
 
    def choose_action(self, s):
        n_actions = self.n_actions_
        s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))
        # input only one sample
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, n_actions)
        return action

    def action_ranking(self, s):
        s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))
        # input only one sample
        if np.random.uniform() < epsilon:
            ranking = self.eval_net(s).data.numpy()
        else:
            ranking = np.arange( self.n_actions )
            rs = np.random.RandomState()
            rs.shuffle(ranking)

        return ranking

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        n_states = self.n_states_

        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :n_states]))
        b_a = Variable(
            torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int)))
        b_r = Variable(
            torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -n_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(
            b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.criterion(q_eval, q_target)

        # update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    dqn_trainer = DQN(n_states=n_states, n_actions=n_actions)

    print('collecting experience ... ')
    all_reward = []
    for i_episode in range(300):
        s = env.reset()
        reward = 0
        while True:
            # env.render()
            a = dqn_trainer.choose_action(s)

            # 环境采取动作得到结果
            s_, r, done, info = env.step(a)

            # 修改奖励以便更快收敛
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn_trainer.store_transition(s, a, r, s_)

            reward += r
            print(dqn_trainer.memory_counter)
            if dqn_trainer.memory_counter > memory_capacity: # 记忆收集够开始学习
                dqn_trainer.learn()
                if done:
                    print('Ep: {} | reward: {:.3f}'.format(i_episode, round(reward, 3)))
                    all_reward.append(reward)
                    break

            if done:
                break
            s = s_


if __name__ == '__main__':
    main()