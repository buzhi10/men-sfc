import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import rl_utils
from enviroment7 import MECenv
import matplotlib
import pickle
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = torch.nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = torch.nn.Linear(hidden_dim[1], action_dim)

    def forward(self, state, valid_actions):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.float32, device=state.device)
        elif isinstance(valid_actions, list):
            valid_actions = torch.tensor(valid_actions, dtype=torch.float32).to(state.device)
        else:
            valid_actions = valid_actions.float().to(state.device)

        res = x.min(dim=-1, keepdim=True)[0]
        x = x * valid_actions + (1 - valid_actions) * (-1e9 + res)

        x = x - x.max(dim=-1, keepdim=True)[0]
        x = F.softmax(x, dim=-1)
        x = x.squeeze(1)
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = torch.nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = torch.nn.Linear(hidden_dim[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, num_episodes):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.total_num_epochs = num_episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state, A):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state, A)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        valid_actions = torch.tensor(transition_dict['valid_actions'],
                                     dtype=torch.float).to(self.device)
        states = states.squeeze()
        next_states = next_states.squeeze()
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states, valid_actions).gather(1,
                                                                          actions)).detach()
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states, valid_actions).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def adjust_learning_rate(self, epoch):
        factor = (epoch / float(self.total_num_epochs)) ** (1 / 4)
        lr = self.actor_lr - (self.actor_lr * factor)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

        lr = self.critic_lr - (self.critic_lr * factor)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr

        return lr

actor_lr = 5e-6
critic_lr = 5e-5
num_episodes = 20000
hidden_dim = [32, 32]
gamma = 0.99
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = MECenv()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
state_dim = len(env.SFCs) * 4 + env.K * 3 + env.K * 9 + len(env.start_SFCs) + 15
action_dim = len(env.start_SFCs) * env.K

agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device, num_episodes)

return_list = []
iteration_count = 0

start_time = time.time()
print(f"Start training time: {start_time:.5f} seconds")
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            iteration_count += 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],
                               'valid_actions': []}
            state = env.reset()
            res = 0
            done = False
            while not done:
                res += 1
                valid_actons = env.valid_A_t()
                action = agent.take_action(state, valid_actons)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['valid_actions'].append(valid_actons)
                state = next_state
                episode_return += reward
            episode_return = episode_return
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


with open('return_list_20.pkl', 'wb') as f:
    pickle.dump(return_list, f)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format("SFCs"))
plt.savefig('PPO1_20.png')


mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format("SFCs"))
plt.savefig('PPO2_20.png')

