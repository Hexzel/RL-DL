import torch
import gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.99
EPSILON = (0.9, 0.05)
N_EPOCHS = 500

class policy(nn.Module):
    
    def __init__(self, state_dim, act_space, epsilon_pair = (0.9, 0.05), gamma = 0.99):
        super(policy, self).__init__()
        
        self.act_dim = len(act_space)
        
        layers = [
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(64, self.act_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.act_space = act_space
        self.epi_start, self.epi_end = epsilon_pair
        self.epi = self.epi_start
        self.gamma = gamma
        self.memory_reset()
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        
        
    def memory_reset(self):
        self.q_values = []
        self.rewards = []
        self.actions = []
        self.greedy_rate = 0
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def act(self, state): 
        state = torch.from_numpy(state)
        q_value = self.forward(state)
        self.q_values.append(q_value)
            
        if np.random.rand() < self.epi:
            act_idx = np.random.choice(self.act_dim)
        
        else:
            self.greedy_rate +=1
            act_idx = q_value.argmax().item()
    
        self.actions.append(act_idx)
        return self.act_space[act_idx]
    
    def decay_epsilon(self, portion):
        self.epi = self.epi_end + (self.epi_start - self.epi_end) * np.exp(portion * (-3))


def train_sara(pi, optimizer):
    pi.train()
    q_values = torch.stack(pi.q_values)
    rewards = torch.tensor(pi.rewards)
    actions = torch.tensor(pi.actions)
    actions = actions.view(-1, 1)
    q_values = q_values.gather(1, actions)
    with torch.no_grad():
        target_q = torch.zeros_like(q_values)
        target_q[:-1] = q_values[1:]
        target_q[-1] = 0.0
        target_q = target_q * pi.gamma + rewards.view(*target_q.shape)
        target_q = target_q.detach()
    
    
    optimizer.zero_grad()
    loss = pi.loss(q_values, target_q)
    loss.backward()
    optimizer.step()
    
    pi.memory_reset()

    return loss


def main():
    env = gym.make('CartPole-v0', render_mode='human')
    act_space = np.arange(env.action_space.n)
    state_dim = env.observation_space.shape[0]
    pi = policy(state_dim, act_space, epsilon_pair = EPSILON, gamma = GAMMA)
    optimizer = torch.optim.RMSprop(pi.parameters(), lr = 0.01)
    total_reward = 0
    reward_history = []
    loss_history = []
    
    
    try:
        for epi in range(N_EPOCHS):
            state = env.reset()
            for t in range(200):
                action = pi.act(state)
                state, reward, done, _ = env.step(action)
                pi.rewards.append(reward)
                env.render()
                
                if done:
                    if (t == 199):
                        solved = True
                    else:
                        solved = False
                        pi.rewards[-1] = -5
                    
                    total_reward = sum(pi.rewards)
                    break
                
            loss = train_sara(pi, optimizer)
            pi.decay_epsilon(epi / N_EPOCHS)
            print(f'Episode {epi}, loss: {loss}\
                \ntotal_reward: {total_reward}, solved: {solved}\n')
            
            reward_history.append(total_reward)
            loss_history.append(loss.item())
    finally:    
        env.close()
        reward_history = np.array(reward_history)
        loss_history = np.array(loss_history)
        epochs = np.arange(reward_history.shape[0])
        plt.plot(epochs, reward_history, color = (0, 0, 1))
        plt.plot(epochs, loss_history, '--', color = (1, 0, 0))
        plt.show()
        
if __name__ == '__main__':
    main()         