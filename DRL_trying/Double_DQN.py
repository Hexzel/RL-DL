import torch
import gym
import numpy as np
import torch.nn as nn
from collections import namedtuple, deque
import random
from SaveTool import StructToNetwork, SaveModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
EPSILON_PAIR = (0.9, 0.01)
N_EPOCHS = 1000
BATCH_SIZE = 128
N_UPDATES = 3
N_BATCHES_PER_EPOCH = 3
TAR_NET_UPDATE_PERIOD = 5

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class policy(nn.Module):
    
    def __init__(self, state_dim = 1, act_dim = 1, struct = None, 
                 epsilon_pair = (0.9, 0.05), gamma = 0.99):
        
        super(policy, self).__init__()
        
        if struct == None:
            self.struct = [
                ("Linear", (state_dim, 256)),
                ("ReLU",),
                ("Linear", (256, 128)),
                ("ReLU",),
                ("Linear", (128, 64)),
                ("ReLU",),
                ("Linear", (64, act_dim))
            ]
        else:
            self.struct = struct

        self.model = StructToNetwork(self.struct)
        self.act_dim = act_dim
        self.epi_start, self.epi_end = epsilon_pair
        self.epi = self.epi_start
        self.gamma = gamma
        self.loss = nn.MSELoss()
        
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    
    def act(self, state, mode = 'Greedy'): 
        with torch.no_grad():
            state = torch.tensor(state, device = DEVICE)
            q_value = self.forward(state)
        
        if (mode == 'Greedy'):
            if np.random.rand() < self.epi:
                act_idx = np.random.choice(self.act_dim)
            
            else:
                act_idx = q_value.argmax().item()
        
        elif (mode == 'Boltzmann'):
            pd = torch.distributions.Categorical(logits = q_value)
            act_idx = pd.sample().item()
        
        else:
            raise Exception("unknown mode for act: " + mode)
            
        return act_idx
    
    def decay_epsilon(self, portion):
        self.epi = self.epi_end + (self.epi_start - self.epi_end) * np.exp(portion * (-3))


def train_DQN(pi, tar_pi, optimizer, memory, n_updates = 1, double = False):
    if (len(memory) < BATCH_SIZE):
        return
    
    batch = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*batch))
    states = torch.from_numpy(np.stack(batch.state, axis = 0)).to(DEVICE)
    actions = torch.tensor(batch.action, device = DEVICE).view(-1, 1)
    
    
    with torch.no_grad():
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=DEVICE, dtype=torch.bool)
        
        non_final_states = torch.stack([torch.from_numpy(s) 
            for s in batch.next_state if s is not None]).to(DEVICE)

        rewards = torch.tensor(batch.reward, device = DEVICE)
    
    losses = np.zeros(n_updates)
    for i in range(n_updates):
        if double:
            q_values = pi.forward(states)
            with torch.no_grad():
                max_actions = q_values.max(1)[1].view(-1, 1)
                max_actions = max_actions[non_final_mask]
                q_values_tar = torch.zeros(BATCH_SIZE, device = DEVICE)
                q_values_tar[non_final_mask] = tar_pi.forward(
                    non_final_states).gather(1, max_actions).view(-1)
                q_values_tar = q_values_tar * tar_pi.gamma + rewards
                q_values_tar = q_values_tar.detach()
            q_values = q_values.gather(1, actions).view(-1) 
        else:
            q_values = pi.forward(states).gather(1, actions).view(-1)   
            with torch.no_grad():
                q_values_tar = torch.zeros(BATCH_SIZE, device = DEVICE)
                q_values_tar[non_final_mask] = tar_pi.forward(non_final_states).max(1)[0]
                q_values_tar = q_values_tar * tar_pi.gamma + rewards
                q_values_tar = q_values_tar.detach()
        
        loss = pi.loss(q_values, q_values_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
    
    return losses.mean()
    
    
    
        
def main():
    env = gym.make('CartPole-v1', render_mode = None, new_step_api = True)
    act_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    pi = policy(state_dim, act_dim, epsilon_pair = EPSILON_PAIR, gamma = GAMMA)
    pi.to(DEVICE)
    tar_pi = policy(state_dim, act_dim, epsilon_pair = EPSILON_PAIR, gamma = GAMMA)
    tar_pi.to(DEVICE)
    tar_pi.load_state_dict(pi.state_dict())
    optimizer = torch.optim.RMSprop(pi.parameters(), lr = 0.01)
    memory = ReplayMemory(4000)
    
    try:
        history = [False, False, False]
        for epi in range(N_EPOCHS):
            state = env.reset()
            total_reward = 0
            for t in range(500):
                action = pi.act(state, mode = 'Greedy')
                next_state, reward, done, _, _ = env.step(action)
                
                if done:   
                    if (t < 499):
                        reward = -15
                    
                    next_state = None
                    
                total_reward += reward
                memory.push(state, action, next_state, reward)
                state = next_state
                
                if done:
                    break
                
            solved = total_reward > 495
            history = history[1:] + [solved]

            print(history)
            losses = np.zeros(N_BATCHES_PER_EPOCH)
            
            if not all(history):
                for i in range(N_BATCHES_PER_EPOCH):
                    loss = train_DQN(pi, tar_pi, optimizer, 
                        memory, n_updates = N_UPDATES, double = True)
                    if (loss != None):
                        losses[i] = loss.item()
                
                
                pi.decay_epsilon(epi / N_EPOCHS)
            
            else:
                SaveModel(pi, 'CartPole-v1_double_DQN_Greedy.pth')
                print("save model")
                pi.decay_epsilon(2)
                
            if (epi % TAR_NET_UPDATE_PERIOD == 0):
                tar_pi.load_state_dict(pi.state_dict())
              
            loss_avg = losses.mean()
            print(f'Episode {epi}, loss: {loss_avg}\
                \ntotal_reward: {total_reward}, solved: {solved}\n')

    finally:    
        env.close()

def Replay():
    try:
        env = gym.make('CartPole-v1', render_mode = None, new_step_api = True)
        policy_info = torch.load('CartPole-v1_double_DQN_Greedy.pth')
        pi = policy(struct = policy_info['struct'], epsilon_pair = (0, 0))
        pi.load_state_dict(policy_info['state_dict'])
        pi.to(DEVICE)
        reward_history = np.zeros(100)
        
        for epi in range(100):
            state = env.reset()
            total_reward = 0
            for t in range(500):
                action = pi.act(state)
                next_state, reward, done, _, _ = env.step(action)
                
                if done:   
                    if (t < 499):
                        reward = -15
                    next_state = None
                
                total_reward += reward
                solved = total_reward > 495
                state = next_state
                
                if done:
                    break
            
            reward_history[epi] = total_reward
            print(f'Episode: {epi}, Reward: {total_reward}, solved: {solved}')
        
        else:
            print(f"Average reward of replay is {reward_history.mean()}")
        
    finally:
        env.close()
    
if __name__ == '__main__':
    
    # main() 
    Replay()