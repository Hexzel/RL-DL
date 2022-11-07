from torch.distributions import Categorical
import gym
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

gamma = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 600


class Pi(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                # nn.Linear(64, 32),
                # nn.ReLU(),
                nn.Linear(64, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()
    
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
        
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        x = x.to(DEVICE) # cuda
        pdparam = self.forward(x)
        pd = Categorical(logits = pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        log_prob = log_prob.to(DEVICE) # cuda
        self.log_probs.append(log_prob)
        
        return action.item()

def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype = np.float32)
    future_ret = 0.0
    
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    
    rets = torch.tensor(rets)
    rets = rets.to(DEVICE) # cuda
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def main():
    env = gym.make('CartPole-v1', render_mode='human', new_step_api=True)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    print(f"in dim: {in_dim}, out dim: {out_dim}")
    pi = Pi(in_dim, out_dim)
    pi = pi.to(DEVICE) # cuda
    optimizer = optim.Adam(pi.parameters(), lr = 0.015)
    
    history = [False, False, False, False, False]
    
    try:
        for epi in range(N_EPOCHS):
            state = env.reset()
            for t in range(500):
                action = pi.act(state)
                state, reward, done, _, _ = env.step(action)
                # if (t == 100):
                #     print("[action, state, reward]", [action, state, reward])
                pi.rewards.append(reward)
                # env.render()
                
                if done:
                    if (t < 499):
                        pi.rewards[-1] = -15
                    break
            
            
            total_reward = sum(pi.rewards)
            solved = total_reward > 495.0
            
            history = history[1:] + [solved]
            if not(all(history)):
                pi.train()
                loss = train(pi, optimizer)
            else:
                loss = -1
                pi.eval()
                print("No train")
            
            
            pi.onpolicy_reset()
            print(f'Episode {epi}, loss: {loss}\
                  \ntotal_reward: {total_reward}, solved: {solved}\n')
    
    finally:
        env.close()
    
    

if __name__ == '__main__':
    main()