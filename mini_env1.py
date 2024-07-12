import gymnasium as gym
from gymnasium import spaces
import numpy as np
import plotly.express as px
import pandas as pd
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, machine_eff = False, energy_prices = False):
        super(CustomEnv, self).__init__()

        # Optimizer parameters
        self.machine_eff = machine_eff
        self.energy_prices = energy_prices

        # History of action for plot
        self.history = []
        self.prices_fx = []
        self.step_count = 0

        # Append prices:
        self.prices_fx.extend([0,0,0,0,1,1,1,1])
        self.prices_fx.extend([1,1,1,1,0,0,0,0])
        self.prices_fx.extend([0,0,0,0,0,0,-1,-1])
        self.prices_fx.extend([-1,-1,0,0,0,1,1,1]) # Afternoon
        self.prices_fx.extend([1,1,1,1,1,1,1,1])
        self.prices_fx.extend([0,0,0,0,0,-1,-1,-1])
        self.prices_fx.extend(self.prices_fx)
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'times': spaces.Box(low=0, high=48, shape=(3,), dtype=np.int32),
            'sizes': spaces.Box(low=0, high=48, shape=(3,), dtype=np.int32),
            'price_values': spaces.Box(low=-1, high=1, shape=(6,), dtype=np.int32)
        })
        
        # Define the action space
        self.action_space = spaces.MultiDiscrete([6, 3])
        
        # Initialize the obs
        self.obs = {
            'times': np.zeros((3,), dtype=np.int32),
            'sizes': np.zeros((3,), dtype=np.int32),
            'price_values': np.zeros((6,), dtype=np.int32)
        }
        
    def reset(self, seed=None):
        # Reset the obs to some initial values
        self.obs = {
            'times': np.zeros((3,), dtype=np.int32),
            'sizes': np.zeros((3,), dtype=np.int32),
            'price_values': np.array(self.prices_fx[:6], dtype=np.int32),
        }

        # Reset history
        self.history = []

        # Reset step count
        self.step_count = 0

        # info
        info = {
            "history" : [],
        }
        return (self.obs, info)

    def step(self, action):
        self.step_count += 1

        reward, size_reward, cost_penalization = 0, 0, 0

        job = int(action[0])
        machine = int(action[1])

        # Update history
        self.history.append(self.get_history(job, machine))

        # Apply the action to the obs (customize this logic as needed)
        if job > 0:
            self.obs['times'][machine] += job
        else:
            self.obs['times'][machine] += 1

        self.obs['sizes'][machine] += job

        # Update prices values
        current_step = min(self.obs['times'])
        self.obs['price_values'] = np.array(self.prices_fx[current_step: current_step + 6], dtype=np.int32)
        
        # Calculate the reward (customize this logic as needed)
        #reward = self.calculate_reward()
        
        # Check if the episode is done (customize this logic as needed)
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        if terminated or truncated:
            reward, size_reward, cost_penalization = self.calculate_final_reward()
        
        # Optionally, set additional info
        info = {
            "step count" : self.step_count,
            "rewards" : {"size reward": size_reward, "cost penalization": cost_penalization},
            "history" : self.history,
        }
        
        return self.obs, reward, terminated, truncated, info
    
    def get_history(self, job, machine):
        if job != 0:
            return [job, machine, job] # Duration, machine, job size
        else:
            return [1, machine, job]

    
    def calculate_final_reward(self):
        mef = 1 if self.machine_eff else 0 # machine efficiency factor
        epf = 1 if self.energy_prices else 0 # energy prices factor

        size_reward = np.sum(self.obs["sizes"])

        his = np.array(self.history)
        prices = self.prices_fx

        machines = [[] for _ in range(max(his[:,1])+1)]
        max_len = 48
        for item in his:
            duration, machine, size = item[0], item[1], item[2]
            for _ in range(duration):
                if len(machines[machine]) < max_len:
                    machines[machine].append(1) if size > 0 else machines[machine].append(0)

        for m in machines:
            if len(machines) < max_len:
                diff = max_len - len(m)
                m.extend(np.zeros(diff, dtype=int).tolist())

        cost_penalization = 0
        for i,m in enumerate(machines):
            for j, energy in enumerate(m):
                cost_penalization += energy * prices[j] * (mef*i + 1) * epf

        
        final_reward = size_reward - cost_penalization
        
        return final_reward, size_reward, cost_penalization
    
    def is_terminated(self):
        terminated = max(self.obs['times']) > 43
        return terminated.item()
    
    def is_truncated(self):
        truncated = self.step_count > 144
        return truncated
