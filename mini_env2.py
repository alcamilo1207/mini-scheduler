import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import plotly.express as px
# import pandas as pd
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, machine_eff = False, energy_prices = False):
        super(CustomEnv, self).__init__()

        # Optimizer parameters
        self.machine_eff = machine_eff
        self.energy_prices = energy_prices

        # Environment variables
        self.history = []
        self.prices_fx = []
        self.dataset_jobs = []
        self.available_jobs = []
        self.machine_times = [0 for _ in range(3)]
        self.step_count = 0

        # Jobs
        self.dataset_jobs.extend([7, 4, 5, 8, 3])
        self.dataset_jobs.extend([2, 7, 6, 2, 3])
        self.dataset_jobs.extend([5, 3, 7, 8, 4])
        for _ in range(2):
            self.dataset_jobs.extend(self.dataset_jobs)

        # Append prices:
        self.prices_fx.extend([1,1,1,1,1,1,1,1])
        self.prices_fx.extend([1,1,1,1,1,1,1,1])
        self.prices_fx.extend([1,1,1,2,2,2,2,2])
        self.prices_fx.extend([2,2,2,2,2,1,1,1])
        self.prices_fx.extend([1,1,1,1,1,1,1,1])
        self.prices_fx.extend([1,1,1,1,1,1,1,1]) # Noon
        for _ in range(2):
            self.prices_fx.extend(self.prices_fx)
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'next_jobs': spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32),
            'future_prices': spaces.Box(low=-15, high=15, shape=(3,), dtype=np.int32),
        })
        
        # Define the action space
        self.action_space = spaces.Discrete(4)
        
        # Initialize the obs
        self.obs = {
            'next_jobs': np.zeros((3,), dtype=np.int32),
            'future_prices': np.zeros((3,), dtype=np.int32)
        }
        
    def reset(self, seed = None):
        # Reset step count
        self.step_count = 0
        self.machine_times = [0 for _ in range(3)]
        self.available_jobs = self.dataset_jobs

        # Reset history
        self.history = []

        # Reset the obs to some initial values
        self.obs = {
            'next_jobs': np.array(self.available_jobs[:3], dtype=np.int32),
            'future_prices': np.array(self.prices_fx[:3], dtype=np.int32),
        }

        self.available_jobs = self.available_jobs[3:]

        # info
        info = {
            "history" : [],
        }
        return (self.obs, info)

    def step(self, action):
        reward = 0

        current_machine = self.step_count % 3
        next_machine = (self.step_count + 1) % 3
        current_machine_time = self.machine_times[current_machine]
        next_machine_time = self.machine_times[next_machine]
        job = int(action.item()) - 1

        size_of_job = 0
        if job > -1:
            size_of_job += self.obs["next_jobs"][job]

        # calculate cost
        cost = self.calculate_cost(current_machine, current_machine_time,size_of_job)
        reward += self.calculate_reward(size_of_job, cost)

        # Update history
        self.history.append(self.get_history(job, current_machine, size_of_job, reward))

        # Check if the episode is done (customize this logic as needed)
        terminated = self.is_terminated()
        truncated = self.is_truncated()

        if terminated or truncated:
            # Optionally, set additional info
            info = {
                "step count" : self.step_count,
                "rewards" : {"size of job": size_of_job, "reward": reward},
                "history" : self.history,
                "prices" : self.prices_fx
            }
        else:
            # Optionally, set additional info
            info = {
                "step count" : self.step_count,
                "rewards" : {"size of job": size_of_job, "reward": reward},
            }

        # Update environment
        self.update_next_jobs(job)
        
        # Apply the action to the obs (customize this logic as needed)
        self.obs['future_prices'] = np.array(self.prices_fx[next_machine_time: next_machine_time + 3], dtype=np.int32)

        self.step_count += 1

        return self.obs, reward, terminated, truncated, info
    
    def update_next_jobs(self, job): 
        if job > -1:
            if len(self.available_jobs) > 0:
                self.obs["next_jobs"][job] = self.available_jobs[0]
                self.available_jobs = self.available_jobs[1:]
            else:
                self.obs["next_jobs"][job] = np.int32(0)

    
    def get_history(self, job, machine, size_of_job, reward):
        if job > -1:
            self.machine_times[machine] += size_of_job.item()
            return [size_of_job.item(), machine, size_of_job.item(), reward] # Duration, machine, size_of_job, reward
        else:
            self.machine_times[machine] += 1
            return [1, machine, size_of_job, reward]
        
    def calculate_cost(self, current_machine,current_machine_time, duration):
        cost = 0
        for i in range(duration):
            cost += self.prices_fx[i + current_machine_time] # * (1 + current_machine)

        return cost
    
    def calculate_reward(self, size, cost):
        if size != 0:
            return size.item()/cost
        else:
            if self.obs["future_prices"][0] == 2:
                return 2
            else:
                return 0

    
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
        terminated = len(self.available_jobs) < 1 and max(self.obs["next_jobs"]) == 0
        terminated = terminated or max(self.machine_times) > 96
        return terminated
    
    def is_truncated(self):
        truncated = self.step_count > 288
        return truncated


# env = CustomEnv(energy_prices=True, machine_eff=True)

# # Reset the environment
# obs = env.reset()
# total_rew = 0
# steps = 0
# while True:
#     print(f"\nObs: {obs}")
#     action = env.action_space.sample()
#     obs, reward, terminated, _ , info = env.step(action)
#     total_rew += reward
#     steps += 1
#     print(f"Action: {action}")
#     print(f"Rewars: {reward}")
#     if terminated:
#         print(info["rewards"])
#         print("step count:",info["step count"])
#         break

# env.close()
# print("total_rew",total_rew, "steps", steps)