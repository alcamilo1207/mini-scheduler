import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import plotly.express as px
# import pandas as pd
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, w1 = 1, w2 = 1, w3 = 1, machine_eff = False, energy_prices = False): 
        super(CustomEnv, self).__init__()

        # Optimizer parameters
        self.machine_eff = machine_eff # Considering machine efficiency?
        self.energy_prices = energy_prices # Considering energy prices?
        self.w1 = w1 # Machine efficiency reward: controls how much reward is given to the most efficient machine in comparizon with the least efficient
        self.w2 = w2 # Idling reward: control how much reward is given when idling at High prices. At Normal/Low prices the reward given for idling is zero
        self.w3 = w3 # Not yet used


        # Environment variables
        self.history = []
        self.prices_fx = np.ones(110)
        self.available_jobs = [] # Store the jobs that have not been scheduled. Changed in the reset and step function
        self.dataset_jobs = [] # Store the dataset of jobs and it does not change after initilizing the environment
        self.machine_times = [0 for _ in range(3)]
        self.step_count = 0

        # Dataset - Jobs
        self.dataset_jobs.extend([7, 4, 5, 6, 5])
        self.dataset_jobs.extend([5, 7, 6, 4, 5])
        for _ in range(2):
            self.dataset_jobs.extend(self.dataset_jobs)

        # Dataset - Energy price (binary e.g., 1: Normal/low, 2: high):
        self.prices_fx[20:28] = 2
        self.prices_fx[55:60] = 2
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'current_machine' : spaces.Discrete(3), 
            'remaining_time' : spaces.Box(low=-10, high=96, shape=(1,), dtype=np.int32),
            'next_jobs': spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32),
            'future_prices': spaces.Box(low=-15, high=15, shape=(3,), dtype=np.int32),
        })
        
        # Define the action space
        self.action_space = spaces.Discrete(4) # 0: idling, 1-3: The position of the selected job in the job queue
        
        # Initialize the obs
        self.obs = {
            'current_machine' : 0, 
            'remaining_time': np.zeros((1,), dtype=np.int32),
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
            'current_machine' : 0, 
            'remaining_time': np.array([96], dtype=np.int32),
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
        next_machine_time = self.machine_times[next_machine]
        job = int(action.item()) - 1

        size_of_job = 0
        if job > -1:
            size_of_job += self.obs["next_jobs"][job]

        # calculate cost
        reward += self.calculate_reward(current_machine, size_of_job,  duration=size_of_job)

        # Update history and machine times
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
        
        # update observations
        self.obs['current_machine'] = next_machine
        self.obs['remaining_time'] = np.array([96 - self.machine_times[next_machine]], dtype=np.int32)
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
        
    def calculate_cost(self, current_machine, current_machine_time, duration):
        cost = 0
        for i in range(duration):
            cost += self.prices_fx[i + current_machine_time] * (1 + current_machine)

        return cost
    
    def calculate_reward(self, current_machine, size, duration):
        current_machine_time = self.machine_times[current_machine]
        cost = self.calculate_cost(current_machine, current_machine_time, duration)
        if size != 0:
            return size.item() * (1 + self.w1*(1 - current_machine) + (1/cost))
        else:
            if self.obs["future_prices"][0] == 2:
                return 2 * self.w2
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