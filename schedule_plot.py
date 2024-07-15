from mini_env4 import CustomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_callback import SaveOnStepCallback
import os
import pandas as pd
import plotly.express as px
import numpy as np


# Schedule plot

def get_schedule_plot(prices_fx = None, color="energy"):
        # Create the save path
    log_dir = "test_4_ppo_checkpoints/rew2_300k"
    os.makedirs(log_dir, exist_ok=True)

    env_test = CustomEnv(w1=0.7, w2=4, w3=1, prices_fx= prices_fx, energy_prices=True, machine_eff=True, test=True)
    vec_env_test = DummyVecEnv([lambda: env_test])

    test_model = PPO.load(os.path.join(log_dir,"model_300k_steps"))
    # Reset the environment
    for env_to_test in [vec_env_test]:
        obs = env_to_test.reset()

        total_rew = 0
        while True:
            action, _ = test_model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env_to_test.step(action)
            total_rew += reward
            #print(f"\nAction: {action}")
            #print(f"Obs: {obs}")
            if terminated:
                print(f"Done: {terminated}")
                break

        env_to_test.close()
        info = info[0] # Due to wrapping
        print("total_rew",total_rew)

        history = np.array(info["history"])
        job_counter = 0
        for his in history:
            if his[2] > 0:
                job_counter += 1

    print(f"completed jobs: {job_counter}/40")

    # Plot the current times
    machine_efficiencies = [1.2, 1, 0.8]
    history = info["history"]
    prices = info["prices"]
        
    schedule = []
    for his in history:
        actual_duration = his[0]
        assigned_to = his[1]
        size = his[2]
        reward = his[3]
        energy = size*(1/machine_efficiencies[assigned_to])
        schedule.append([assigned_to, energy, size, actual_duration, reward])

    df = pd.DataFrame(schedule, columns=["assigned_to", "energy","size","actual_duration", "reward"])
    dates = pd.date_range("2024-01-01", periods=12, freq='2h')
    x_values = [i*8 for i in range(12)]
    fig1 = px.bar(df, x="actual_duration", y="assigned_to", color=color, orientation="h")#
    fig1.update_xaxes(tickvals=x_values, ticktext=[d.strftime('%H:00') for d in dates])
    fig1.update_traces(width=.9)
    fig1.update_coloraxes(colorbar={'orientation':'h', 'thickness':15, 'y': 1.1})

    fig2 = px.line(prices, line_shape="hv")
    fig2.update_xaxes(range=[0, 96])

    return fig1, fig2