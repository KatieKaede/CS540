import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            old_obs = obs

            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                qs = [Q_table[(obs, a)] for a in range(env.action_space.n)]
                action = int(np.argmax(qs))

            next_obs,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated

            if not done:
                next_qs = [Q_table[(next_obs, a)] for a in range(env.action_space.n)]
                target = reward + DISCOUNT_FACTOR * max(next_qs)
            else:
                target = reward

            current = Q_table[(old_obs, action)]
            Q_table[(old_obs, action)] = (1 - LEARNING_RATE)*current + LEARNING_RATE*target

            obs = next_obs
            episode_reward += reward
        
        EPSILON *= EPSILON_DECAY

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################