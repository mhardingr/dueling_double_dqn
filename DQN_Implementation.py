#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:54:24 2018

@author: ibrahim
"""

import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, collections, random
import matplotlib.pyplot as plt
import math
from gym import spaces, logger

learning_rate_CP = 0.001
learning_rate_MC = 0.0001

steps_beyond_done = None

def next_state_func(state, action):
        # copied from gym's cartpole.py
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart)
        length = 0.5 # actually half the pole's length
        polemass_length = (masspole * length)
        force_mag = 10.0
        tau = 0.02  # seconds between state updates
        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        global steps_beyond_done
        
        x, x_dot, theta, theta_dot = state
        force = force_mag if action==1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        x  = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
        state = (x,x_dot,theta,theta_dot)
        done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif steps_beyond_done is None:
            # Pole just fell!
            steps_beyond_done = None
            reward = 1.0
        else:
            if steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            steps_beyond_done += 1
            reward = 0.0
        return np.array(state), reward, done, {}
        # copied from gym's cartpole.py


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.
    def __init__(self, environment_name):
        if environment_name == ('CartPole-v0'):
            state_dim = 4
            Q_dim = 2
            learning_rate = learning_rate_CP
        elif environment_name == 'MountainCar-v0':
            state_dim = 2
            Q_dim = 3
            learning_rate = learning_rate_MC
        else:
            assert "Environment not available"
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(8, input_dim=state_dim, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(Q_dim, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
            self.model.save_weights(suffix)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
    # Helper funciton to load model weights.
        self.model.load_weights(weight_file)

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        batch = random.sample(self.memory, batch_size)
        return batch


    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
        
class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #        (a) Epsilon Greedy Policy.
    #         (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env_name = environment_name
        self.env = gym.make(environment_name)
        self.model = QNetwork(environment_name)
        self.model_target = QNetwork(environment_name)
        self.memory = Replay_Memory()
        self.gamma = 1 if environment_name == 'MountainCar-v0' else 0.99
        self.initial_epsilon = 0.5
        self.epsilon = self.initial_epsilon
        self.final_epsilon = 0.05
        self.exploration_decay_steps = 10**5
        self.num_episodes = 1000
        self.batch_size = 32
        self.avg_training_episodes_return=[]
        self.avg_performance_episodes_return = []
        self.avg_performance_episodes_return_2SLA = []
        self.decay = 0.999

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        self.epsilon = max(self.final_epsilon, self.epsilon - self.initial_epsilon/self.exploration_decay_steps)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action
    
    def epsilon_greedy_policy_005(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if random.random() < 0.05:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action
    def two_step_lookahead(self,state):
        global steps_beyond_done
        sbd = steps_beyond_done
        next_state_0, reward_0, done_0, _ = next_state_func(state, 0)
        sbd1 = steps_beyond_done        
        
        if not done_0:
            next_state_0_0, reward_0_0, done_0_0, _ = next_state_func(next_state_0, 0)
            if not done_0_0:
                next_state_0_0 = np.expand_dims(next_state_0_0,0)
                return_action_0_0 = reward_0 + reward_0_0 + max(self.model.model.predict(next_state_0_0)[0])
            else:
                return_action_0_0 = reward_0 + reward_0_0
                
            steps_beyond_done = sbd1   
            next_state_0_1, reward_0_1, done_0_1,_ = next_state_func(next_state_0, 1)
            if not done_0_1:
                next_state_0_1 = np.expand_dims(next_state_0_1,0)
                return_action_0_1 = reward_0 + reward_0_1 + max(self.model.model.predict(next_state_0_1)[0])
            else:
                return_action_0_1 = reward_0 + reward_0_1    
        else:
            return_action_0_0 = reward_0
            return_action_0_1 = reward_0
        max_return_action_0 =max(return_action_0_0,return_action_0_1)
        
        steps_beyond_done = sbd
        next_state_1, reward_1, done_1, _ = next_state_func(state, 1)
        sbd2 = steps_beyond_done 
        if not done_1:
            next_state_1_0, reward_1_0, done_1_0, _ = next_state_func(next_state_1, 0)
            if not done_1_0:
                next_state_1_0 = np.expand_dims(next_state_1_0,0)
                return_action_1_0 = reward_1 + reward_1_0 + max(self.model.model.predict(next_state_1_0)[0])
            else:
                return_action_1_0 = reward_1 + reward_1_0
           
            steps_beyond_done = sbd2                   
            next_state_1_1, reward_1_1, done_1_1,_ = next_state_func(next_state_1, 1)
            if not done_1_1:
                next_state_1_1 = np.expand_dims(next_state_1_1,0)
                return_action_1_1 = reward_1 + reward_1_1 + max(self.model.model.predict(next_state_1_1)[0])
            else:
                return_action_1_1 = reward_1 + reward_1_1    
        else:
            return_action_1_0 = reward_1
            return_action_1_1 = reward_1
        max_return_action_1 =max(return_action_1_0,return_action_1_1)
        
#        next_state_1, reward_1, done_1, _ = next_state_func(state, 1)
#        next_state_1_0, reward_1_0, done_1_0, _ = next_state_func(next_state_1, 0)
#        next_state_1_1, reward_1_1, done_1_1, _ = next_state_func(next_state_1, 1)
#        next_state_1_0 = np.expand_dims(next_state_1_0,0)
#        next_state_1_1 = np.expand_dims(next_state_1_1,0)
#        return_action_1_0 = reward_1 + reward_1_0 + max(self.model.model.predict(next_state_1_0)[0])
#        return_action_1_1 = reward_1 + reward_1_1 + max(self.model.model.predict(next_state_1_1)[0])
#        max_return_action_1 =max(return_action_1_0,return_action_1_1)

        action = 0 if max_return_action_0 >= max_return_action_1 else 1
        
        return action


    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        action = np.argmax(q_values)
        return action
    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.
        counter = 1
        episodes_return = []
        num_of_episodes_to_update_train_and_perf_curve = 200
        num_episodes_for_performance_curve = 20
        self.avg_training_episodes_return = []
        self.avg_performance_episodes_return = []
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            returns = 0
            done =False
            while not done:
                q_values = self.model.model.predict(state)[0]
                action =  self.epsilon_greedy_policy(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                returns +=  reward 
            episodes_return.append(returns)
            ## get the points of the training curve
            if counter % num_of_episodes_to_update_train_and_perf_curve == 0:
                self.avg_training_episodes_return.append(sum(episodes_return)/num_of_episodes_to_update_train_and_perf_curve)
                episodes_return = []
            ## get the points for the performance curve
#                self.avg_performance_episodes_return.append(self.performance_plot_data(num_episodes_for_performance_curve))
            ## due a batch update
            batch = self.memory.sample_batch(self.batch_size)
            batch_states = []
            batch_q_values =[]
            for state, action, reward, next_state, done in batch:
                q_values = self.model.model.predict(state)[0]
                q_value = reward + self.gamma * np.amax(self.model_target.model.model.predict(next_state)[0])
                if done:
                    q_values[action] = reward
                else:
                    q_values[action] = q_value
                batch_states.append(state[0])
                batch_q_values.append(q_values)
            self.model.model.fit(np.array(batch_states), np.array(batch_q_values),batch_size= self.batch_size, epochs=1, verbose=0)
            if counter % 100 == 0:
                self.model_target.model.set_weights(self.model.model.get_weights())
                filename = 'weights_'+str(self.env_name)+'_'+str(counter)
                print(filename)
                self.model.save_model_weights(filename)
            print(counter)
            counter += 1
        plt.figure()
        plt.plot(self.avg_training_episodes_return,label='training_curve')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.legend(loc='best')
        plt.show()
#        plt.figure()
#        plt.plot(self.avg_performance_episodes_return,label='performance_curve')
#        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
#        plt.ylabel('Average Reward per Episode')
#        plt.legend(loc='best')
#        plt.show()
        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        for episode in range(100):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            returns = 0
            done =False
            while not done:
#                self.env.render()
                q_values = self.model.model.predict(state)[0]
                action =  self.greedy_policy(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                state = next_state
                returns += reward  
        print("total_returns=",returns)
        
    def performance_plot_data(self,num_episodes, model_file=None):
        episodes_return = []
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            returns = 0
            done =False
            while not done:
                q_values = self.model.model.predict(state)[0]
                action =  self.epsilon_greedy_policy_005(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                state = next_state
                # no discounting
                returns += reward 
            episodes_return.append(returns)
        return (sum(episodes_return)/num_episodes)
    def performance_plot_data_2_steps_LA(self,num_episodes, model_file=None):
        episodes_return = []
        for episode in range(num_episodes):
            state = self.env.reset()
            global steps_beyond_done
            steps_beyond_done = None
            returns = 0
            done =False
            while not done:
                action =  self.two_step_lookahead(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                # no discounting
                returns += reward 
            episodes_return.append(returns)
        return (sum(episodes_return)/num_episodes)
    
    def performance_curves_from_weight_files(self):
        self.avg_performance_episodes_return = []
        self.avg_performance_episodes_return_2SLA = []
        start = 200
        stop = self.num_episodes +1
        step = 200
        num_episodes_for_performance_curve = 20
        for indx in range(start, stop,step):
            filename = 'weights_'+str(self.env_name)+'_'+str(indx)
            self.model.load_model_weights(filename)
            self.avg_performance_episodes_return.append(self.performance_plot_data(num_episodes_for_performance_curve))
            self.avg_performance_episodes_return_2SLA.append(self.performance_plot_data_2_steps_LA(num_episodes_for_performance_curve))
            

    def plots(self):
        num_of_episodes_to_update_train_and_perf_curve = 200
        plt.figure()
        plt.plot(self.avg_performance_episodes_return,label='performance_curve')
        plt.plot(self.avg_performance_episodes_return_2SLA,label='performance_curve 2 steps look_ahead')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.legend(loc='best')
        plt.show()
    
    
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        state = self.env.reset()
        state = np.expand_dims(state,0)
        for i in range(32):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            #reward
            next_state = np.expand_dims(next_state,0)
            self.memory.append((state, action, reward, next_state, done))
            if not done:
                state = next_state
            else:
                state = self.env.reset()
                state = np.expand_dims(state,0)
                

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
#    main(sys.argv)
    agent = DQN_Agent('CartPole-v0')
    agent.burn_in_memory()
    agent.train()
    agent.test()
    agent.performance_curves_from_weight_files()
    agent.plots()

