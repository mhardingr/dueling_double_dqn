#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:54:24 2018

@author: ibrahim
"""

import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, collections, random
import matplotlib.pyplot as plt

learning_rate_CP = 0.001
learning_rate_MC = 0.0001



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


    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        self.epsilon = max(self.final_epsilon, self.epsilon - self.initial_epsilon/self.exploration_decay_steps)
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
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
        episodes_20_return = []
        self.avg_training_episodes_return = []
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            discount=1
            returns = 0
            done =False
            while not done:
                q_values = self.model.model.predict(state)[0]
                action =  self.epsilon_greedy_policy(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                returns += discount * reward 
                discount=discount*self.gamma
            episodes_20_return.append(returns)
            if counter % 20 == 0:
                self.avg_training_episodes_return.append(sum(episodes_20_return)/20)
                episodes_20_return = []
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
        plt.xlabel('Training Epochs (1 epoch corresponds to 20 episodes/weight updates)')
        plt.ylabel('Average Reward per Episode')
        plt.legend(loc='best')
        plt.show()
        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        for episode in range(100):
            state = self.env.reset()
            state = np.expand_dims(state,0)#np.reshape(state, [1,self.env.observation_space.shape[0]])
            returns = 0
            done =False
            while not done:
                self.env.render()
                q_values = self.model.model.predict(state)[0]
                action =  self.greedy_policy(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)#np.reshape(next_state, [1,self.env.observation_space.shape[0]])
                state = next_state
                returns += reward  
        print("total_returns=",returns)
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        state = self.env.reset()
        state = np.expand_dims(state,0)#np.reshape(state, [1,self.env.observation_space.shape[0]])
#        state = np.expand_dims(state,0)
        for i in range(32):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            #reward
            next_state = np.expand_dims(next_state,0)#np.reshape(next_state, [1,self.env.observation_space.shape[0]])
            self.memory.append((state, action, reward, next_state, done))
            if not done:
                state = next_state
            else:
                state = self.env.reset()
                state = np.expand_dims(state,0)#np.reshape(state, [1,self.env.observation_space.shape[0]])
                

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

