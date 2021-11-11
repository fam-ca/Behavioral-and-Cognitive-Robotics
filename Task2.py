import numpy as np
import gym

class Network:
    def __init__(self, env, nhiddens):
        ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            noutputs = env.action_space.shape[0]
        else:
            noutputs = env.action_space.n

        self.ninputs = ninputs
        self.nhiddens = nhiddens
        self.noutputs = noutputs
        
    def update(self, ninputs, observation, W1, b1, W2, b2):
        # change the shape of observation vector
        observation.resize(ninputs,1) # matrix [ninputs x 1]

        Z1 = np.dot(W1, observation) + b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        return action

    def evaluate(self, env, nepisodes, W1, b1, W2, b2):
        done = False
        fitness = 0
        fitnesses = []
        ninputs = self.ninputs
        for _ in range(nepisodes):
            observation = env.reset()
            while not done:
                action = self.update(ninputs, observation, W1, b1, W2, b2)
                env.render()
                observation, reward, done, info = env.step(action)
                fitness += reward
            fitnesses.append(fitness)
        return np.mean(fitnesses)

    def setparameters(self):
        W1 = np.random.randn(nhiddens,ninputs) * pvariance      
        W2 = np.random.randn(noutputs, nhiddens) * pvariance   
        b1 = np.zeros(shape=(nhiddens, 1))                     
        b2 = np.zeros(shape=(noutputs, 1))  
        return W1, b1, W2, b2

env = gym.make("CartPole-v0")
network = Network(env, 5)

ninputs, nhiddens, noutputs = network.ninputs, network.nhiddens, network.noutputs

pvariance = 0.1     
ppvariance = 0.02   
nepisodes = 10     
W1, b1, W2, b2 = network.setparameters()                   

observation = env.reset()
avr_fit = network.evaluate(env, nepisodes, W1, b1, W2, b2)
print("Average fitness", avr_fit) 