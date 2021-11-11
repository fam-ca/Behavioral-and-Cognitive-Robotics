import gym
import numpy as np
import time

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
        
    def update(self, observation):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        ninputs = self.ninputs
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

    def evaluate(self, env, nepisodes, render=False):
        fitnesses = []
        for _ in range(nepisodes):
            done = False
            fitness = 0
            observation = env.reset()
            while not done:
                if render:
                    env.render()
                    time.sleep(0.1)
                action = self.update(observation)
                observation, reward, done, info = env.step(action)
                fitness += reward
            fitnesses.append(fitness)

        return round(np.mean(fitnesses), 1)

    def getnparameters(self):
        # calculate number of paramters and return this number
        # n of sensory neurons * n of internal neurons + n of internal neurons* n moto neurons
        # + biases 
        ninputs, nhiddens, noutputs = self.ninputs, self.nhiddens, self.noutputs
        nparameters = nhiddens*ninputs + noutputs*nhiddens + nhiddens + noutputs
               
        return nparameters

    def setparameters(self, genotype):
        # genotype is a vector with 37 params
        ninputs, nhiddens, noutputs = self.ninputs, self.nhiddens, self.noutputs
 
        W1 = genotype[0:nhiddens*ninputs]
        W1.resize(nhiddens,ninputs)
        W2 = genotype[nhiddens*ninputs:nhiddens*ninputs+noutputs*nhiddens]
        W2.resize(noutputs, nhiddens)
        b1 = np.zeros(shape=(nhiddens, 1))        # bias internal neurons
        b2 = np.zeros(shape=(noutputs, 1))        # bias motor neurons
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

np.random.seed(42)
# env = gym.make("Pendulum-v1")
env = gym.make("CartPole-v0")
network = Network(env, 5)

popsize = 10
generange = 0.2                                   # range of initial weights genotype
mutrange = 0.05                                   # gives variation when creating new individuals
nepisodes = 3
ngenerations = 100

nparameters = network.getnparameters()
population = np.random.randn(popsize, nparameters) * generange # matrix 10 x 37
for g in range(ngenerations):
    fitness_tuple = []
    fitness = []
    for i in range(popsize):

        network.setparameters(population[i])

        fit = network.evaluate(env, nepisodes)
        fitness_tuple.append((i,fit)) 
        fitness.append(fit)
    fitness_tuple.sort(key=lambda x:x[1]) 

    for i in range(popsize//2):
        population[fitness_tuple[i][0]] = population[fitness_tuple[popsize//2+i][0]] + np.random.randn(1,nparameters) * mutrange
    print (f"generation {g+1} fitness best {np.max(fitness)} fitness average {np.mean(fitness):.1f}")
    # if np.max(fitness[:][1]) >= 200:
    #     break

network.evaluate(env, nepisodes, True)
env.close()