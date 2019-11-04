"""
Author: Max Martinez Ruts
Date: September 2019
Idea:

Try to predict the future value of stocks by using neuroevolution.
The idea is to create a generation of different strategies and testing the strategies on different domains of the
stcok market value for each generation. For every single generation, the strategies that lead to the highest capitals
after being trained on the same sample of the market are the ones that are going to be carried to the next generation
with subtle modifications.

The NN in charge of determining the policy of the strategy take as inputs the nth derivatives of the last 10 values of
the sample taken and outputs the percentage the current capital to invest in the stock market.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd
import random

# Get taylor coefficients to later determine the nth derivatives of a sample of sequential data
def get_coeffs(n):
    taylor_table = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            taylor_table[j, n - i - 1] = (-(i)) ** j
    taylor_derivatives = np.zeros((n, 1))
    taylor_derivatives[n - 2] = 1
    coeffs = np.dot(np.linalg.inv(taylor_table), taylor_derivatives)
    return coeffs


# Retrieve important parameters from stock market data
coeffs = get_coeffs(3)
stock = pd.read_csv("DJI.csv")
opens = np.array(stock['Open'])
closes = np.array(stock['Close'])
returns = closes / opens
gains = closes - opens

# plt.plot(returns)

max_gain = 0


# Brain is a class that instantiates the NN in charge of making the investment decisions
class Brain:
    def __init__(self, model):
        if model == 0:
            self.model = tf.keras.models.clone_model(model_base)
            input = np.asarray([list(np.zeros(i_n))])
            self.model.predict(input, 1)[0]

        else:
            self.model = tf.keras.models.clone_model(model, input_tensors=None)

    # Mix the genome of two parents
    def crossover(self, genes_1, genes_2):

        weights_hidden = (genes_1[0] + genes_2[0]) / 2
        biases_hidden = (genes_1[1] + genes_2[1]) / 2
        weights_outputs = (genes_1[2] + genes_2[2]) / 2
        biases_outputs = (genes_1[3] + genes_2[3]) / 2
        self.weights = [weights_hidden, biases_hidden, weights_outputs, biases_outputs]
        self.model.set_weights(self.weights)

    # Modify the genome randomly
    def mutate(self):
        self.weights = self.model.get_weights()
        w1 = np.random.randn(i_n, 30)
        r = np.random.rand(i_n, 30)
        w1 = np.where(r > 0.7, w1, 0)

        b1 = np.random.randn(30)
        r = np.random.rand(30)
        b1 = np.where(r > 0.7, b1, 0)

        w2 = np.random.randn(30, 1)
        r = np.random.rand(30, 1)
        w2 = np.where(r > 0.7, w2, 0)

        b2 = np.random.randn(1) / 2
        r = np.random.rand(1)
        b2 = np.where(r > 0.7, b2, 0)
        self.weights[0] += w1
        self.weights[1] += b1
        self.weights[2] += w2
        self.weights[3] += b2
        self.model.set_weights(self.weights)

    # Set weights to the given weights
    def create(self):
        self.model.set_weights(self.weights)


# Brain is a class that instantiates each strategy being tested. A strategy will contain a Brain and several other parameters to test the effectiveness of it
class Strategy:
    def __init__(self, model=0):
        self.scoretrack = []
        self.capitaltrack = []
        self.score = 0
        self.fitness = 0
        self.brain = Brain(model)
        self.capital = 1000

    #
    def cash_in(self, last_10):
        n_derivatives = []
        for i in range(2, len(last_10)):
            n_derivatives.append(np.dot(last_10[-i - 1:], get_coeffs(i + 1))[0])
        input = np.asarray([n_derivatives + [last_10[-1], last_10[-2]]])
        output = self.brain.model.predict(input, 1)[0]

        # Invest as much as the output indicates of your capital to the stock market
        self.percentage_invested = output[0]

        # Money to invest
        money_in = min(self.capital * self.percentage_invested, last_10[-1])
        self.capital -= money_in

        # Actions that you bought
        self.actions = money_in / (last_10[-1])

        # Substract from score the quantity invested
        self.score -= last_10[-1] * self.percentage_invested

    def cash_out(self, fst, last):
        # Add to score the quantity earned
        self.score += last * self.percentage_invested

        # Get the capital from the actions in the stock market
        self.capital += self.actions * last

        # Keep track of scores and capital
        self.scoretrack.append(self.score + fst - last)
        self.capitaltrack.append(self.capital)

    # Reset strategy parameters (except for brain). This is so that it can be tested with the same brain in another sample of the market stock value
    def reset(self):
        self.scoretrack = []
        self.capitaltrack = []
        self.score = 0
        self.capital = 1000
        self.fitness = 0


population_size = 10  # N. strategies in one generation
strategies = []  # List containing all strategies
last_strategies = []  # List containing strategies from last generation

# Parameters for debugging the effectiveness of each generation
histograms = []
medians = []
means = []
mins = []
maxs = []
i_n = 5

# Build a vanilla NN to subsequentally create copies of it
model_base = tf.keras.models.Sequential()
input_layer = tf.keras.layers.Flatten()
hidden_layer = tf.keras.layers.Dense(units=30, input_shape=[i_n], activation='sigmoid')
output_layer = tf.keras.layers.Dense(units=1, input_shape=[30], activation='sigmoid')
model_base.add(input_layer)
model_base.add(hidden_layer)
model_base.add(output_layer)
input = np.asarray([list(np.zeros(i_n))])
start = time.time()
model_base.predict(input, verbose=0)

# Starting capital
capital = 1000

# For each being created, mutate its brain
for n in range(population_size):
    strategy = Strategy()
    strategies.append(strategy)
    strategy.brain.mutate()

# For each generation
for generation in range(1, 100):
    # Set a random starting point (one such that there is no overlap with the last 1000 values), and then define the sample of that stock taken
    # The idea is to test each generation on a different domain of the stock market so that overfitting is avoided (data amplification)

    """ TRAINING """
    st = random.randint(0, len(gains) - 2001)
    stock = list(gains[st:st + 1000])

    # Test each strategy on the sample stock taken
    for i in range(10, len(stock) - 1):
        for strategy in strategies:
            strategy.cash_in(stock[i - i_n:i])
        for strategy in strategies:
            strategy.cash_out(stock[10], stock[i])

    collective_score = 0  # Added score of all strategies
    scores = []  # List containing all scores

    plt.show()

    for strategy in strategies:
        strategy.score = strategy.score + stock[0] - stock[
            len(stock) - 1]  # Score relative to the difference of the opening and closing of the sample taken
        collective_score += strategy.score
        scores.append(strategy.score)

    median_score = np.median(scores)

    for strategy in strategies:
        strategy.fitness = strategy.score / collective_score  # Fitness of strategy is the percentage of score relative to the collective score

    p1s = []  # List of parents N.1
    p2s = []  # List of parents N.2

    print('-----Training Results-------')
    print('Collective data:', collective_score, 'Median:', np.median(scores), 'Mean:', np.mean(scores))

    # Choose parents by selecting strategies with more fitness more often
    for strategy in strategies:
        index = 0
        r = np.random.uniform(0, 1)
        while r > 0:
            r -= strategies[index].fitness
            index += 1
        index -= 1
        p1s.append(strategies[index].brain.model.get_weights())

        index = 0
        r = np.random.uniform(0, 1)
        while r > 0:
            r -= strategies[index].fitness
            index += 1
        index -= 1

        p2s.append(strategies[index].brain.model.get_weights())

    # Reset scores
    for v in range(len(strategies)):
        strategies[v].reset()

    """ TESTING """
    # Always get the same range of the stock so that it can be tested and compared each generation
    test_stock = list(gains[4000:5000])

    # Test each strategy on the test stock
    for i in range(10, len(test_stock) - 1):
        for strategy in strategies:
            strategy.cash_in(test_stock[i - i_n:i])
        for strategy in strategies:
            strategy.cash_out(test_stock[10], test_stock[i])

    plt.plot(test_stock)

    # Plot each strategie capital track
    for strategy in strategies:
        plt.plot(strategy.scoretrack)
    plt.show()

    collective_score = 0
    scores = []
    for strategy in strategies:
        collective_score += strategy.score + test_stock[0] - test_stock[len(test_stock) - 1]
        scores.append(strategy.score)

    # Plot and save relevant data
    histograms.append(scores)
    medians.append(np.median(scores))
    means.append(np.mean(scores))
    maxs.append(np.max(scores))
    mins.append(np.min(scores))
    fig = plt.figure()
    plt.plot(np.arange(0, generation), means, label='Mean')
    plt.plot(np.arange(0, generation), medians, label='Median')
    plt.plot(np.arange(0, generation), mins, label='Min')
    plt.plot(np.arange(0, generation), maxs, label='Max')
    plt.legend(loc='upper left')
    plt.xlabel('Generation [-]')
    plt.ylabel('Score [-]')
    fig.savefig('progression/___progress_' + str(generation) + '.png')
    fig = plt.figure()
    plt.hist(histograms[-1], bins=np.linspace(0, 100, 40))
    plt.xlim(0, 1000)
    plt.ylim(0, population_size)
    plt.xlabel('Score [-]')
    plt.ylabel('Frequence [-]')
    plt.title('Generation ' + str(generation))
    fig.savefig('histograms/___histogram_' + str(generation) + '.png')

    print('-----Test results------')
    print('Collective data:', collective_score, 'Median:', np.median(scores), 'Mean:', np.mean(scores))

    # Now create a new generation
    for v in range(len(strategies)):
        child = strategies[v]  # Create a child strategy for each strategy of the last generation
        child.reset()  # Reset scores
        child.brain.crossover(p1s[v], p2s[v])  # Build brain by mixing genotype of two parents
        child.brain.mutate()  # Mutate brain
        child.brain.create()
        strategies[v] = child


