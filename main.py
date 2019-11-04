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

import strategy as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random




# Retrieve important parameters from stock market data
stock = pd.read_csv("DJI.csv")
opens = np.array(stock['Open'])
closes = np.array(stock['Close'])
returns = closes / opens
gains = closes - opens

# plt.plot(returns)

max_gain = 0





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


# Starting capital
capital = 1000

# For each being created, mutate its brain
for n in range(population_size):
    strategy = st.Strategy()
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


