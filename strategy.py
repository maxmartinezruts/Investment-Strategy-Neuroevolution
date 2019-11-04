import brain as br
import numpy as np
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


# Brain is a class that instantiates each strategy being tested. A strategy will contain a Brain and several other parameters to test the effectiveness of it
class Strategy:
    def __init__(self, model=0):
        self.scoretrack = []
        self.capitaltrack = []
        self.score = 0
        self.fitness = 0
        self.brain = br.Brain(model)
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