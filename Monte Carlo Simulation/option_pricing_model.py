#  Importing Libraries

import numpy as np
from scipy.stats import norm
import math
import time

#  Black Scholes formula for option pricing

class BlackScholes:
    def __init__(self, S0, E, T, rf, sigma):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        
    def call_option_simulation(self):
        d1 = (np.log(self.S0/self.E) + (self.rf + 0.5*self.sigma**2)*self.T)/ self.sigma*np.sqrt(self.T)
        d2 = d1 - self.sigma*np.sqrt(self.T)
        stock_price = self.S0*norm.cdf(d1, 0, 1) - self.E*np.exp(-1.0*self.rf*self.T)*norm.cdf(d2, 0, 1)
        return stock_price
    
    def put_option_simulation(self):
        d1 = (np.log(self.S0/self.E) + (self.rf + 0.5*self.sigma**2)*self.T)/ self.sigma*np.sqrt(self.T)
        d2 = d1 - self.sigma*np.sqrt(self.T)
        stock_price = self.E*np.exp(-1.0*self.rf*self.T)*norm.cdf(-d2, 0, 1) - self.S0*norm.cdf(-d1, 0, 1)
        return stock_price
    
#  Monte Carlo Simulation for option pricing

class MonteCarlo:
    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations
        
    def call_option_simulation(self):
#         we have two columns: the first with 0s and the second column will store the payoff
#         we need the first column of 0s: payoff function is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])
        
#         dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
    
#         equation for the S(t) stock price
        stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2) + self.sigma*np.sqrt(self.T)*rand)
        
#         we need S-E because we need to calculate max(S-E,0)
        option_data[:, 1] = stock_price - self.E
    
#         average for the Monte Carlo method
        average = np.sum(np.amax(option_data, axis = 1))/float(self.iterations)
    
#         we have to use the discount factor as exp(-rT) for continuous discounting
        return np.exp(-1.0*self.rf*self.T)*average

    def put_option_simulation(self):
#         we have two columns: the first with 0s and the second column will store the payoff
#         we need the first column of 0s: payoff function is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])
    
#         dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
    
#         equation for the S(t) stock price
        stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*self.sigma**2) + self.sigma*np.sqrt(self.T)*rand)
    
#         we need S-E because we need to calculate max(S-E,0)
        option_data[:, 1] = self.E - stock_price
    
#         average for the Monte Carlo method
        average = np.sum(np.amax(option_data, axis = 1))/float(self.iterations)
    
#         we have to use the discount factor as exp(-rT) for continuous discounting
        return np.exp(-1.0*self.rf*self.T)*average   
    
# Parameters for the stock

if __name__ == "__main__":
    S0 = 100                       # underlying stock price at t = 0
    E = 100                        # strike price
    T = 1                          # expiry
    rf = 0.3                      # risk-free rate
    sigma = 0.2                   # volatility of the underlying stock
    iterations = 1000000         # number of iterations in the Monte Carlo Simulation
    
    model1 = MonteCarlo(S0, E, T, rf, sigma, iterations)
    model2 = BlackScholes(S0, E, T, rf, sigma)
    print("Call option price with Monte Carlo approach: ", model1.call_option_simulation())
    print("Put option price with Monte Carlo approach: ", model1.put_option_simulation())
    print("Call option price with Black Scholes approach: ", model2.call_option_simulation())
    print("Put option price with Black Scholes approach: ", model2.put_option_simulation())