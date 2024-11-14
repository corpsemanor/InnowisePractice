import math
import time

class RandomGenerator:
    def __init__(self, seed=None):
        """
        Initializes a random number generator.

        :param seed: Initial value for the random number generator. If not provided, the current time is used.
        """
        self.seed = seed or int(time.time())
        self.counter = self.seed
        self.M = 2**48
        self.A = 25214903917
        self.C = 11 

    def _lcg(self):
        """
        Linear congruential generator for random numbers.

        :return: The next random number in the interval [0, 1).
        """
        self.counter = (self.A * self.counter + self.C) % self.M
        return self.counter / self.M

    def generate(self, n=1, N=100):
        """
        Generates a list of random numbers in the interval [0, N].

        :param n: Number of random values to generate.
        :param N: Upper bound of the interval for random numbers.
        :return: List of random numbers.
        """
        return [int(self._lcg() * N) for _ in range(n)]  

class UniformGenerator(RandomGenerator):
    def __init__(self, seed=None):
        """
        Initializes a uniform random number generator.

        :param seed: Initial value for the random number generator. If not provided, the current time is used.
        """
        super().__init__(seed)

    def generate(self, n=1, a=0, b=1, N=None):
        """
        Generates uniformly distributed numbers in the range [a, b] or [0, N].

        :param n: Number of random values to generate.
        :param a: Lower bound of the interval.
        :param b: Upper bound of the interval.
        :param N: Optional upper bound for the interval [0, N].
        :return: List of random numbers.
        """
        if N is not None:
            return [a + (b - a) * (self._lcg() * N) / N for _ in range(n)]
        else:
            return [a + (b - a) * self._lcg() for _ in range(n)]

class NormalGenerator(RandomGenerator):
    def __init__(self, mu=0, sigma=1, seed=None, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0):
        """
        Initializes a normal random number generator for 1D and 2D normal distributions.

        :param mu: Mean of the normal distribution (for 1D).
        :param sigma: Standard deviation of the normal distribution (for 1D).
        :param seed: Initial value for the random number generator. If not provided, the current time is used.
        :param mu_x: Mean of the normal distribution in the x-dimension (for 2D).
        :param mu_y: Mean of the normal distribution in the y-dimension (for 2D).
        :param sigma_x: Standard deviation in the x-dimension (for 2D).
        :param sigma_y: Standard deviation in the y-dimension (for 2D).
        :param rho: Correlation coefficient between x and y (for 2D).
        """
        super().__init__(seed)
        self.mu = mu
        self.sigma = sigma
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho

    def _box_muller(self):
        """
        Box-Muller method for generating normal random numbers.

        :return: A random normal value.
        """
        u1 = self._lcg()
        u2 = self._lcg()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return z0

    def generate(self, n=1):
        """
        Generates a list of normal random numbers with mean (mu) and standard deviation (sigma).

        :param n: Number of random values to generate.
        :return: List of normal random numbers.
        """
        return [self.mu + self.sigma * self._box_muller() for _ in range(n)]
    
    def pdf(self, x):
        """
        Probability density function (PDF) of the normal distribution at point x.

        :param x: Point where the PDF is to be evaluated.
        :return: Value of the probability density at point x.
        """
        coefficient = 1 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        """
        Cumulative distribution function (CDF) of the normal distribution at point x.

        :param x: Point where the CDF is to be evaluated.
        :return: Value of the cumulative distribution at point x.
        """
        z = (x - self.mu) / self.sigma
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def pdf_2d(self, x, y):
        """
        Probability density function (PDF) for a 2D normal distribution at point (x, y).
        
        :param x: x-coordinate where the PDF is to be evaluated.
        :param y: y-coordinate where the PDF is to be evaluated.
        :return: Value of the probability density at point (x, y).
        """
        coefficient = 1 / (2 * math.pi * self.sigma_x * self.sigma_y * math.sqrt(1 - self.rho ** 2))
        exp_numerator = ((x - self.mu_x) ** 2) / (self.sigma_x ** 2) + ((y - self.mu_y) ** 2) / (self.sigma_y ** 2) \
                        - 2 * self.rho * ((x - self.mu_x) * (y - self.mu_y)) / (self.sigma_x * self.sigma_y)
        exponent = -exp_numerator / (2 * (1 - self.rho ** 2))
        return coefficient * math.exp(exponent)