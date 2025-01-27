from generators import UniformGenerator
from numerical_methods import find_maximum, binary_search, compute_support_from_cdf, compute_support_from_pdf

class DistributionSampler:
    def __init__(self, pdf=None, cdf=None, support=None, seed=None):
        """
        Initializes a distribution sampler.

        :param pdf: Probability density function (PDF) for the distribution.
        :param cdf: Cumulative distribution function (CDF) for the distribution.
        :param support: Range of values for the random variable (if not provided, it will be computed automatically).
        :param seed: Initial value for the random number generator.
        """
        self.pdf = pdf
        self.cdf = cdf
        self.support = support
        self.seed = seed
        self.random_gen = UniformGenerator(seed=seed)
    
    def inverse_transform_sampling(self, n=1):
        """
        Inverse transform sampling method.

        :param n: Number of random samples to generate.
        :return: List of samples generated using inverse transform sampling.
        :raises ValueError: If CDF is not provided.
        """
        if not self.cdf:
            raise ValueError("CDF function not specified for inverse transform method")
        
        support = compute_support_from_cdf(self.cdf)
        uniform_samples = self.random_gen.generate(n)
        return [binary_search(self.cdf, y, *support) for y in uniform_samples]

    def rejection_sampling(self, n=1):
        """
        Rejection sampling method.

        :param n: Number of random samples to generate.
        :return: List of samples generated using rejection sampling.
        :raises ValueError: If PDF is not provided.
        """
        if not self.pdf:
            raise ValueError("PDF function not specified for rejection sampling method")
        
        support = compute_support_from_pdf(self.pdf)

        samples = []
        max_pdf = find_maximum(self.pdf, *support)
        
        attempts = 0
        max_attempts = n * 100
        while len(samples) < n and attempts < max_attempts:
            x = self.random_gen.generate()[0] * (support[1] - support[0]) + support[0]
            y = self.random_gen.generate()[0] * max_pdf
            if y <= self.pdf(x):
                samples.append(x)
            attempts += 1
        
        return samples
    

class BivariateSampler:
    """
    A class for generating samples from a 2D discrete or continuous random variable, 
    given a PDF for continuous variables or probability table for discrete variables.
    """

    def __init__(self, pdf=None, support=None, probability_table=None, seed=None):
        """
        Initializes the sampler with either a continuous PDF and support or a discrete probability table.
        
        Parameters:
            pdf (callable): Probability density function for continuous variables.
            support (tuple): Tuple (x_min, x_max, y_min, y_max) defining the bounds of the 2D support.
            probability_table (dict): Dictionary with (x, y) pairs as keys and probabilities as values for discrete variables.
            seed (int): Seed for random number generator to ensure reproducibility.
        """
        self.pdf = pdf
        self.support = support
        self.probability_table = probability_table
        self.random_gen = UniformGenerator(seed=seed)

    def generate_samples_continuous(self, n=1):
        """
        Generates samples using rejection sampling from a continuous 2D PDF.
        
        Parameters:
            n (int): Number of samples to generate.
        
        Returns:
            list of tuples: Generated samples as (x, y) pairs.
        """
        x_min, x_max, y_min, y_max = self.support
        max_pdf_value = max(self.pdf(x, y) for x in range(int(x_min), int(x_max)) for y in range(int(y_min), int(y_max)))

        samples = []
        while len(samples) < n:
            x = self.random_gen.generate(1, x_min, x_max)[0]
            y = self.random_gen.generate(1, y_min, y_max)[0]
            threshold = self.random_gen.generate(1, 0, max_pdf_value)[0]
            if threshold < self.pdf(x, y):
                samples.append((x, y))
        
        return samples

    def generate_samples_discrete(self, n=1):
        """
        Generates samples from a discrete 2D distribution based on a probability table.
        
        Parameters:
            n (int): Number of samples to generate.
        
        Returns:
            list of tuples: Generated samples as (x, y) pairs.
        """
        if not self.probability_table:
            raise ValueError("Probability table must be provided for discrete sampling")
        
        pairs, probs = zip(*self.probability_table.items())
        cumulative_probs = [sum(probs[:i+1]) for i in range(len(probs))]
        
        samples = []
        for _ in range(n):
            rand_value = self.random_gen.generate(1, 0, 1)[0]
            for i, threshold in enumerate(cumulative_probs):
                if rand_value <= threshold:
                    samples.append(pairs[i])
                    break
        
        return samples
