import math
import time

class Lib:

    _counter = 0

    def get_seed(self):
        seed = int(time.time() * 1000) + self._counter
        self._counter += int(seed//17)
        if seed > 1000000000000:
            seed = seed % 1000050000
        return seed

    def float_range(self, start, stop, step):
        while start < stop:
            yield float(start)
            start += step

    def normal_distribution(self, x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def LCG(self, N, length=1, seed = None, a=25214903917, c = 11, m = 2**48):
        sequence = []
        if seed is None:
            seed = self.get_seed()

        for _ in range(length):
            seed = (a * seed + c) % m
            number = seed % (N + 1)
            sequence.append(number)

        return sequence if length != 1 else sequence[0]
        
    def LCG_float(self, a, b, length=1, seed = None, _a=25214903917, c = 11, m = 2**48):

        sequence = []
        if seed is None:
            seed = self.get_seed()
        for _ in range(length):
            seed = (_a * seed + c) % m
            random_value = seed / m
            scaled_value = a + (b - a) * random_value
            sequence.append(scaled_value)
        
        return sequence if length != 1 else sequence[0]

    def approximate_inverse_cdf(self, cdf, target_prob, x_min=-10, x_max=10, tolerance=1e-5):
        while (x_max - x_min) > tolerance:
            x_mid = (x_min + x_max) / 2
            if cdf(x_mid) < target_prob:
                x_min = x_mid
            else:
                x_max = x_mid
        return (x_min + x_max) / 2

    def generate_samples(self, n, cdf=None, pdf=None):
        if cdf is not None:

            inverse_cdf = lambda p: self.approximate_inverse_cdf(cdf, p)
            uniform_randoms = self.LCG_float(0,1,n)
            samples = [inverse_cdf(p) for p in uniform_randoms]

        elif pdf is not None:

            samples = []
            x_values = [i * 0.01 for i in range(-500, 500)]
            max_pdf_value = max(pdf(x) for x in x_values)

            while len(samples) < n:
                x = self.LCG_float(-5,5)
                y = self.LCG_float(0, max_pdf_value)
                if y < pdf(x):
                    samples.append(x)
        
        return samples

    def unnormed_pdf(self, x, y):
        return math.exp(x**2-y**2)


    def binary_normal(self, x, y):
        return (1 / (2 * math.pi)) * math.exp(-0.5 * (x**2 + y**2))

    def rectangle_uniform(self, x, y, a=0, b=1, c=0, d=1):
        if a <= x <= b and c <= y <= d:
            return 1 / ((b - a) * (d - c))
        else:
            return 0
        
    def arbitrary_binary_normal(self, x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0):
        coeff = 1 / (2 * math.pi * sigma_x * sigma_y * math.sqrt(1 - rho**2))
        exponent = -0.5 * ( (x - mu_x)**2 / sigma_x**2 + (y - mu_y)**2 / sigma_y**2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) )
        return coeff * math.exp(exponent)

    def rectangle_integral(self, func, x_min, x_max, y_min, y_max, dx=0.1, dy=0.1):
        integral = 0.0
        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                integral += func(x, y) * dx * dy
                y += dy
            x += dx
        return integral

    def calculate_normalization_factor(self, pdf, x_min, x_max, y_min, y_max, dx=0.1, dy=0.1):
        return self.rectangle_integral(pdf, x_min, x_max, y_min, y_max, dx, dy)

    def calculate_variance(self, pdf, mean_x, mean_y, dx=0.01, dy=0.01):
        total_pdf = self.rectangle_integral(pdf, mean_x - 5, mean_x + 5, mean_y - 5, mean_y + 5, dx, dy)
        variance_x = self.rectangle_integral(lambda x, y: pdf(x, y) * ((x - mean_x) ** 2) / total_pdf, 
                                        mean_x - 5, mean_x + 5, mean_y - 5, mean_y + 5, dx, dy)
        variance_y = self.rectangle_integral(lambda x, y: pdf(x, y) * ((y - mean_y) ** 2) / total_pdf, 
                                        mean_x - 5, mean_x + 5, mean_y - 5, mean_y + 5, dx, dy)
        return variance_x, variance_y

    def determine_bounds(self, mean_x, mean_y, variance_x, variance_y):
        sigma_x = math.sqrt(variance_x)
        sigma_y = math.sqrt(variance_y)
        print('sigma', sigma_x, sigma_y)
        x_min = mean_x - 3 * sigma_x
        x_max = mean_x + 3 * sigma_x
        y_min = mean_y - 3 * sigma_y
        y_max = mean_y + 3 * sigma_y
        print("bounds", x_min, x_max, y_min, y_max)
        return x_min, x_max, y_min, y_max


    def generate_samples_rejection(self, n, pdf=None, normalized_pdf=None, mean_x=0, mean_y=0, dx=0.1, dy=0.1):
        if normalized_pdf is None and pdf is not None:
            variance_x, variance_y = self.calculate_variance(pdf, mean_x, mean_y)
            x_min, x_max, y_min, y_max = self.determine_bounds(mean_x, mean_y, variance_x, variance_y)
            
            normalization_factor = self.calculate_normalization_factor(pdf, x_min, x_max, y_min, y_max, dx, dy)
            normalized_pdf = lambda x, y: pdf(x, y) / normalization_factor
        elif normalized_pdf is not None:
            variance_x, variance_y = self.calculate_variance(normalized_pdf, mean_x, mean_y)
            x_min, x_max, y_min, y_max = self.determine_bounds(mean_x, mean_y, variance_x, variance_y)
        else:
            raise ValueError("Either pdf or normalized_pdf must be provided.")
        
        x_values = [i * dx for i in range(int(x_min / dx), int(x_max / dx))]
        y_values = [i * dy for i in range(int(y_min / dy), int(y_max / dy))]
        max_pdf_value = max(normalized_pdf(x, y) for x in x_values for y in y_values)
        
        samples = []
        while len(samples) < n:
            x = self.LCG_float(x_min, x_max)
            y = self.LCG_float(y_min, y_max)

            pdf_value = normalized_pdf(x, y)
            
            threshold = self.LCG_float(0, max_pdf_value)
            if threshold < pdf_value:
                samples.append((x, y))
        
        return samples
    
    def generate_samples_discrete(self, n, possible_pairs, probability_list):
        total_prob = sum(probability_list)
        normalized_probs = [p / total_prob for p in probability_list]
        cumulative_probs = []
        cumulative_sum = 0

        for prob in normalized_probs:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)

        samples = []
        for _ in range(n):
            random_value = self.LCG_float(0, 1)

            for i, threshold in enumerate(cumulative_probs):
                if random_value <= threshold:
                    samples.append(possible_pairs[i])
                    break
        
        return samples
    
    def create_discrete(self, X, Y):
        probability_table = {}
        for x in X:
            for y in Y:
                pair = (x, y)
                probability_table[pair] = self.LCG_float(0, 1)
        
        total_sum = sum(probability_table.values())
        for pair in probability_table:
            probability_table[pair] /= total_sum

        return probability_table