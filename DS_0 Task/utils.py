from collections import defaultdict

def create_random_probability_table(generator, x_values, y_values):
    """
    Creates a random probability table for a discrete 2D random variable.
    
    :param generator: Custom generator instance to generate probabilities.
    :param x_values: List of possible x values in the 2D random variable.
    :param y_values: List of possible y values in the 2D random variable.
    :return: Dictionary representing the probability table.
    """
    probability_table = defaultdict(float)
    total = 0.0
    
    for x in x_values:
        for y in y_values:
            prob = generator.generate(1, 0, 1)[0]
            probability_table[(x, y)] = prob
            total += prob

    for key in probability_table:
        probability_table[key] /= total
    
    return dict(probability_table)
