def find_maximum(f, a, b, num_points=100):
    """
    Finds the maximum value of a function within a given interval [a, b].

    :param f: The function whose maximum value is to be found.
    :param a: The start of the interval.
    :param b: The end of the interval.
    :param num_points: The number of points to sample in the interval [a, b]. Default is 100.
    :return: The maximum value of the function in the interval [a, b].
    """
    step = (b - a) / num_points
    max_val = f(a)
    x = a
    while x <= b:
        max_val = max(max_val, f(x))
        x += step
    return max_val

def binary_search(f, target, a, b, tol=1e-5, max_iter=100):
    """
    Performs binary search to find the point where the function is closest to the target value.

    :param f: The function to search.
    :param target: The target value to search for.
    :param a: The start of the search interval.
    :param b: The end of the search interval.
    :param tol: The tolerance for the search. The function stops when the difference is smaller than `tol`. Default is 1e-5.
    :param max_iter: The maximum number of iterations. Default is 100.
    :return: The point in the interval [a, b] where `f` is closest to the target.
    """
    for _ in range(max_iter):
        mid = (a + b) / 2
        f_mid = f(mid)
        if abs(f_mid - target) < tol:
            return mid
        elif f_mid < target:
            a = mid
        else:
            b = mid
    return (a + b) / 2

def compute_support_from_pdf(pdf, epsilon=1e-6):
    """
    Computes the support of a probability distribution based on its PDF.

    :param pdf: The probability density function of the distribution.
    :param epsilon: The threshold below which the PDF is considered negligible. Default is 1e-6.
    :return: A tuple (left_bound, right_bound) representing the support of the distribution.
    :raises ValueError: If the PDF is not provided.
    """
    if not pdf:
        raise ValueError("PDF function not specified")

    left_bound = -99
    right_bound = 99

    while pdf(left_bound) > epsilon:
        left_bound -= 0.1

    while pdf(right_bound) > epsilon:
        right_bound += 0.1

    return left_bound, right_bound

def compute_support_from_cdf(cdf, epsilon=1e-6):
    """
    Computes the support of a probability distribution based on its CDF.

    :param cdf: The cumulative distribution function of the distribution.
    :param epsilon: The threshold for CDF values. The CDF should be close to 0 at the left bound and close to 1 at the right bound. Default is 1e-6.
    :return: A tuple (left_bound, right_bound) representing the support of the distribution.
    :raises ValueError: If the CDF is not provided.
    """
    if not cdf:
        raise ValueError("CDF function not specified")

    left_bound = -99
    right_bound = 99

    while cdf(left_bound) > epsilon:
        left_bound -= 0.1
    while cdf(right_bound) < 1 - epsilon:
        right_bound += 0.1
    return left_bound, right_bound
