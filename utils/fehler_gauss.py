import numpy as np
from sympy import lambdify, diff


def fehler_gauss(formula, dependent_symbols, values, uncertainties, print_errors=False):
    u_formula = 0
    formula_l = lambdify(dependent_symbols, formula)
    for i, s in enumerate(dependent_symbols):
        derivative = diff(formula, s)
        derivative_l = lambdify(dependent_symbols, derivative)
        _u = derivative_l(*values) ** 2 * uncertainties[i] ** 2
        if print_errors:
            print(s, _u)
        u_formula += _u

    return formula_l(*values), np.sqrt(u_formula)


def weighted_mean(measurements, uncertainties):
    weights = 1 / uncertainties**2
    weighted_mean = np.dot(measurements, weights) / weights.sum()
    n = measurements.shape[0]
    internal_uncertainty = np.sqrt(1 / weights.sum())
    external_uncertainty = np.sqrt(
        np.dot(weights, (measurements - weighted_mean) ** 2) / ((n - 1) * weights.sum())
    )
    uncertainty = max(internal_uncertainty, external_uncertainty)
    return weighted_mean, uncertainty


def format_result(vals, us, exponents, digits, unit):
    results = []
    for val, u, exponent, digit in zip(vals, us, exponents, digits):
        val_fmt = str(np.round(val * 10**exponent, digit))
        u_fmt = str(np.round(u * 10**exponent, digit))
        results.append(
            r"\SI{" + val_fmt + f"({u_fmt[-digit:]})e{-exponent}" + r"}{" + unit + r"}"
        )
    return results
