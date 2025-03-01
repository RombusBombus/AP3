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