import math
import csv

def f_prime(x):
    """Derivative of f(x) = x^2 - 4x + 4"""
    return 2 * x - 4


def run_optimizers(x0=0.0, eta=0.1, epsilon=1e-8, iterations=4):

    x_ada = x0
    x_rms = x0
    x_adam = x0

    G = 0.0
    gamma = 0.9
    Eg2 = 0.0

    beta1 = 0.9
    beta2 = 0.99
    m = 0.0
    v = 0.0

    results = []
    results.append([0, x0, x0, x0])

    for t in range(1, iterations + 1):

        # AdaGrad
        g_ada = f_prime(x_ada)
        G += g_ada**2
        x_ada = x_ada - (eta / (math.sqrt(G) + epsilon)) * g_ada

        # RMSprop
        g_rms = f_prime(x_rms)
        Eg2 = gamma * Eg2 + (1 - gamma) * g_rms**2
        x_rms = x_rms - (eta / (math.sqrt(Eg2) + epsilon)) * g_rms

        # Adam
        g_adam = f_prime(x_adam)
        m = beta1 * m + (1 - beta1) * g_adam
        v = beta2 * v + (1 - beta2) * g_adam**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x_adam = x_adam - (eta / (math.sqrt(v_hat) + epsilon)) * m_hat

        results.append([t, x_ada, x_rms, x_adam])

    with open("optimizer_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "AdaGrad", "RMSprop", "Adam"])
        writer.writerows(results)

    return results