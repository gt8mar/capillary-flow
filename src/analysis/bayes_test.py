import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

def run_bayesian_model():
    # Simulated test data
    np.random.seed(42)
    n = 50
    pressure = np.linspace(0, 10, n)
    true_intercept = 5
    true_slope = -0.5
    noise = np.random.normal(0, 1, n)
    velocity = true_intercept + true_slope * pressure + noise

    # Define and fit Bayesian models with different priors
    traces = {}

    # Model 1: Weak prior
    with pm.Model() as weak_prior_model:
        intercept = pm.Normal("Intercept", mu=0, sigma=100)  # Weak prior for intercept
        slope = pm.Normal("Slope", mu=0, sigma=10)          # Weak prior for slope
        sigma = pm.HalfNormal("Sigma", sigma=2)            # Error term must be positive
        mu = intercept + slope * pressure
        y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=velocity)
        trace_weak = pm.sample(1000, return_inferencedata=True)
        traces['Weak Prior'] = trace_weak

    # Model 2: Informative prior
    with pm.Model() as informative_prior_model:
        intercept = pm.Normal("Intercept", mu=5, sigma=1)  # Informative prior for intercept
        slope = pm.Normal("Slope", mu=-0.5, sigma=0.1)     # Informative prior for slope
        sigma = pm.HalfNormal("Sigma", sigma=2)
        mu = intercept + slope * pressure
        y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=velocity)
        trace_informative = pm.sample(1000, return_inferencedata=True)
        traces['Informative Prior'] = trace_informative

    # Model 3: Improper prior
    with pm.Model() as improper_prior_model:
        intercept = pm.Uniform("Intercept", lower=0, upper=10)  # Improper prior for intercept
        slope = pm.Uniform("Slope", lower=-10, upper=10)       # Improper prior for slope
        sigma = pm.HalfNormal("Sigma", sigma=2)
        mu = intercept + slope * pressure
        y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=velocity)
        trace_improper = pm.sample(1000, return_inferencedata=True)
        traces['Improper Prior'] = trace_improper

    # Plot posterior distributions for all models
    for name, trace in traces.items():
        print(f"Posterior distributions for {name}:")
        az.plot_posterior(trace, figsize=(12, 6), show=True)

if __name__ == '__main__':
    run_bayesian_model()
