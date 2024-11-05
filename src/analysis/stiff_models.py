"""
Capillary Stiffness Modeling Package

This package implements statistical models for analyzing capillary stiffness
based on flow reduction measurements.

Author: Marcus Forst
"""

# models.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class StiffnessModelResults:
    """Container for model fitting results"""
    baseline_stiffness: float
    age_effect: float
    random_effects: Dict[str, float]
    predicted_values: np.ndarray
    fit_result: any

class CapillaryStiffnessModel:
    """Implementation of the capillary stiffness model"""
    
    def __init__(self):
        self.results = None
    
    def _flow_model(self, params, pressure, data, age, participant):
        """Core model function implementing the flow reduction equation"""
        k0 = params['k0']
        beta_age = params['beta_age']
        sigma = params['sigma']
        
        participants = np.unique(participant)
        model = np.zeros_like(data)
        
        for p in participants:
            idx = participant == p
            age_p = age[idx][0]
            k_i = k0 + beta_age * age_p + params[f'u_{p}']
            model[idx] = (1 - pressure[idx] / k_i) ** 4
        
        return (data - model) / sigma
    
    def fit(self, pressure: np.ndarray, flow_ratio: np.ndarray, 
            age: np.ndarray, participant: np.ndarray) -> StiffnessModelResults:
        """
        Fit the stiffness model to data
        
        Parameters:
        -----------
        pressure : array-like
            Applied pressure values
        flow_ratio : array-like
            Measured flow reduction ratios
        age : array-like
            Participant ages
        participant : array-like
            Participant identifiers
            
        Returns:
        --------
        StiffnessModelResults
            Fitted model results
        """
        from lmfit import Parameters, minimize
        
        # Initialize parameters
        params = Parameters()
        params.add('k0', value=50, min=0)
        params.add('beta_age', value=0.5)
        params.add('sigma', value=1, min=0.01)
        
        # Add random effects
        participants = np.unique(participant)
        for p in participants:
            params.add(f'u_{p}', value=0, vary=True)
            
        # Fit model
        result = minimize(self._flow_model, params, 
                        args=(pressure, flow_ratio, age, participant))
        
        # Extract results
        k0 = result.params['k0'].value
        beta_age = result.params['beta_age'].value
        random_effects = {p: result.params[f'u_{p}'].value for p in participants}
        
        # Calculate predicted values
        predicted = np.zeros_like(flow_ratio)
        for p in participants:
            idx = participant == p
            age_p = age[idx][0]
            k_i = k0 + beta_age * age_p + random_effects[p]
            predicted[idx] = (1 - pressure[idx] / k_i) ** 4
            
        self.results = StiffnessModelResults(
            baseline_stiffness=k0,
            age_effect=beta_age,
            random_effects=random_effects,
            predicted_values=predicted,
            fit_result=result
        )
        
        return self.results

# preprocessing.py
def calculate_flow_reduction_ratio(df):
    """Calculate flow reduction ratios from raw data"""
    df = df.copy()
    # Calculate baseline flow
    df['Baseline_Velocity'] = (df[df['Pressure'] == 0.2]
                         .groupby(['Capillary', 'Location'])['Log_Video_Median_Velocity']
                         .transform('mean'))
    # Calculate reduction ratio
    df['Flow_Reduction_Ratio'] = df['Log_Video_Median_Velocity'] / df['Baseline_Velocity']
    return df

# visualization.py
import matplotlib.pyplot as plt

def plot_flow_reduction_fits(pressure, flow_ratio, predicted, participant, 
                           xlabel='Pressure Applied', 
                           ylabel='Flow Reduction Ratio',
                           title='Observed vs. Predicted Flow Reduction Ratios'):
    """Plot observed and predicted flow reduction ratios"""
    plt.figure(figsize=(10, 6))
    
    for p in np.unique(participant):
        idx = participant == p
        plt.scatter(pressure[idx], flow_ratio[idx], 
                   label=f'Participant {p}', alpha=0.3)
        plt.plot(pressure[idx], predicted[idx], 
                color='black', alpha=0.1)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend().remove()
    return plt.gcf()

def plot_stiffness_vs_age(ages, stiffness,
                         xlabel='Age',
                         ylabel='Estimated Capillary Stiffness (k_i)',
                         title='Capillary Stiffness vs. Age'):
    """Plot relationship between stiffness and age"""
    plt.figure(figsize=(8, 6))
    plt.scatter(ages, stiffness)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return plt.gcf()

# example_usage.py
def main():
    """Example usage of the stiffness modeling package"""
    import pandas as pd
    
    # Load and preprocess data
    df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\summary_df_nhp_video_medians.csv')  # Replace with actual data loading
    df = calculate_flow_reduction_ratio(df)
    
    # Prepare data for modeling
    pressure = df['Pressure'].values
    flow_ratio = df['Flow_Reduction_Ratio'].values
    age = df['Age'].values
    participant = df['Participant'].values
    
    # Fit model
    model = CapillaryStiffnessModel()
    results = model.fit(pressure, flow_ratio, age, participant)
    
    # Print results
    print(f"Baseline stiffness (k0): {results.baseline_stiffness:.2f}")
    print(f"Effect of age on stiffness (beta_age): {results.age_effect:.2f}")
    
    # Calculate individual stiffness estimates
    stiffness_estimates = []
    for p in np.unique(participant):
        age_p = df[df['Participant'] == p]['Age'].iloc[0]
        k_i = (results.baseline_stiffness + 
               results.age_effect * age_p + 
               results.random_effects[p])
        stiffness_estimates.append({
            'Participant': p,
            'Age': age_p,
            'Stiffness': k_i
        })
    stiffness_df = pd.DataFrame(stiffness_estimates)
    
    # Create visualizations
    plot_flow_reduction_fits(pressure, flow_ratio, 
                           results.predicted_values, participant)
    plt.show()
    
    plot_stiffness_vs_age(stiffness_df['Age'], 
                         stiffness_df['Stiffness'])
    plt.show()

if __name__ == '__main__':
    main()