"""
Capillary Stiffness Modeling Package

This package implements statistical models for analyzing capillary stiffness
based on flow reduction measurements with robust error handling.

Author: Marcus Forst
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from lmfit import Parameters, minimize
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

@dataclass
class StiffnessModelResults:
    """Container for model fitting results"""
    baseline_stiffness: float
    age_effect: float
    random_effects: Dict[str, float]
    predicted_values: np.ndarray
    fit_result: any

class DataValidator:
    """Utility class for validating input data"""
    
    @staticmethod
    def validate_arrays(*arrays):
        """Check arrays for NaN, inf, and consistent lengths"""
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(f"Arrays have inconsistent lengths: {lengths}")
            
        for i, arr in enumerate(arrays):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values detected in array {i}")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Infinite values detected in array {i}")
    
    @staticmethod
    def validate_pressure(pressure):
        """Validate pressure values"""
        if np.any(pressure < 0):
            raise ValueError("Negative pressure values detected")
        if np.any(pressure > 1000):  # Adjust threshold as needed
            warnings.warn("Very high pressure values detected")
    
    @staticmethod
    def validate_flow_ratio(flow_ratio):
        """Validate flow ratio values"""
        if np.any(flow_ratio < 0):
            raise ValueError("Negative flow ratio values detected")
        if np.any(flow_ratio > 2):  # Typical flow ratios should be <= 1
            warnings.warn("Unusually high flow ratio values detected")

class CapillaryStiffnessModel:
    """Implementation of the capillary stiffness model with error handling"""
    
    def __init__(self, k0_init: float = 50, beta_age_init: float = 0.5):
        self.results = None
        self.k0_init = k0_init
        self.beta_age_init = beta_age_init
        
    def _safe_division(self, pressure, k_i):
        """Safely handle division to prevent numerical issues"""
        with np.errstate(divide='raise', invalid='raise'):
            try:
                ratio = pressure / k_i
                # Ensure ratio doesn't exceed 1 to prevent negative values in power
                ratio = np.minimum(ratio, 0.99)
                return (1 - ratio) ** 4
            except FloatingPointError as e:
                logger.error(f"Numerical error in flow calculation: {e}")
                raise ModelError("Numerical error in flow calculation")
    
    def _flow_model(self, params, pressure, data, age, participant):
        """Core model function with error handling"""
        try:
            k0 = params['k0']
            beta_age = params['beta_age']
            sigma = params['sigma']
            
            if k0 <= 0 or sigma <= 0:
                return 1e6 * np.ones_like(data)  # Penalty for invalid parameters
            
            participants = np.unique(participant)
            model = np.zeros_like(data)
            
            for p in participants:
                idx = participant == p
                age_p = age[idx][0]
                k_i = k0 + beta_age * age_p + params[f'u_{p}']
                
                if k_i <= 0:
                    return 1e6 * np.ones_like(data)  # Penalty for invalid stiffness
                
                model[idx] = self._safe_division(pressure[idx], k_i)
            
            residuals = (data - model) / sigma
            return residuals
            
        except Exception as e:
            logger.error(f"Error in flow model calculation: {e}")
            raise ModelError(f"Error in flow model calculation: {str(e)}")
    
    def _initialize_parameters(self, participants):
        """Initialize model parameters with bounds"""
        params = Parameters()
        params.add('k0', value=self.k0_init, min=1e-6)  # Prevent zero/negative stiffness
        params.add('beta_age', value=self.beta_age_init)
        params.add('sigma', value=1, min=1e-6)
        
        # Add random effects with bounds to prevent extreme values
        for p in participants:
            params.add(f'u_{p}', value=0, max=100, min=-100)
            
        return params
    
    def fit(self, pressure: np.ndarray, flow_ratio: np.ndarray, 
            age: np.ndarray, participant: np.ndarray) -> StiffnessModelResults:
        """
        Fit the stiffness model to data with comprehensive error checking
        
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
        """
        try:
            # Convert inputs to numpy arrays
            pressure = np.asarray(pressure, dtype=float)
            flow_ratio = np.asarray(flow_ratio, dtype=float)
            age = np.asarray(age, dtype=float)
            participant = np.asarray(participant)
            
            # Validate inputs
            logger.info("Validating input data...")
            DataValidator.validate_arrays(pressure, flow_ratio, age, participant)
            DataValidator.validate_pressure(pressure)
            DataValidator.validate_flow_ratio(flow_ratio)
            
            # Initialize parameters
            params = self._initialize_parameters(np.unique(participant))
            
            # Fit model with error handling
            logger.info("Fitting model...")
            result = minimize(self._flow_model, params,
                            args=(pressure, flow_ratio, age, participant),
                            nan_policy='raise')
            
            if not result.success:
                logger.warning(f"Fit may not have converged: {result.message}")
            
            # Extract and validate results
            k0 = result.params['k0'].value
            beta_age = result.params['beta_age'].value
            random_effects = {p: result.params[f'u_{p}'].value 
                            for p in np.unique(participant)}
            
            # Calculate predicted values
            predicted = np.zeros_like(flow_ratio)
            for p in np.unique(participant):
                idx = participant == p
                age_p = age[idx][0]
                k_i = k0 + beta_age * age_p + random_effects[p]
                predicted[idx] = self._safe_division(pressure[idx], k_i)
            
            self.results = StiffnessModelResults(
                baseline_stiffness=k0,
                age_effect=beta_age,
                random_effects=random_effects,
                predicted_values=predicted,
                fit_result=result
            )
            
            logger.info("Model fitting completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            raise ModelError(f"Model fitting failed: {str(e)}")

def calculate_flow_reduction_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate flow reduction ratios with error checking"""
    try:
        df = df.copy()
        
        # Check for required columns
        required_columns = ['Pressure', 'Capillary', 'Location', 'Log_Video_Median_Velocity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values
        if df['Log_Video_Median_Velocity'].isna().any():
            raise ValueError("NaN values found in velocity data")
        
        # Calculate baseline flow with error checking
        baseline_data = df[df['Pressure'] == 0.2]
        if len(baseline_data) == 0:
            raise ValueError("No baseline data (Pressure = 0.2) found")
            
        df['Baseline_Vel'] = (baseline_data
                             .groupby(['Capillary', 'Location'])['Log_Video_Median_Velocity']
                             .transform('mean'))
        
        # Check for zero baseline velocities
        if (df['Baseline_Vel'] == 0).any():
            raise ValueError("Zero baseline velocities detected")
            
        # Calculate reduction ratio
        df['Flow_Reduction_Ratio'] = df['Log_Video_Median_Velocity'] / df['Baseline_Vel']
        
        return df
        
    except Exception as e:
        logger.error(f"Error in flow reduction calculation: {e}")
        raise

def main():
    """Example usage with error handling"""
    try:
        # Load data
        df = pd.read_csv('C:\\Users\\gt8mar\\capillary-flow\\summary_df_nhp_video_medians.csv')
        
        # Preprocess data
        df = calculate_flow_reduction_ratio(df)
        
        # Prepare model inputs
        pressure = df['Pressure'].values
        flow_ratio = df['Flow_Reduction_Ratio'].values
        age = df['Age'].values
        participant = df['Participant'].values
        
        # Fit model
        model = CapillaryStiffnessModel()
        results = model.fit(pressure, flow_ratio, age, participant)
        
        # Print results
        print("\nModel Results:")
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
        
        return pd.DataFrame(stiffness_estimates)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()