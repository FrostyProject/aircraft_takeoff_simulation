#analysis.py Version 1.1
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy.interpolate import interp1d
import config
from optimization import optimize_CL_and_T, OptimizationError
from physics import calculate_drag_coefficient

class AnalysisError(Exception):
  """Custom exception for analysis errors."""
  pass

class TakeoffAnalysis:
  """Class to handle takeoff performance analysis."""
  
  def __init__(self):
      """Initialize analysis parameters."""
      self.results = {}
      self.optimization_history = []
      self.sensitivity_data = {}
  
  def analyze_single_configuration(
      self,
      chord: float,
      wingspan: float = config.WINGSPAN,
      weight: float = config.WEIGHT
  ) -> Dict[str, Any]:
      """
      Analyze takeoff performance for a single configuration.
      
      Args:
          chord (float): Wing chord length in feet
          wingspan (float): Wing span in feet
          weight (float): Aircraft weight in pounds
      
      Returns:
          Dict[str, Any]: Analysis results including optimal CL, thrust,
                         and performance metrics
      
      Raises:
          AnalysisError: If analysis fails
      """
      try:
          # Calculate derived parameters
          S = chord * wingspan  # wing area
          AR = wingspan / chord  # aspect ratio
          m = weight / config.GRAVITY  # mass in slugs
          
          # Run optimization
          best_CL, best_T, history, CD, opt_data = optimize_CL_and_T(
              m=m,
              S=S,
              rho=config.RHO,
              sigma=config.SIGMA,
              mu=config.MU,
              g=config.GRAVITY,
              dt=config.DT,
              t_max=config.T_MAX,
              power_limit=config.WATT_LIMIT,
              target_distance=config.TARGET_TAKEOFF_DISTANCE,
              AR=AR,
              epsilon=config.EPSILON,
              weight=weight
          )
          
          # Calculate performance metrics
          takeoff_distance = history['distance'][-1]
          takeoff_velocity = history['velocity'][-1]
          takeoff_time = history['time'][-1]
          
          # Calculate power required
          power_required = best_T * takeoff_velocity
          
          # Package results
          results = {
              'chord': chord,
              'wing_area': S,
              'aspect_ratio': AR,
              'CL': best_CL,
              'CD': CD,
              'thrust': best_T,
              'takeoff_distance': takeoff_distance,
              'takeoff_velocity': takeoff_velocity,
              'takeoff_time': takeoff_time,
              'power_required': power_required,
              'optimization_data': opt_data,
              'time_history': history
          }
          
          self.results[chord] = results
          return results
          
      except OptimizationError as e:
          raise AnalysisError(f"Optimization failed for chord={chord}: {str(e)}")
      except Exception as e:
          raise AnalysisError(f"Analysis failed for chord={chord}: {str(e)}")
  
  def perform_chord_sweep(
      self,
      chord_range: Optional[np.ndarray] = None
  ) -> pd.DataFrame:
      """
      Perform analysis across range of chord lengths.
      
      Args:
          chord_range (Optional[np.ndarray]): Array of chord values to analyze.
              If None, uses range from config.
      
      Returns:
          pd.DataFrame: Analysis results for all chord values
      """
      if chord_range is None:
          chord_range = config.get_chord_range()
      
      results_list = []
      for chord in chord_range:
          try:
              result = self.analyze_single_configuration(chord)
              results_list.append(result)
          except AnalysisError as e:
              if config.DEBUG_MODE:
                  print(f"Analysis failed for chord={chord}: {str(e)}")
              continue
      
      return pd.DataFrame(results_list)
  
  def analyze_sensitivity(
      self,
      base_chord: float,
      parameter_ranges: Dict[str, Tuple[float, float, int]]
  ) -> Dict[str, pd.DataFrame]:
      """
      Perform sensitivity analysis on key parameters.
      
      Args:
          base_chord (float): Baseline chord length
          parameter_ranges (Dict[str, Tuple[float, float, int]]):
              Dictionary of parameters to vary and their ranges
              Format: {param_name: (min_val, max_val, num_points)}
      
      Returns:
          Dict[str, pd.DataFrame]: Sensitivity analysis results
      """
      sensitivity_results = {}
      
      for param, (min_val, max_val, num_points) in parameter_ranges.items():
          param_values = np.linspace(min_val, max_val, num_points)
          param_results = []
          
          for value in param_values:
              try:
                  if param == 'chord':
                      result = self.analyze_single_configuration(value)
                  elif param == 'wingspan':
                      result = self.analyze_single_configuration(
                          base_chord, wingspan=value)
                  elif param == 'weight':
                      result = self.analyze_single_configuration(
                          base_chord, weight=value)
                  else:
                      continue
                  
                  param_results.append({
                      param: value,
                      **{k: v for k, v in result.items() 
                         if not isinstance(v, dict)}
                  })
                  
              except AnalysisError:
                  continue
          
          sensitivity_results[param] = pd.DataFrame(param_results)
      
      self.sensitivity_data = sensitivity_results
      return sensitivity_results
  
  def calculate_performance_metrics(self) -> Dict[str, float]:
      """
      Calculate overall performance metrics from analysis results.
      
      Returns:
          Dict[str, float]: Dictionary of performance metrics
      """
      if not self.results:
          raise AnalysisError("No analysis results available")
      
      try:
          # Convert results to DataFrame for analysis
          df = pd.DataFrame(self.results).T
          
          metrics = {
              'min_takeoff_distance': df['takeoff_distance'].min(),
              'max_takeoff_distance': df['takeoff_distance'].max(),
              'min_power_required': df['power_required'].min(),
              'max_power_required': df['power_required'].max(),
              'optimal_chord': df.loc[
                  df['takeoff_distance'].idxmin(), 'chord'],
              'mean_CL': df['CL'].mean(),
              'mean_CD': df['CD'].mean(),
              'mean_thrust': df['thrust'].mean()
          }
          
          return metrics
          
      except Exception as e:
          raise AnalysisError(f"Failed to calculate metrics: {str(e)}")
  
  def interpolate_results(
      self,
      parameter: str,
      new_points: np.ndarray
  ) -> np.ndarray:
      """
      Interpolate results for a given parameter.
      
      Args:
          parameter (str): Name of parameter to interpolate
          new_points (np.ndarray): Points at which to interpolate
      
      Returns:
          np.ndarray: Interpolated values
      """
      if not self.results:
          raise AnalysisError("No analysis results available")
      
      try:
          chords = np.array(list(self.results.keys()))
          values = np.array([r[parameter] 
                           for r in self.results.values()])
          
          interpolator = interp1d(
              chords,
              values,
              kind='cubic',
              bounds_error=False,
              fill_value='extrapolate'
          )
          
          return interpolator(new_points)
          
      except Exception as e:
          raise AnalysisError(
              f"Failed to interpolate {parameter}: {str(e)}")
  
  def export_results(self, filename: str) -> None:
      """
      Export analysis results to CSV file.
      
      Args:
          filename (str): Name of file to save results
      """
      try:
          df = pd.DataFrame(self.results).T
          df.to_csv(f"{config.DATA_DIR}/{filename}")
      except Exception as e:
          raise AnalysisError(f"Failed to export results: {str(e)}")