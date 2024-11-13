#visualization.py Version 1.1
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import config

class VisualizationError(Exception):
  """Custom exception for visualization errors."""
  pass

class TakeoffVisualizer:
  """Class to handle visualization of takeoff analysis results."""
  
  def __init__(self):
      """Initialize visualizer with default style settings."""
      self.style_config = config.get_plot_settings()
      plt.style.use(self.style_config['style'])
      sns.set_palette("husl")
      
  def setup_figure(
      self,
      figsize: Optional[Tuple[int, int]] = None
  ) -> Tuple[plt.Figure, plt.Axes]:
      """
      Create and setup a new figure.
      
      Args:
          figsize (Optional[Tuple[int, int]]): Figure size (width, height)
      
      Returns:
          Tuple[plt.Figure, plt.Axes]: Figure and axes objects
      """
      if figsize is None:
          figsize = self.style_config['figsize']
      
      fig, ax = plt.subplots(figsize=figsize, dpi=self.style_config['dpi'])
      return fig, ax
  
  def plot_trajectory(
      self,
      history: Dict[str, np.ndarray],
      save: bool = True,
      filename: str = "trajectory.png"
  ) -> None:
      """
      Plot takeoff trajectory including distance and velocity.
      
      Args:
          history (Dict[str, np.ndarray]): Time history of takeoff parameters
          save (bool): Whether to save the plot
          filename (str): Name of file to save plot
      """
      try:
          fig, (ax1, ax2) = plt.subplots(
              2, 1,
              figsize=(10, 8),
              dpi=self.style_config['dpi']
          )
          
          # Distance plot
          ax1.plot(
              history['time'],
              history['distance'],
              color=self.style_config['colors']['primary'],
              label='Distance'
          )
          ax1.set_xlabel('Time (s)')
          ax1.set_ylabel('Distance (ft)')
          ax1.grid(True, color=self.style_config['colors']['grid'])
          ax1.legend()
          
          # Velocity plot
          ax2.plot(
              history['time'],
              history['velocity'],
              color=self.style_config['colors']['secondary'],
              label='Velocity'
          )
          ax2.set_xlabel('Time (s)')
          ax2.set_ylabel('Velocity (ft/s)')
          ax2.grid(True, color=self.style_config['colors']['grid'])
          ax2.legend()
          
          plt.tight_layout()
          
          if save:
              plt.savefig(
                  Path(config.PLOT_DIR) / filename,
                  dpi=self.style_config['dpi']
              )
          
          plt.close()
          
      except Exception as e:
          raise VisualizationError(f"Failed to plot trajectory: {str(e)}")
  
  def plot_chord_sweep_results(
      self,
      results_df: pd.DataFrame,
      save: bool = True,
      filename: str = "chord_sweep.png"
  ) -> None:
      """
      Plot results of chord length sweep analysis.
      
      Args:
          results_df (pd.DataFrame): DataFrame containing sweep results
          save (bool): Whether to save the plot
          filename (str): Name of file to save plot
      """
      try:
          fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
              2, 2,
              figsize=(12, 10),
              dpi=self.style_config['dpi']
          )
          
          # Takeoff distance vs chord
          ax1.plot(
              results_df['chord'],
              results_df['takeoff_distance'],
              color=self.style_config['colors']['primary']
          )
          ax1.set_xlabel('Chord Length (ft)')
          ax1.set_ylabel('Takeoff Distance (ft)')
          ax1.grid(True, color=self.style_config['colors']['grid'])
          
          # Power required vs chord
          ax2.plot(
              results_df['chord'],
              results_df['power_required'],
              color=self.style_config['colors']['secondary']
          )
          ax2.set_xlabel('Chord Length (ft)')
          ax2.set_ylabel('Power Required (W)')
          ax2.grid(True, color=self.style_config['colors']['grid'])
          
          # CL vs chord
          ax3.plot(
              results_df['chord'],
              results_df['CL'],
              color=self.style_config['colors']['tertiary']
          )
          ax3.set_xlabel('Chord Length (ft)')
          ax3.set_ylabel('Lift Coefficient')
          ax3.grid(True, color=self.style_config['colors']['grid'])
          
          # Thrust vs chord
          ax4.plot(
              results_df['chord'],
              results_df['thrust'],
              color=self.style_config['colors']['quaternary']
          )
          ax4.set_xlabel('Chord Length (ft)')
          ax4.set_ylabel('Thrust (lbf)')
          ax4.grid(True, color=self.style_config['colors']['grid'])
          
          plt.tight_layout()
          
          if save:
              plt.savefig(
                  Path(config.PLOT_DIR) / filename,
                  dpi=self.style_config['dpi']
              )
          
          plt.close()
          
      except Exception as e:
          raise VisualizationError(
              f"Failed to plot chord sweep results: {str(e)}")
  
  def plot_sensitivity_analysis(
      self,
      sensitivity_data: Dict[str, pd.DataFrame],
      save: bool = True,
      filename: str = "sensitivity.png"
  ) -> None:
      """
      Plot sensitivity analysis results.
      
      Args:
          sensitivity_data (Dict[str, pd.DataFrame]): Sensitivity analysis results
          save (bool): Whether to save the plot
          filename (str): Name of file to save plot
      """
      try:
          n_params = len(sensitivity_data)
          fig, axes = plt.subplots(
              n_params,
              1,
              figsize=(10, 4*n_params),
              dpi=self.style_config['dpi']
          )
          
          if n_params == 1:
              axes = [axes]
          
          for ax, (param, df) in zip(axes, sensitivity_data.items()):
              ax.plot(
                  df[param],
                  df['takeoff_distance'],
                  color=self.style_config['colors']['primary']
              )
              ax.set_xlabel(f'{param}')
              ax.set_ylabel('Takeoff Distance (ft)')
              ax.grid(True, color=self.style_config['colors']['grid'])
              ax.set_title(f'Sensitivity to {param}')
          
          plt.tight_layout()
          
          if save:
              plt.savefig(
                  Path(config.PLOT_DIR) / filename,
                  dpi=self.style_config['dpi']
              )
          
          plt.close()
          
      except Exception as e:
          raise VisualizationError(
              f"Failed to plot sensitivity analysis: {str(e)}")
  
  def create_performance_summary(
      self,
      metrics: Dict[str, float],
      save: bool = True,
      filename: str = "performance_summary.png"
  ) -> None:
      """
      Create a summary plot of key performance metrics.
      
      Args:
          metrics (Dict[str, float]): Dictionary of performance metrics
          save (bool): Whether to save the plot
          filename (str): Name of file to save plot
      """
      try:
          fig, ax = plt.subplots(
              figsize=(10, 6),
              dpi=self.style_config['dpi']
          )
          
          metrics_to_plot = {
              k: v for k, v in metrics.items()
              if isinstance(v, (int, float))
          }
          
          y_pos = np.arange(len(metrics_to_plot))
          
          ax.barh(
              y_pos,
              list(metrics_to_plot.values()),
              color=self.style_config['colors']['primary']
          )
          
          ax.set_yticks(y_pos)
          ax.set_yticklabels(list(metrics_to_plot.keys()))
          ax.set_xlabel('Value')
          ax.set_title('Performance Metrics Summary')
          
          plt.tight_layout()
          
          if save:
              plt.savefig(
                  Path(config.PLOT_DIR) / filename,
                  dpi=self.style_config['dpi']
              )
          
          plt.close()
          
      except Exception as e:
          raise VisualizationError(
              f"Failed to create performance summary: {str(e)}")
  
  def plot_optimization_history(
      self,
      history: List[Dict[str, float]],
      save: bool = True,
      filename: str = "optimization_history.png"
  ) -> None:
      """
      Plot optimization history showing convergence.
      
      Args:
          history (List[Dict[str, float]]): List of optimization steps
          save (bool): Whether to save the plot
          filename (str): Name of file to save plot
      """
      try:
          df = pd.DataFrame(history)
          
          fig, (ax1, ax2) = plt.subplots(
              2, 1,
              figsize=(10, 8),
              dpi=self.style_config['dpi']
          )
          
          # Plot CL convergence
          ax1.plot(
              df.index,
              df['CL'],
              color=self.style_config['colors']['primary'],
              marker='o'
          )
          ax1.set_xlabel('Iteration')
          ax1.set_ylabel('Lift Coefficient')
          ax1.grid(True, color=self.style_config['colors']['grid'])
          
          # Plot thrust convergence
          ax2.plot(
              df.index,
              df['thrust'],
              color=self.style_config['colors']['secondary'],
              marker='o'
          )
          ax2.set_xlabel('Iteration')
          ax2.set_ylabel('Thrust (lbf)')
          ax2.grid(True, color=self.style_config['colors']['grid'])
          
          plt.tight_layout()
          
          if save:
              plt.savefig(
                  Path(config.PLOT_DIR) / filename,
                  dpi=self.style_config['dpi']
              )
          
          plt.close()
          
      except Exception as e:
          raise VisualizationError(
              f"Failed to plot optimization history: {str(e)}")