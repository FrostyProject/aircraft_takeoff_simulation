#plotting.py Version 1.1
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import config
import logging

logger = logging.getLogger(__name__)

class AdvancedPlotter:
  """Class for creating advanced visualizations."""
  
  def __init__(self):
      """Initialize plotter with style settings."""
      self.style_config = config.get_plot_settings()
      plt.style.use(self.style_config['style'])
      sns.set_palette("husl")
      
  def create_comparison_plot(
      self,
      data: Dict[str, pd.DataFrame],
      x_col: str,
      y_col: str,
      labels: Optional[List[str]] = None,
      title: str = "",
      save_path: Optional[Path] = None
  ) -> None:
      """
      Create comparison plot for multiple datasets.
      
      Args:
          data: Dictionary of DataFrames to compare
          x_col: Column name for x-axis
          y_col: Column name for y-axis
          labels: Labels for each dataset
          title: Plot title
          save_path: Path to save plot
      """
      try:
          fig, ax = plt.subplots(
              figsize=self.style_config['figsize'],
              dpi=self.style_config['dpi']
          )
          
          for i, (key, df) in enumerate(data.items()):
              label = labels[i] if labels else key
              ax.plot(
                  df[x_col],
                  df[y_col],
                  label=label,
                  linewidth=2
              )
          
          ax.set_xlabel(x_col)
          ax.set_ylabel(y_col)
          ax.set_title(title)
          ax.grid(True, alpha=0.3)
          ax.legend()
          
          if save_path:
              plt.savefig(save_path, dpi=self.style_config['dpi'])
              
          plt.close()
          
      except Exception as e:
          logger.error(f"Failed to create comparison plot: {str(e)}")
          raise
  
  def create_contour_plot(
      self,
      X: np.ndarray,
      Y: np.ndarray,
      Z: np.ndarray,
      title: str = "",
      save_path: Optional[Path] = None
  ) -> None:
      """
      Create contour plot for parameter sweeps.
      
      Args:
          X, Y: Meshgrid arrays
          Z: Values for contour
          title: Plot title
          save_path: Path to save plot
      """
      try:
          fig, ax = plt.subplots(
              figsize=self.style_config['figsize'],
              dpi=self.style_config['dpi']
          )
          
          contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
          fig.colorbar(contour, ax=ax)
          
          ax.set_title(title)
          ax.set_xlabel('Parameter 1')
          ax.set_ylabel('Parameter 2')
          
          if save_path:
              plt.savefig(save_path, dpi=self.style_config['dpi'])
              
          plt.close()
          
      except Exception as e:
          logger.error(f"Failed to create contour plot: {str(e)}")
          raise
  
  def create_sensitivity_heatmap(
      self,
      data: pd.DataFrame,
      save_path: Optional[Path] = None
  ) -> None:
      """
      Create heatmap for sensitivity analysis.
      
      Args:
          data: DataFrame with sensitivity data
          save_path: Path to save plot
      """
      try:
          plt.figure(
              figsize=self.style_config['figsize'],
              dpi=self.style_config['dpi']
          )
          
          sns.heatmap(
              data.corr(),
              annot=True,
              cmap='RdYlBu',
              center=0
          )
          
          plt.title('Parameter Sensitivity Correlation')
          
          if save_path:
              plt.savefig(save_path, dpi=self.style_config['dpi'])
              
          plt.close()
          
      except Exception as e:
          logger.error(f"Failed to create sensitivity heatmap: {str(e)}")
          raise
  
  def create_optimization_progress_plot(
      self,
      history: List[Dict[str, float]],
      metrics: List[str],
      save_path: Optional[Path] = None
  ) -> None:
      """
      Plot optimization progress over iterations.
      
      Args:
          history: List of optimization steps
          metrics: List of metrics to plot
          save_path: Path to save plot
      """
      try:
          n_metrics = len(metrics)
          fig, axes = plt.subplots(
              n_metrics,
              1,
              figsize=(10, 4*n_metrics),
              dpi=self.style_config['dpi']
          )
          
          if n_metrics == 1:
              axes = [axes]
          
          df = pd.DataFrame(history)
          
          for ax, metric in zip(axes, metrics):
              ax.plot(
                  df.index,
                  df[metric],
                  marker='o',
                  linewidth=2
              )
              ax.set_xlabel('Iteration')
              ax.set_ylabel(metric)
              ax.grid(True, alpha=0.3)
          
          plt.tight_layout()
          
          if save_path:
              plt.savefig(save_path, dpi=self.style_config['dpi'])
              
          plt.close()
          
      except Exception as e:
          logger.error(f"Failed to create optimization progress plot: {str(e)}")
          raise
  
  def create_performance_radar_plot(
      self,
      metrics: Dict[str, float],
      save_path: Optional[Path] = None
  ) -> None:
      """
      Create radar plot for performance metrics.
      
      Args:
          metrics: Dictionary of performance metrics
          save_path: Path to save plot
      """
      try:
          # Prepare the data
          categories = list(metrics.keys())
          values = list(metrics.values())
          
          # Number of variables
          num_vars = len(categories)
          
          # Compute angle for each axis
          angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
          angles += angles[:1]
          
          # Initialize the plot
          fig, ax = plt.subplots(
              figsize=(10, 10),
              subplot_kw=dict(projection='polar')
          )
          
          # Plot data
          values += values[:1]
          ax.plot(angles, values)
          ax.fill(angles, values, alpha=0.25)
          
          # Set the labels
          ax.set_xticks(angles[:-1])
          ax.set_xticklabels(categories)
          
          plt.title('Performance Metrics')
          
          if save_path:
              plt.savefig(save_path, dpi=self.style_config['dpi'])
              
          plt.close()
          
      except Exception as e:
          logger.error(f"Failed to create radar plot: {str(e)}")
          raise