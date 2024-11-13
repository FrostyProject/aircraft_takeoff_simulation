#main.py Version 1.1
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

import config
from analysis import TakeoffAnalysis, AnalysisError
from visualization import TakeoffVisualizer, VisualizationError

# Setup logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
      logging.FileHandler('takeoff_analysis.log'),
      logging.StreamHandler(sys.stdout)
  ]
)
logger = logging.getLogger(__name__)

class TakeoffAnalyzer:
  """Main class to coordinate takeoff analysis and visualization."""
  
  def __init__(self):
      """Initialize analyzer components."""
      try:
          config.initialize()
          self.analysis = TakeoffAnalysis()
          self.visualizer = TakeoffVisualizer()
          self.results: Dict[str, Any] = {}
          
          logger.info("TakeoffAnalyzer initialized successfully")
          
      except Exception as e:
          logger.error(f"Failed to initialize TakeoffAnalyzer: {str(e)}")
          raise
  
  def run_complete_analysis(
      self,
      save_results: bool = True
  ) -> Dict[str, Any]:
      """
      Run complete analysis including chord sweep and sensitivity analysis.
      
      Args:
          save_results (bool): Whether to save results to files
      
      Returns:
          Dict[str, Any]: Complete analysis results
      """
      try:
          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          logger.info("Starting complete analysis")
          
          # Perform chord sweep analysis
          results_df = self.analysis.perform_chord_sweep()
          
          # Calculate performance metrics
          metrics = self.analysis.calculate_performance_metrics()
          
          # Perform sensitivity analysis
          sensitivity_ranges = {
              'chord': (config.MIN_CHORD, config.MAX_CHORD, 20),
              'wingspan': (config.WINGSPAN * 0.8, config.WINGSPAN * 1.2, 20),
              'weight': (config.WEIGHT * 0.8, config.WEIGHT * 1.2, 20)
          }
          
          sensitivity_results = self.analysis.analyze_sensitivity(
              metrics['optimal_chord'],
              sensitivity_ranges
          )
          
          # Store results
          self.results = {
              'chord_sweep': results_df,
              'metrics': metrics,
              'sensitivity': sensitivity_results,
              'timestamp': timestamp
          }
          
          # Create visualizations
          self._create_all_plots()
          
          # Save results if requested
          if save_results:
              self._save_results()
          
          logger.info("Complete analysis finished successfully")
          return self.results
          
      except Exception as e:
          logger.error(f"Complete analysis failed: {str(e)}")
          raise
  
  def _create_all_plots(self) -> None:
      """Create all visualization plots."""
      try:
          logger.info("Creating visualization plots")
          
          # Plot chord sweep results
          self.visualizer.plot_chord_sweep_results(
              self.results['chord_sweep'],
              filename=f"chord_sweep_{self.results['timestamp']}.png"
          )
          
          # Plot sensitivity analysis
          self.visualizer.plot_sensitivity_analysis(
              self.results['sensitivity'],
              filename=f"sensitivity_{self.results['timestamp']}.png"
          )
          
          # Create performance summary
          self.visualizer.create_performance_summary(
              self.results['metrics'],
              filename=f"performance_{self.results['timestamp']}.png"
          )
          
          # Plot trajectory for optimal configuration
          optimal_chord = self.results['metrics']['optimal_chord']
          optimal_result = self.analysis.results[optimal_chord]
          self.visualizer.plot_trajectory(
              optimal_result['time_history'],
              filename=f"trajectory_{self.results['timestamp']}.png"
          )
          
          logger.info("All plots created successfully")
          
      except Exception as e:
          logger.error(f"Failed to create plots: {str(e)}")
          raise
  
  def _save_results(self) -> None:
      """Save analysis results to files."""
      try:
          logger.info("Saving analysis results")
          
          # Create results directory if it doesn't exist
          Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
          
          # Save chord sweep results
          self.results['chord_sweep'].to_csv(
              Path(config.DATA_DIR) / 
              f"chord_sweep_{self.results['timestamp']}.csv"
          )
          
          # Save metrics
          pd.DataFrame([self.results['metrics']]).to_csv(
              Path(config.DATA_DIR) / 
              f"metrics_{self.results['timestamp']}.csv"
          )
          
          # Save sensitivity results
          for param, df in self.results['sensitivity'].items():
              df.to_csv(
                  Path(config.DATA_DIR) / 
                  f"sensitivity_{param}_{self.results['timestamp']}.csv"
              )
          
          logger.info("Results saved successfully")
          
      except Exception as e:
          logger.error(f"Failed to save results: {str(e)}")
          raise
  
  def analyze_specific_configuration(
      self,
      chord: float,
      wingspan: Optional[float] = None,
      weight: Optional[float] = None
  ) -> Dict[str, Any]:
      """
      Analyze a specific aircraft configuration.
      
      Args:
          chord (float): Wing chord length
          wingspan (Optional[float]): Wing span
          weight (Optional[float]): Aircraft weight
      
      Returns:
          Dict[str, Any]: Analysis results for specified configuration
      """
      try:
          logger.info(f"Analyzing specific configuration: chord={chord}")
          
          result = self.analysis.analyze_single_configuration(
              chord=chord,
              wingspan=wingspan if wingspan is not None else config.WINGSPAN,
              weight=weight if weight is not None else config.WEIGHT
          )
          
          # Plot trajectory for this configuration
          self.visualizer.plot_trajectory(
              result['time_history'],
              filename=f"trajectory_specific_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
          )
          
          logger.info("Specific configuration analysis completed")
          return result
          
      except Exception as e:
          logger.error(
              f"Failed to analyze specific configuration: {str(e)}")
          raise

def main():
  """Main execution function."""
  try:
      # Create analyzer instance
      analyzer = TakeoffAnalyzer()
      
      # Run complete analysis
      results = analyzer.run_complete_analysis(save_results=True)
      
      # Print summary of results
      optimal_chord = results['metrics']['optimal_chord']
      min_takeoff_distance = results['metrics']['min_takeoff_distance']
      
      print("\nAnalysis Results Summary:")
      print("-" * 50)
      print(f"Optimal chord length: {optimal_chord:.2f} ft")
      print(f"Minimum takeoff distance: {min_takeoff_distance:.2f} ft")
      print(f"Results saved in: {config.OUTPUT_DIR}")
      print("-" * 50)
      
      logger.info("Analysis completed successfully")
      return 0
      
  except Exception as e:
      logger.error(f"Program failed: {str(e)}")
      return 1

if __name__ == "__main__":
  sys.exit(main())