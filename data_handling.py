#data_handling.py Version 1.1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import pickle
from datetime import datetime
import config
import logging

logger = logging.getLogger(__name__)

class DataHandler:
  """Class for handling data operations and storage."""
  
  def __init__(self, base_dir: Optional[Path] = None):
      """
      Initialize DataHandler.
      
      Args:
          base_dir (Optional[Path]): Base directory for data storage
      """
      self.base_dir = Path(base_dir) if base_dir else Path(config.DATA_DIR)
      self.base_dir.mkdir(parents=True, exist_ok=True)
      self.cache: Dict[str, Any] = {}
      
  def save_results(
      self,
      data: Dict[str, Any],
      prefix: str = "analysis",
      formats: Optional[List[str]] = None
  ) -> Dict[str, Path]:
      """
      Save analysis results in multiple formats.
      
      Args:
          data: Data to save
          prefix: Prefix for filenames
          formats: List of formats to save (csv, json, pickle)
      
      Returns:
          Dict[str, Path]: Paths to saved files
      """
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      if formats is None:
          formats = ['csv', 'json', 'pickle']
          
      saved_files = {}
      
      try:
          for fmt in formats:
              filename = self.base_dir / f"{prefix}_{timestamp}.{fmt}"
              
              if fmt == 'csv':
                  if isinstance(data, pd.DataFrame):
                      data.to_csv(filename)
                  else:
                      pd.DataFrame(data).to_csv(filename)
              elif fmt == 'json':
                  with open(filename, 'w') as f:
                      json.dump(data, f, default=str)
              elif fmt == 'pickle':
                  with open(filename, 'wb') as f:
                      pickle.dump(data, f)
              
              saved_files[fmt] = filename
              logger.info(f"Saved data to {filename}")
              
          return saved_files
          
      except Exception as e:
          logger.error(f"Failed to save data: {str(e)}")
          raise
  
  def load_results(
      self,
      filename: Union[str, Path],
      format: Optional[str] = None
  ) -> Any:
      """
      Load saved results.
      
      Args:
          filename: Path to file
          format: File format (if not specified, inferred from extension)
      
      Returns:
          Loaded data
      """
      try:
          filepath = Path(filename)
          if format is None:
              format = filepath.suffix[1:]
              
          if format == 'csv':
              return pd.read_csv(filepath)
          elif format == 'json':
              with open(filepath, 'r') as f:
                  return json.load(f)
          elif format == 'pickle':
              with open(filepath, 'rb') as f:
                  return pickle.load(f)
          else:
              raise ValueError(f"Unsupported format: {format}")
              
      except Exception as e:
          logger.error(f"Failed to load data: {str(e)}")
          raise
  
  def preprocess_data(
      self,
      data: pd.DataFrame,
      operations: List[Dict[str, Any]]
  ) -> pd.DataFrame:
      """
      Preprocess data according to specified operations.
      
      Args:
          data: Input DataFrame
          operations: List of preprocessing operations
      
      Returns:
          Preprocessed DataFrame
      """
      try:
          df = data.copy()
          
          for op in operations:
              if op['type'] == 'normalize':
                  df[op['column']] = (df[op['column']] - df[op['column']].mean()) / df[op['column']].std()
              elif op['type'] == 'fillna':
                  df[op['column']].fillna(op['value'], inplace=True)
              elif op['type'] == 'drop':
                  df.drop(columns=op['columns'], inplace=True)
                  
          return df
          
      except Exception as e:
          logger.error(f"Failed to preprocess data: {str(e)}")
          raise
  
  def aggregate_results(
      self,
      data_list: List[Dict[str, Any]],
      key_metrics: List[str]
  ) -> pd.DataFrame:
      """
      Aggregate results from multiple analyses.
      
      Args:
          data_list: List of analysis results
          key_metrics: Metrics to include in aggregation
      
      Returns:
          Aggregated DataFrame
      """
      try:
          aggregated_data = []
          
          for data in data_list:
              metrics = {
                  metric: data.get(metric)
                  for metric in key_metrics
                  if metric in data
              }
              aggregated_data.append(metrics)
              
          return pd.DataFrame(aggregated_data)
          
      except Exception as e:
          logger.error(f"Failed to aggregate results: {str(e)}")
          raise
  
  def export_summary(
      self,
      data: Dict[str, Any],
      filename: str = "analysis_summary.txt"
  ) -> None:
      """
      Export a human-readable summary of results.
      
      Args:
          data: Analysis results
          filename: Output filename
      """
      try:
          filepath = self.base_dir / filename
          
          with open(filepath, 'w') as f:
              f.write("Analysis Summary\n")
              f.write("=" * 50 + "\n\n")
              
              for key, value in data.items():
                  if isinstance(value, (int, float)):
                      f.write(f"{key}: {value:.4f}\n")
                  else:
                      f.write(f"{key}: {value}\n")
                      
          logger.info(f"Exported summary to {filepath}")
          
      except Exception as e:
          logger.error(f"Failed to export summary: {str(e)}")
          raise