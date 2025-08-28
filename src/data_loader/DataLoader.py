"""
Event Data Loader for converting event stream data to 3D point clouds
Handles loading and preprocessing of event data from selected_events.txt
Author: Chengyi Ma
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict
import os
from pathlib import Path
import warnings

from src.point_cloud.PointCloud import PointCloud


class EventDataLoader:
    """
    Data loader for event stream data with conversion capabilities to 3D point clouds.
    
    The event data format is expected to be: timestamp, x, y, polarity
    where:
    - timestamp: microsecond timestamp
    - x, y: spatial coordinates (pixel positions)  
    - polarity: event polarity (typically 0 or 1, or -1 and 1)
    """
    
    def __init__(self, data_path: str, delimiter: str = ' ', 
                 column_names: Optional[List[str]] = None,
                 time_unit: str = 'microseconds'):
        """
        Initialize the event data loader.
        
        Args:
            data_path: Path to the event data file
            delimiter: Delimiter used in the data file (default: space)
            column_names: Custom column names (default: ['timestamp', 'x', 'y', 'polarity'])
            time_unit: Unit of timestamps ('microseconds', 'milliseconds', 'seconds')
        """
        self.data_path = Path(data_path)
        self.delimiter = delimiter
        self.time_unit = time_unit
        self.initial_timestamp = None # The first timestamp from original data

        # Default column names for event data
        self.column_names = ['timestamp', 'x', 'y', 'polarity']
        
        # Data storage
        self._raw_data = None
        self._processed_data = None
        self._data_stats = None
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load event data from file.
        
            
        Returns:
            DataFrame with loaded event data
        """
        print(f"Loading event data from: {self.data_path}")
        
        try:
            # Load data with pandas
            self._raw_data = pd.read_csv(
                self.data_path,
                delimiter=self.delimiter,
                names=self.column_names,
                dtype={
                    'timestamp': np.int64,
                    'x': np.int32, 
                    'y': np.int32,
                    'polarity': np.int8
                }
            )
            
            print(f"Loaded {len(self._raw_data)} events")
            print(f"Data shape: {self._raw_data.shape}")
            
            
            # Set initial timestamp to 0
            self.initial_timestamp = self._raw_data['timestamp'].iloc[0]
            self._raw_data['timestamp'] = self._raw_data['timestamp'] - self._raw_data['timestamp'].iloc[0]

            return self._raw_data.copy()
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
        
    def convert_to_point_cloud(self) -> PointCloud:
        """
        Convert raw event data into point cloud
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Extract coordinates directly
        x = self._raw_data['x'].values
        y = self._raw_data['y'].values
        t = self._raw_data['timestamp'].values
        
        # Create point cloud array [x, y, z] where z = timestamp
        point_cloud_array = np.column_stack([x, y, t])
        
        return PointCloud(point_cloud_array)

    def data_segmentation(self, segment_size: int) -> List[PointCloud]:
        """
        Segment the point cloud data into smaller chunks.
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Split the data into segments
        segments = []
        for i in range(0, len(self._raw_data), segment_size):
            segment = self._raw_data[i:i + segment_size]
            segments.append(PointCloud.from_dataframe(segment))

        return segments


def load_event_data(data_path: str, max_events: Optional[int] = None,
                   print_summary: bool = True) -> EventDataLoader:
    """
    Convenience function to quickly load event data.
    
    Args:
        data_path: Path to the event data file
        max_events: Maximum number of events to load
        print_summary: Whether to print data summary
        
    Returns:
        Configured EventDataLoader instance with loaded data
    """
    loader = EventDataLoader(data_path)
    loader.load_data()
    
    return loader



    
