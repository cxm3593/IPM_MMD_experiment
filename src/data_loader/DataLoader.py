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
        
        # Default column names for event data
        if column_names is None:
            self.column_names = ['timestamp', 'x', 'y', 'polarity']
        else:
            self.column_names = column_names
        
        # Data storage
        self._raw_data = None
        self._processed_data = None
        self._data_stats = None
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load_data(self, max_rows: Optional[int] = None, 
                  skip_rows: int = 0) -> pd.DataFrame:
        """
        Load event data from file.
        
        Args:
            max_rows: Maximum number of rows to load (None for all)
            skip_rows: Number of rows to skip from the beginning
            
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
                nrows=max_rows,
                skiprows=skip_rows,
                dtype={
                    'timestamp': np.int64,
                    'x': np.int32, 
                    'y': np.int32,
                    'polarity': np.int8
                }
            )
            
            print(f"Loaded {len(self._raw_data)} events")
            print(f"Data shape: {self._raw_data.shape}")
            
            # Calculate basic statistics
            self._calculate_data_stats()
            
            return self._raw_data.copy()
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def _calculate_data_stats(self):
        """Calculate and store basic statistics about the loaded data."""
        if self._raw_data is None:
            return
        
        self._data_stats = {
            'total_events': len(self._raw_data),
            'time_range': {
                'start': int(self._raw_data['timestamp'].min()),
                'end': int(self._raw_data['timestamp'].max()),
                'duration': int(self._raw_data['timestamp'].max() - self._raw_data['timestamp'].min())
            },
            'spatial_range': {
                'x_min': int(self._raw_data['x'].min()),
                'x_max': int(self._raw_data['x'].max()),
                'y_min': int(self._raw_data['y'].min()),
                'y_max': int(self._raw_data['y'].max()),
                'width': int(self._raw_data['x'].max() - self._raw_data['x'].min() + 1),
                'height': int(self._raw_data['y'].max() - self._raw_data['y'].min() + 1)
            },
            'polarity_distribution': self._raw_data['polarity'].value_counts().to_dict(),
            'event_rate': len(self._raw_data) / (self._raw_data['timestamp'].max() - self._raw_data['timestamp'].min()) * 1e6 if len(self._raw_data) > 1 else 0
        }
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dictionary with data statistics and information
        """
        if self._data_stats is None:
            if self._raw_data is None:
                raise RuntimeError("No data loaded. Call load_data() first.")
            self._calculate_data_stats()
        
        return self._data_stats.copy()
    
    def print_data_summary(self):
        """Print a human-readable summary of the loaded data."""
        if self._raw_data is None:
            print("No data loaded")
            return
        
        stats = self.get_data_info()
        
        print("\n=== Event Data Summary ===")
        print(f"File: {self.data_path.name}")
        print(f"Total events: {stats['total_events']:,}")
        
        # Time information
        duration_ms = stats['time_range']['duration'] / 1000
        print(f"Duration: {duration_ms:.2f} ms")
        print(f"Event rate: {stats['event_rate']:.1f} events/second")
        
        # Spatial information  
        print(f"Spatial range: {stats['spatial_range']['width']} x {stats['spatial_range']['height']} pixels")
        print(f"   X: [{stats['spatial_range']['x_min']}, {stats['spatial_range']['x_max']}]")
        print(f"   Y: [{stats['spatial_range']['y_min']}, {stats['spatial_range']['y_max']}]")
        
        # Polarity information
        print(f"Polarity distribution:")
        for polarity, count in stats['polarity_distribution'].items():
            percentage = (count / stats['total_events']) * 100
            print(f"   Polarity {polarity}: {count:,} events ({percentage:.1f}%)")
    
    def filter_events(self, time_start: Optional[int] = None,
                     time_end: Optional[int] = None,
                     x_range: Optional[Tuple[int, int]] = None,
                     y_range: Optional[Tuple[int, int]] = None,
                     polarity: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Filter events based on specified criteria.
        
        Args:
            time_start: Start timestamp (inclusive)
            time_end: End timestamp (inclusive)
            x_range: Tuple of (x_min, x_max) for spatial filtering
            y_range: Tuple of (y_min, y_max) for spatial filtering
            polarity: Polarity value(s) to keep
            
        Returns:
            Filtered DataFrame
        """
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        filtered_data = self._raw_data.copy()
        
        # Time filtering
        if time_start is not None:
            filtered_data = filtered_data[filtered_data['timestamp'] >= time_start]
        if time_end is not None:
            filtered_data = filtered_data[filtered_data['timestamp'] <= time_end]
        
        # Spatial filtering
        if x_range is not None:
            x_min, x_max = x_range
            filtered_data = filtered_data[
                (filtered_data['x'] >= x_min) & (filtered_data['x'] <= x_max)
            ]
        if y_range is not None:
            y_min, y_max = y_range
            filtered_data = filtered_data[
                (filtered_data['y'] >= y_min) & (filtered_data['y'] <= y_max)
            ]
        
        # Polarity filtering
        if polarity is not None:
            if isinstance(polarity, (list, tuple)):
                filtered_data = filtered_data[filtered_data['polarity'].isin(polarity)]
            else:
                filtered_data = filtered_data[filtered_data['polarity'] == polarity]
        
        print(f"Filtered: {len(self._raw_data)} → {len(filtered_data)} events")
        return filtered_data
    
    def get_time_windows(self, window_duration: float, 
                        overlap: float = 0.0) -> List[Tuple[int, int]]:
        """
        Generate time windows for temporal segmentation.
        
        Args:
            window_duration: Duration of each window (in same units as timestamps)
            overlap: Overlap fraction between windows (0.0 to 1.0)
            
        Returns:
            List of (start_time, end_time) tuples
        """
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        start_time = self._raw_data['timestamp'].min()
        end_time = self._raw_data['timestamp'].max()
        
        windows = []
        step = window_duration * (1 - overlap)
        
        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + window_duration, end_time)
            windows.append((int(current_start), int(current_end)))
            current_start += step
        
        return windows
    
    def prepare_for_pointcloud_conversion(self, normalization: str = 'minmax',
                                        time_scaling: str = 'relative') -> pd.DataFrame:
        """
        Prepare event data for conversion to 3D point cloud format.
        
        Args:
            normalization: Spatial normalization method ('minmax', 'zscore', 'none')
            time_scaling: Time scaling method ('relative', 'absolute', 'normalized')
            
        Returns:
            Processed DataFrame ready for point cloud conversion
        """
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        processed_data = self._raw_data.copy()
        
        # Spatial normalization
        if normalization == 'minmax':
            # Normalize x, y to [0, 1]
            processed_data['x_norm'] = (processed_data['x'] - processed_data['x'].min()) / \
                                     (processed_data['x'].max() - processed_data['x'].min())
            processed_data['y_norm'] = (processed_data['y'] - processed_data['y'].min()) / \
                                     (processed_data['y'].max() - processed_data['y'].min())
        elif normalization == 'zscore':
            # Z-score normalization
            processed_data['x_norm'] = (processed_data['x'] - processed_data['x'].mean()) / \
                                     processed_data['x'].std()
            processed_data['y_norm'] = (processed_data['y'] - processed_data['y'].mean()) / \
                                     processed_data['y'].std()
        elif normalization == 'none':
            processed_data['x_norm'] = processed_data['x']
            processed_data['y_norm'] = processed_data['y']
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Time scaling
        if time_scaling == 'relative':
            # Scale time to start from 0
            processed_data['time_norm'] = processed_data['timestamp'] - processed_data['timestamp'].min()
            # Convert to seconds
            processed_data['time_norm'] = processed_data['time_norm'] / 1e6
        elif time_scaling == 'normalized':
            # Normalize time to [0, 1]
            time_range = processed_data['timestamp'].max() - processed_data['timestamp'].min()
            processed_data['time_norm'] = (processed_data['timestamp'] - processed_data['timestamp'].min()) / time_range
        elif time_scaling == 'absolute':
            # Keep absolute timestamps (convert to seconds)
            processed_data['time_norm'] = processed_data['timestamp'] / 1e6
        else:
            raise ValueError(f"Unknown time scaling method: {time_scaling}")
        
        self._processed_data = processed_data
        
        print(f"Data prepared for point cloud conversion")
        print(f"   Spatial normalization: {normalization}")
        print(f"   Time scaling: {time_scaling}")
        
        return processed_data
    
    def convert_to_point_cloud(self, normalization: str = 'minmax',
                              time_scaling: str = 'relative',
                              use_processed_data: bool = False) -> np.ndarray:
        """
        Convert event data to 3D point cloud format.
        
        Maps event data (timestamp, x, y, polarity) to 3D points (x, y, z) where:
        - x, y remain as spatial coordinates (normalized)
        - timestamp becomes z-coordinate (scaled)
        - polarity is ignored for now
        
        Args:
            normalization: Spatial normalization method ('minmax', 'zscore', 'none')
            time_scaling: Time scaling method ('relative', 'absolute', 'normalized')
            use_processed_data: If True, use already processed data (ignores normalization params)
            
        Returns:
            numpy array of shape (N, 3) representing the 3D point cloud
        """
        if self._raw_data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        if use_processed_data and self._processed_data is not None:
            # Use already processed data
            data = self._processed_data
            point_cloud = np.column_stack([
                data['x_norm'].values,
                data['y_norm'].values, 
                data['time_norm'].values
            ])
            print(f"Point cloud created from processed data: {point_cloud.shape}")
        else:
            # Process data fresh for point cloud conversion
            processed_data = self.prepare_for_pointcloud_conversion(normalization, time_scaling)
            point_cloud = np.column_stack([
                processed_data['x_norm'].values,
                processed_data['y_norm'].values,
                processed_data['time_norm'].values
            ])
            print(f"Point cloud created: {point_cloud.shape}")
        
        # Print point cloud statistics
        print(f"   X range: [{point_cloud[:, 0].min():.4f}, {point_cloud[:, 0].max():.4f}]")
        print(f"   Y range: [{point_cloud[:, 1].min():.4f}, {point_cloud[:, 1].max():.4f}]")
        print(f"   Z range: [{point_cloud[:, 2].min():.4f}, {point_cloud[:, 2].max():.4f}]")
        
        return point_cloud
        
        return processed_data
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get the raw loaded data."""
        return self._raw_data.copy() if self._raw_data is not None else None
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get the processed data ready for point cloud conversion."""
        return self._processed_data.copy() if self._processed_data is not None else None
    
    def save_processed_data(self, output_path: str, format: str = 'csv'):
        """
        Save processed data to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'numpy')
        """
        if self._processed_data is None:
            raise RuntimeError("No processed data available. Call prepare_for_pointcloud_conversion() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._processed_data.to_csv(output_path, index=False)
        elif format == 'parquet':
            self._processed_data.to_parquet(output_path, index=False)
        elif format == 'numpy':
            np.save(output_path, self._processed_data.values)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Processed data saved to: {output_path}")


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
    loader.load_data(max_rows=max_events)
    
    if print_summary:
        loader.print_data_summary()
    
    return loader


def convert_events_to_point_cloud(data_path: str, max_events: Optional[int] = None,
                                 normalization: str = 'minmax', 
                                 time_scaling: str = 'relative') -> np.ndarray:
    """
    Convenience function to directly convert event data to point cloud.
    
    Args:
        data_path: Path to the event data file
        max_events: Maximum number of events to load
        normalization: Spatial normalization method ('minmax', 'zscore', 'none')
        time_scaling: Time scaling method ('relative', 'absolute', 'normalized')
        
    Returns:
        numpy array of shape (N, 3) representing the 3D point cloud
    """
    loader = EventDataLoader(data_path)
    loader.load_data(max_rows=max_events)
    return loader.convert_to_point_cloud(normalization, time_scaling)


# Example usage and testing
if __name__ == "__main__":
    # This will be used for testing the data loader
    data_path = "../../data/selected_events.txt"
    
    try:
        # Load a sample of the data for testing
        loader = load_event_data(data_path, max_events=10000)
        
        # Prepare for point cloud conversion
        processed_data = loader.prepare_for_pointcloud_conversion()
        
        print("\n=== Sample processed data ===")
        print(processed_data[['x_norm', 'y_norm', 'time_norm', 'polarity']].head(10))
        
    except Exception as e:
        print(f"Error in example: {e}")
