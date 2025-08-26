"""
Utility functions for point cloud processing and spike train conversion
Author: Chengyi Ma
"""

import numpy as np
from typing import Optional, Union, List, Tuple
import warnings

# Optional imports for spike train libraries
try:
    import neo
    import quantities as pq
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False

try:
    import pyspike as spk
    PYSPIKE_AVAILABLE = True
except ImportError:
    PYSPIKE_AVAILABLE = False


class PointCloudToSpikeConverter:
    """
    Unified converter for transforming point clouds to spike trains.
    Supports both Neo (for van Rossum) and PySpike (for SPIKE-distance) formats.
    """
    
    def __init__(self, time_dimension: str = 'z', time_scale: float = 1.0,
                 sampling_rate: float = 1000.0):
        """
        Initialize the converter.
        
        Args:
            time_dimension: Which dimension to use as time ('x', 'y', or 'z')
            time_scale: Scaling factor for time dimension
            sampling_rate: Sampling rate for Neo spike trains (Hz)
        """
        self.time_dimension = time_dimension.lower()
        self.time_scale = time_scale
        self.sampling_rate = sampling_rate
        
        # Dimension mapping
        self.dim_map = {'x': 0, 'y': 1, 'z': 2}
        if self.time_dimension not in self.dim_map:
            raise ValueError(f"time_dimension must be 'x', 'y', or 'z', got {time_dimension}")
    
    def pointcloud_to_spike_times(self, pc) -> np.ndarray:
        """
        Extract spike times from a point cloud.
        
        Args:
            pc: PointCloud object
            
        Returns:
            Sorted array of spike times starting from 0
        """
        # Extract time dimension
        time_dim_idx = self.dim_map[self.time_dimension]
        spike_times = pc.points[:, time_dim_idx] * self.time_scale
        
        # Sort spike times and ensure they're positive
        spike_times = np.sort(spike_times)
        spike_times = spike_times - np.min(spike_times)  # Shift to start at 0
        
        # Add small offset to avoid zero start time issues
        spike_times = spike_times + 1e-6
        
        return spike_times
    
    def pointcloud_to_neo_spiketrain(self, pc):
        """
        Convert a point cloud to a Neo SpikeTrain object (for van Rossum distance).
        
        Args:
            pc: PointCloud object
            
        Returns:
            Neo SpikeTrain object
        """
        if not NEO_AVAILABLE:
            raise ImportError("Neo library is required for van Rossum distance. "
                            "Install with: pip install neo quantities")
        
        spike_times = self.pointcloud_to_spike_times(pc)
        
        # Determine t_stop (end time)
        if len(spike_times) > 0:
            t_stop = np.max(spike_times) + 1.0  # Add 1 second buffer
        else:
            t_stop = 1.0  # Default 1 second for empty spike trains
        
        # Create Neo SpikeTrain object
        spike_train = neo.SpikeTrain(
            times=spike_times * pq.s,
            t_start=0.0 * pq.s,
            t_stop=t_stop * pq.s,
            sampling_rate=self.sampling_rate * pq.Hz
        )
        
        return spike_train
    
    def pointcloud_to_pyspike_spiketrain(self, pc):
        """
        Convert a point cloud to a PySpike SpikeTrain object (for SPIKE-distance).
        
        Args:
            pc: PointCloud object
            
        Returns:
            PySpike SpikeTrain object
        """
        if not PYSPIKE_AVAILABLE:
            raise ImportError("PySpike library is required for SPIKE-distance. "
                            "Install with: pip install PySpike")
        
        spike_times = self.pointcloud_to_spike_times(pc)
        
        # Determine edges (start and end times)
        if len(spike_times) > 0:
            t_start = 0.0
            t_end = np.max(spike_times) + 1.0  # Add 1 second buffer
        else:
            t_start = 0.0
            t_end = 1.0  # Default 1 second for empty spike trains
            spike_times = np.array([])  # Empty array for no spikes
        
        # Create PySpike SpikeTrain object
        spike_train = spk.SpikeTrain(spike_times, edges=[t_start, t_end])
        
        return spike_train
    
    def compare_conversion_methods(self, pc) -> dict:
        """
        Compare different spike time extraction methods for analysis.
        
        Args:
            pc: PointCloud object
            
        Returns:
            Dictionary with conversion statistics
        """
        spike_times = self.pointcloud_to_spike_times(pc)
        
        stats = {
            'num_points': len(pc.points),
            'num_spike_times': len(spike_times),
            'time_dimension_used': self.time_dimension,
            'time_range': (float(np.min(spike_times)), float(np.max(spike_times))) if len(spike_times) > 0 else (0.0, 0.0),
            'mean_isi': float(np.mean(np.diff(spike_times))) if len(spike_times) > 1 else 0.0,
            'spike_rate': len(spike_times) / (np.max(spike_times) - np.min(spike_times)) if len(spike_times) > 1 else 0.0
        }
        
        return stats


def pointcloud_to_spike_times_simple(pc, time_dimension: str = 'z', time_scale: float = 1.0) -> np.ndarray:
    """
    Simple function to convert point cloud to spike times.
    
    Args:
        pc: PointCloud object
        time_dimension: Which dimension to use as time ('x', 'y', or 'z')
        time_scale: Scaling factor for time dimension
        
    Returns:
        Sorted array of spike times starting from 0
    """
    converter = PointCloudToSpikeConverter(time_dimension, time_scale)
    return converter.pointcloud_to_spike_times(pc)


def compare_spike_conversion_approaches(pc, approaches: List[str] = ['x', 'y', 'z']) -> dict:
    """
    Compare different dimensional approaches for spike time extraction.
    
    Args:
        pc: PointCloud object
        approaches: List of dimensions to try ['x', 'y', 'z']
        
    Returns:
        Dictionary comparing different approaches
    """
    results = {}
    
    for dim in approaches:
        try:
            converter = PointCloudToSpikeConverter(time_dimension=dim)
            stats = converter.compare_conversion_methods(pc)
            results[dim] = stats
        except Exception as e:
            results[dim] = {'error': str(e)}
    
    return results


def analyze_point_cloud_temporal_structure(pc, time_dimension: str = 'z') -> dict:
    """
    Analyze the temporal structure of a point cloud when treated as spike data.
    
    Args:
        pc: PointCloud object
        time_dimension: Which dimension to analyze as time
        
    Returns:
        Dictionary with temporal analysis
    """
    converter = PointCloudToSpikeConverter(time_dimension=time_dimension)
    spike_times = converter.pointcloud_to_spike_times(pc)
    
    if len(spike_times) < 2:
        return {'error': 'Insufficient spike times for analysis'}
    
    # Calculate inter-spike intervals
    isis = np.diff(spike_times)
    
    analysis = {
        'total_duration': float(np.max(spike_times) - np.min(spike_times)),
        'num_spikes': len(spike_times),
        'mean_firing_rate': len(spike_times) / (np.max(spike_times) - np.min(spike_times)),
        'isi_statistics': {
            'mean': float(np.mean(isis)),
            'std': float(np.std(isis)),
            'min': float(np.min(isis)),
            'max': float(np.max(isis)),
            'cv': float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0.0
        },
        'regularity_measure': {
            'cv_isi': float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0.0,
            'fano_factor': float(np.var(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0.0
        }
    }
    
    return analysis


# Convenience functions for backward compatibility
def create_neo_spiketrain_from_pointcloud(pc, time_dimension: str = 'z', 
                                        time_scale: float = 1.0, 
                                        sampling_rate: float = 1000.0):
    """Convenience function for Neo spike train creation."""
    converter = PointCloudToSpikeConverter(time_dimension, time_scale, sampling_rate)
    return converter.pointcloud_to_neo_spiketrain(pc)


def create_pyspike_spiketrain_from_pointcloud(pc, time_dimension: str = 'z', 
                                            time_scale: float = 1.0):
    """Convenience function for PySpike spike train creation."""
    converter = PointCloudToSpikeConverter(time_dimension, time_scale)
    return converter.pointcloud_to_pyspike_spiketrain(pc)


def validate_spike_conversion_libraries():
    """Check which spike train libraries are available."""
    status = {
        'neo_available': NEO_AVAILABLE,
        'pyspike_available': PYSPIKE_AVAILABLE,
        'can_use_van_rossum': NEO_AVAILABLE,
        'can_use_spike_distance': PYSPIKE_AVAILABLE
    }
    
    if not NEO_AVAILABLE:
        status['neo_install_command'] = "pip install neo quantities"
    
    if not PYSPIKE_AVAILABLE:
        status['pyspike_install_command'] = "pip install PySpike"
    
    return status