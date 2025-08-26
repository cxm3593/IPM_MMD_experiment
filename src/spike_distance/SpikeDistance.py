"""
SPIKE-distance implementation for spike trains/event data
Uses the PySpike library for SPIKE-distance calculation
Author: Chengyi Ma
"""

import numpy as np
from typing import Optional, Union, List
import warnings

try:
    import pyspike as spk
    PYSPIKE_AVAILABLE = True
except ImportError:
    PYSPIKE_AVAILABLE = False
    warnings.warn("PySpike library not available. Install with: pip install PySpike")

# Import the new utility functions
from ..utils.Utility import PointCloudToSpikeConverter, create_pyspike_spiketrain_from_pointcloud


class SpikeDistance:
    """
    SPIKE-distance calculator for point clouds converted to spike trains.
    
    SPIKE-distance is a parameter-free measure of spike train synchrony that
    captures both timing precision and rate differences between spike trains.
    """
    
    def __init__(self, time_dimension: str = 'z', time_scale: float = 1.0,
                 distance_type: str = 'spike', interval: Optional[tuple] = None):
        """
        Initialize SPIKE-distance calculator.
        
        Args:
            time_dimension: Which dimension to use as time ('x', 'y', or 'z')
            time_scale: Scaling factor for time dimension
            distance_type: Type of distance ('spike', 'isi', 'spike_sync')
            interval: Time interval for distance calculation (start, end)
        """
        if not PYSPIKE_AVAILABLE:
            raise ImportError("PySpike library is required for SPIKE-distance. "
                            "Install with: pip install PySpike")
        
        self.time_dimension = time_dimension.lower()
        self.time_scale = time_scale
        self.distance_type = distance_type.lower()
        self.interval = interval
        
        # Initialize the point cloud converter
        self.converter = PointCloudToSpikeConverter(
            time_dimension=time_dimension,
            time_scale=time_scale
        )
        
        # Validate distance type
        valid_types = ['spike', 'isi', 'spike_sync']
        if self.distance_type not in valid_types:
            raise ValueError(f"distance_type must be one of {valid_types}, got {distance_type}")
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """
        Compare two point clouds using SPIKE-distance.
        
        Args:
            pc1: First PointCloud object
            pc2: Second PointCloud object
            
        Returns:
            SPIKE-distance between the spike trains derived from point clouds
        """
        # Convert point clouds to spike trains using utility function
        spike_train1 = self.converter.pointcloud_to_pyspike_spiketrain(pc1)
        spike_train2 = self.converter.pointcloud_to_pyspike_spiketrain(pc2)
        
        # Calculate appropriate distance
        if self.distance_type == 'spike':
            distance = spk.spike_distance(spike_train1, spike_train2, interval=self.interval)
        elif self.distance_type == 'isi':
            distance = spk.isi_distance(spike_train1, spike_train2, interval=self.interval)
        elif self.distance_type == 'spike_sync':
            distance = spk.spike_sync(spike_train1, spike_train2, interval=self.interval)
        
        return float(distance)
    
    def _pointcloud_to_spike_times(self, pc):
        """
        Convert a point cloud to spike times for PySpike.
        DEPRECATED: Use converter.pointcloud_to_pyspike_spiketrain() instead.
        
        Args:
            pc: PointCloud object
            
        Returns:
            PySpike SpikeTrain object
        """
        warnings.warn("_pointcloud_to_spike_times is deprecated. Use converter.pointcloud_to_pyspike_spiketrain() instead.",
                     DeprecationWarning, stacklevel=2)
        return self.converter.pointcloud_to_pyspike_spiketrain(pc)
    
    def get_config_info(self) -> dict:
        """Get configuration information for this distance calculator."""
        return {
            'metric_type': f'spike_distance_{self.distance_type}',
            'time_dimension': self.time_dimension,
            'time_scale': self.time_scale,
            'distance_type': self.distance_type,
            'interval': self.interval
        }
    
    def compare_spike_trains_directly(self, spike_train1, spike_train2) -> float:
        """
        Compare two PySpike spike trains directly.
        
        Args:
            spike_train1: First PySpike SpikeTrain object
            spike_train2: Second PySpike SpikeTrain object
            
        Returns:
            SPIKE-distance between spike trains
        """
        if self.distance_type == 'spike':
            distance = spk.spike_distance(spike_train1, spike_train2, interval=self.interval)
        elif self.distance_type == 'isi':
            distance = spk.isi_distance(spike_train1, spike_train2, interval=self.interval)
        elif self.distance_type == 'spike_sync':
            distance = spk.spike_sync(spike_train1, spike_train2, interval=self.interval)
        
        return float(distance)
    
    def compare_multiple_point_clouds(self, point_clouds: List) -> np.ndarray:
        """
        Compare multiple point clouds pairwise using SPIKE-distance.
        
        Args:
            point_clouds: List of PointCloud objects
            
        Returns:
            Distance matrix of shape (n, n) where n is number of point clouds
        """
        n = len(point_clouds)
        distance_matrix = np.zeros((n, n))
        
        # Convert all point clouds to spike trains
        spike_trains = [self._pointcloud_to_spike_times(pc) for pc in point_clouds]
        
        # Calculate pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.compare_spike_trains_directly(spike_trains[i], spike_trains[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric matrix
        
        return distance_matrix
    
    def get_distance_profile(self, pc1, pc2, num_samples: int = 100) -> tuple:
        """
        Get the distance profile over time for two point clouds.
        
        Args:
            pc1: First PointCloud object
            pc2: Second PointCloud object
            num_samples: Number of time samples for the profile
            
        Returns:
            Tuple of (time_points, distance_values)
        """
        spike_train1 = self._pointcloud_to_spike_times(pc1)
        spike_train2 = self._pointcloud_to_spike_times(pc2)
        
        if self.distance_type == 'spike':
            profile = spk.spike_profile(spike_train1, spike_train2)
        elif self.distance_type == 'isi':
            profile = spk.isi_profile(spike_train1, spike_train2)
        else:
            raise ValueError(f"Profile not supported for distance_type: {self.distance_type}")
        
        # Sample the profile
        t_start = min(spike_train1.t_start, spike_train2.t_start)
        t_end = max(spike_train1.t_end, spike_train2.t_end)
        time_points = np.linspace(t_start, t_end, num_samples)
        distance_values = [profile.y(t) for t in time_points]
        
        return time_points, distance_values


def create_test_spike_trains():
    """Create test spike trains for validation."""
    if not PYSPIKE_AVAILABLE:
        raise ImportError("PySpike library not available")
    
    # Import here to avoid issues when library not available
    import pyspike as spk
    
    # Create simple test spike trains
    spike_times1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    spike_times2 = [0.15, 0.35, 0.55, 0.75, 0.95]  # Slightly shifted
    spike_times3 = [0.2, 0.4, 0.6, 0.8, 1.0]  # More shifted
    
    edges = [0.0, 1.5]
    
    spike_train1 = spk.SpikeTrain(spike_times1, edges)
    spike_train2 = spk.SpikeTrain(spike_times2, edges)
    spike_train3 = spk.SpikeTrain(spike_times3, edges)
    
    return [spike_train1, spike_train2, spike_train3]
