"""
van Rossum Distance implementation for spike trains/event data
Uses the elephant library for van Rossum distance calculation
Author: Chengyi Ma
"""

import numpy as np
from typing import Optional, Union, List
import warnings

try:
    import elephant
    from elephant.spike_train_dissimilarity import van_rossum_distance
    from neo import SpikeTrain
    import quantities as pq
    ELEPHANT_AVAILABLE = True
except ImportError:
    ELEPHANT_AVAILABLE = False
    warnings.warn("Elephant library not available. Install with: pip install elephant")

# Import the new utility functions
from ..utils.Utility import PointCloudToSpikeConverter, create_neo_spiketrain_from_pointcloud


class VanRossumDistance:
    """
    van Rossum distance calculator for point clouds converted to spike trains.
    
    The van Rossum distance measures the distance between spike trains by convolving
    each spike train with an exponential kernel and computing the Euclidean distance
    between the resulting continuous functions.
    """
    
    def __init__(self, tau: float = 1.0, time_dimension: str = 'z', 
                 time_scale: float = 1.0, sampling_rate: float = 1000.0):
        """
        Initialize van Rossum distance calculator.
        
        Args:
            tau: Time constant for exponential kernel (in time units)
            time_dimension: Which dimension to use as time ('x', 'y', or 'z')
            time_scale: Scaling factor for time dimension
            sampling_rate: Sampling rate for spike train conversion (Hz)
        """
        if not ELEPHANT_AVAILABLE:
            raise ImportError("Elephant library is required for van Rossum distance. "
                            "Install with: pip install elephant")
        
        self.tau = tau * pq.s  # Convert to quantities
        self.time_dimension = time_dimension.lower()
        self.time_scale = time_scale
        self.sampling_rate = sampling_rate * pq.Hz
        
        # Initialize the point cloud converter
        self.converter = PointCloudToSpikeConverter(
            time_dimension=time_dimension,
            time_scale=time_scale,
            sampling_rate=sampling_rate
        )
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """
        Compare two point clouds using van Rossum distance.
        
        Args:
            pc1: First PointCloud object
            pc2: Second PointCloud object
            
        Returns:
            van Rossum distance between the spike trains derived from point clouds
        """
        # Convert point clouds to spike trains using utility function
        spike_train1 = self.converter.pointcloud_to_neo_spiketrain(pc1)
        spike_train2 = self.converter.pointcloud_to_neo_spiketrain(pc2)
        
        # Calculate van Rossum distance
        distance = van_rossum_distance([spike_train1, spike_train2], time_constant=self.tau)[0, 1]
        
        return float(distance)
    
    def _pointcloud_to_spike_train(self, pc):
        """
        Convert a point cloud to a spike train using specified time dimension.
        DEPRECATED: Use converter.pointcloud_to_neo_spiketrain() instead.
        
        Args:
            pc: PointCloud object
            
        Returns:
            Neo SpikeTrain object
        """
        warnings.warn("_pointcloud_to_spike_train is deprecated. Use converter.pointcloud_to_neo_spiketrain() instead.",
                     DeprecationWarning, stacklevel=2)
        return self.converter.pointcloud_to_neo_spiketrain(pc)
    
    def get_config_info(self) -> dict:
        """Get configuration information for this distance calculator."""
        return {
            'metric_type': 'van_rossum',
            'tau': float(self.tau.magnitude),
            'time_dimension': self.time_dimension,
            'time_scale': self.time_scale,
            'sampling_rate': float(self.sampling_rate.magnitude)
        }
    
    def compare_spike_trains_directly(self, spike_train1, spike_train2) -> float:
        """
        Compare two spike trains directly using van Rossum distance.
        
        Args:
            spike_train1: First Neo SpikeTrain object
            spike_train2: Second Neo SpikeTrain object
            
        Returns:
            van Rossum distance between spike trains
        """
        distance = van_rossum_distance([spike_train1, spike_train2], time_constant=self.tau)[0, 1]
        return float(distance)
    
    def compare_multiple_point_clouds(self, point_clouds: List) -> np.ndarray:
        """
        Compare multiple point clouds pairwise using van Rossum distance.
        
        Args:
            point_clouds: List of PointCloud objects
            
        Returns:
            Distance matrix of shape (n, n) where n is number of point clouds
        """
        n = len(point_clouds)
        distance_matrix = np.zeros((n, n))
        
        # Convert all point clouds to spike trains
        spike_trains = [self._pointcloud_to_spike_train(pc) for pc in point_clouds]
        
        # Calculate pairwise distances using elephant's optimized function
        full_distance_matrix = van_rossum_distance(spike_trains, time_constant=self.tau)
        
        return np.array(full_distance_matrix)


def create_test_spike_trains():
    """Create test spike trains for validation."""
    if not ELEPHANT_AVAILABLE:
        raise ImportError("Elephant library not available")
    
    # Import here to avoid issues when library not available
    from neo import SpikeTrain
    import quantities as pq
    
    # Create simple test spike trains
    spike_times1 = [0.1, 0.3, 0.5, 0.7, 0.9] * pq.s
    spike_times2 = [0.15, 0.35, 0.55, 0.75, 0.95] * pq.s  # Slightly shifted
    spike_times3 = [0.2, 0.4, 0.6, 0.8, 1.0] * pq.s  # More shifted
    
    t_stop = 1.5 * pq.s
    
    spike_train1 = SpikeTrain(spike_times1, t_start=0*pq.s, t_stop=t_stop)
    spike_train2 = SpikeTrain(spike_times2, t_start=0*pq.s, t_stop=t_stop)
    spike_train3 = SpikeTrain(spike_times3, t_start=0*pq.s, t_stop=t_stop)
    
    return [spike_train1, spike_train2, spike_train3]
