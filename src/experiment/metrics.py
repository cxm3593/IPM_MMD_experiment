"""
Concrete implementations of distance metrics for the experiment framework
Author: Chengyi Ma
"""

import warnings
from typing import Dict, Any, Optional
from .base_metric import DistanceMetric


class MMDMetric(DistanceMetric):
    """Maximum Mean Discrepancy distance metric."""
    
    def __init__(self, kernel: str = 'rbf', gamma: float = 1.0):
        """Initialize MMD metric."""
        from src.ipm.MMD import MMD
        self.mmd = MMD(kernel=kernel, gamma=gamma)
        self.kernel = kernel
        self.gamma = gamma
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """Compare two point clouds using MMD."""
        return self.mmd.compare_point_clouds(pc1, pc2)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get MMD configuration."""
        return {
            'metric_type': 'mmd',
            'kernel': self.kernel,
            'gamma': self.gamma
        }
    
    def get_metric_name(self) -> str:
        return 'mmd'
    
    def get_short_name(self) -> str:
        return 'mmd'


class VanRossumMetric(DistanceMetric):
    """van Rossum distance metric."""
    
    def __init__(self, tau: float = 1.0, time_dimension: str = 'z', 
                 time_scale: float = 1.0, sampling_rate: float = 1000.0):
        """Initialize van Rossum metric."""
        try:
            from src.van_rossum.VanRossumDistance import VanRossumDistance
            self.vr = VanRossumDistance(
                tau=tau, 
                time_dimension=time_dimension,
                time_scale=time_scale,
                sampling_rate=sampling_rate
            )
            self.available = True
        except ImportError:
            warnings.warn("van Rossum distance not available. Install elephant library.")
            self.available = False
            self.vr = None
        
        self.tau = tau
        self.time_dimension = time_dimension
        self.time_scale = time_scale
        self.sampling_rate = sampling_rate
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """Compare two point clouds using van Rossum distance."""
        if not self.available:
            raise RuntimeError("van Rossum distance not available. Install elephant library.")
        return self.vr.compare_point_clouds(pc1, pc2)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get van Rossum configuration."""
        return {
            'metric_type': 'van_rossum',
            'tau': self.tau,
            'time_dimension': self.time_dimension,
            'time_scale': self.time_scale,
            'sampling_rate': self.sampling_rate,
            'available': self.available
        }
    
    def get_metric_name(self) -> str:
        return 'van_rossum'
    
    def get_short_name(self) -> str:
        return 'vr'
    
    def is_available(self) -> bool:
        """Check if van Rossum distance is available."""
        return self.available


class SpikeMetric(DistanceMetric):
    """SPIKE-distance metric."""
    
    def __init__(self, time_dimension: str = 'z', time_scale: float = 1.0,
                 distance_type: str = 'spike', interval: Optional[tuple] = None):
        """Initialize SPIKE-distance metric."""
        try:
            from src.spike_distance.SpikeDistance import SpikeDistance
            self.spike_dist = SpikeDistance(
                time_dimension=time_dimension,
                time_scale=time_scale,
                distance_type=distance_type,
                interval=interval
            )
            self.available = True
        except ImportError:
            warnings.warn("SPIKE-distance not available. Install PySpike library.")
            self.available = False
            self.spike_dist = None
        
        self.time_dimension = time_dimension
        self.time_scale = time_scale
        self.distance_type = distance_type
        self.interval = interval
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """Compare two point clouds using SPIKE-distance."""
        if not self.available:
            raise RuntimeError("SPIKE-distance not available. Install PySpike library.")
        return self.spike_dist.compare_point_clouds(pc1, pc2)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get SPIKE-distance configuration."""
        return {
            'metric_type': f'spike_distance_{self.distance_type}',
            'time_dimension': self.time_dimension,
            'time_scale': self.time_scale,
            'distance_type': self.distance_type,
            'interval': self.interval,
            'available': self.available
        }
    
    def get_metric_name(self) -> str:
        return f'spike_{self.distance_type}'
    
    def get_short_name(self) -> str:
        return f'sp_{self.distance_type[:3]}'
    
    def is_available(self) -> bool:
        """Check if SPIKE-distance is available."""
        return self.available


def create_default_metrics() -> Dict[str, DistanceMetric]:
    """Create default set of distance metrics."""
    metrics = {
        'mmd': MMDMetric(kernel='rbf', gamma=1.0),
        'van_rossum': VanRossumMetric(tau=1.0, time_dimension='z'),
        'spike_distance': SpikeMetric(time_dimension='z', distance_type='spike')
    }
    
    return metrics


def get_available_metrics() -> Dict[str, DistanceMetric]:
    """Get only available distance metrics (libraries installed)."""
    all_metrics = create_default_metrics()
    available = {}
    
    # MMD is always available
    available['mmd'] = all_metrics['mmd']
    
    # Check van Rossum
    if hasattr(all_metrics['van_rossum'], 'is_available') and all_metrics['van_rossum'].is_available():
        available['van_rossum'] = all_metrics['van_rossum']
    
    # Check SPIKE-distance
    if hasattr(all_metrics['spike_distance'], 'is_available') and all_metrics['spike_distance'].is_available():
        available['spike_distance'] = all_metrics['spike_distance']
    
    return available
