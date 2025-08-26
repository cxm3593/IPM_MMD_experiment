"""
Base interface for distance metrics in the experiment framework
Author: Chengyi Ma
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class DistanceMetric(ABC):
    """
    Abstract base class for all distance metrics.
    Provides a common interface for MMD, van Rossum, SPIKE-distance, etc.
    """
    
    @abstractmethod
    def compare_point_clouds(self, pc1, pc2) -> float:
        """
        Compare two point clouds and return a distance metric.
        
        Args:
            pc1: First PointCloud object
            pc2: Second PointCloud object
            
        Returns:
            Distance between the point clouds
        """
        pass
    
    @abstractmethod
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for this distance metric.
        
        Returns:
            Dictionary containing configuration details
        """
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Get the name of this distance metric.
        
        Returns:
            String name of the metric
        """
        pass
    
    def get_short_name(self) -> str:
        """
        Get a short name for filenames and labels.
        
        Returns:
            Short string name (default: metric_name)
        """
        return self.get_metric_name()
