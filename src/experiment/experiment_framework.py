"""
Experiment framework for systematic IPM/MMD testing
Author: Chengyi Ma
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Tuple
from abc import ABC, abstractmethod


class Transformation(ABC):
    """Abstract base class for point cloud transformations."""
    
    @abstractmethod
    def apply(self, point_cloud, **params) -> 'PointCloud':
        """Apply transformation to point cloud."""
        pass
    
    @abstractmethod
    def get_param_ranges(self) -> Dict[str, np.ndarray]:
        """Get parameter ranges for systematic testing."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get transformation name."""
        pass
    
    @abstractmethod
    def format_params(self, **params) -> str:
        """Format parameters for display/filename."""
        pass


class JitterTransformation(Transformation):
    """Jitter transformation with data-relative distance scaling."""
    
    def __init__(self, data_std: float = 1.0):
        self.data_std = data_std
        self.data_scale = np.sqrt(3) * data_std  # 3D Gaussian scaling
    
    def apply(self, point_cloud, percentage: float, distance: float, seed: int = 42):
        """Apply jitter transformation."""
        from src.point_cloud.PointCloud import PointCloud
        return PointCloud.jitter(point_cloud, percentage=percentage, distance=distance, seed=seed)
    
    def get_param_ranges(self) -> Dict[str, np.ndarray]:
        """Get parameter ranges for systematic testing."""
        return {
            'percentage': np.linspace(0.1, 1.0, 10),  # 10% to 100%
            'distance': np.linspace(0.1 * self.data_scale, 1.0 * self.data_scale, 5)  # Data-relative
        }
    
    def get_name(self) -> str:
        return "jitter"
    
    def format_params(self, percentage: float, distance: float, **kwargs) -> str:
        return f"p{percentage:.1f}_d{distance:.2f}"


class TranslationTransformation(Transformation):
    """Translation transformation."""
    
    def __init__(self, data_std: float = 1.0):
        self.data_std = data_std
        self.data_scale = data_std
    
    def apply(self, point_cloud, x_offset: float, y_offset: float, z_offset: float, **kwargs):
        """Apply translation transformation."""
        from src.point_cloud.PointCloud import PointCloud
        return PointCloud.translate(point_cloud, np.array([x_offset, y_offset, z_offset]))
    
    def get_param_ranges(self) -> Dict[str, np.ndarray]:
        """Get parameter ranges for systematic testing."""
        offset_range = np.linspace(0, 2.0 * self.data_scale, 5)  # 0 to 2σ
        return {
            'x_offset': offset_range,
            'y_offset': np.array([0.0]),  # Keep y,z fixed for systematic testing
            'z_offset': np.array([0.0])
        }
    
    def get_name(self) -> str:
        return "translation"
    
    def format_params(self, x_offset: float, y_offset: float, z_offset: float, **kwargs) -> str:
        return f"x{x_offset:.2f}_y{y_offset:.2f}_z{z_offset:.2f}"


class ScalingTransformation(Transformation):
    """Scaling transformation."""
    
    def apply(self, point_cloud, scale_factor: float, **kwargs):
        """Apply scaling transformation."""
        from src.point_cloud.PointCloud import PointCloud
        return PointCloud.scale(point_cloud, scale_factor)
    
    def get_param_ranges(self) -> Dict[str, np.ndarray]:
        """Get parameter ranges for systematic testing."""
        return {
            'scale_factor': np.linspace(0.5, 2.0, 8)  # 0.5x to 2x scaling
        }
    
    def get_name(self) -> str:
        return "scaling"
    
    def format_params(self, scale_factor: float, **kwargs) -> str:
        return f"s{scale_factor:.2f}"


class MMDExperiment:
    """Systematic MMD experiment framework."""
    
    def __init__(self, 
                 point_cloud_config: Dict[str, Any],
                 mmd_kernel: str = 'rbf',
                 mmd_gamma: float = 1.0,
                 output_base_dir: str = "output"):
        """
        Initialize experiment framework.
        
        Args:
            point_cloud_config: Configuration for original point cloud
            mmd_kernel: Kernel type for MMD calculation
            mmd_gamma: Kernel parameter
            output_base_dir: Base directory for outputs
        """
        self.point_cloud_config = point_cloud_config
        self.mmd_kernel = mmd_kernel
        self.mmd_gamma = mmd_gamma
        self.output_base_dir = output_base_dir
        
        # Initialize MMD calculator
        from src.ipm.MMD import MMD
        self.mmd_calc = MMD(kernel=mmd_kernel, gamma=mmd_gamma)
    
    def create_output_directory(self, transformation_name: str) -> str:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base_dir, f"{transformation_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def run_systematic_experiment(self, transformation: Transformation) -> Tuple[List[Dict], str]:
        """
        Run systematic experiment with given transformation.
        
        Args:
            transformation: Transformation to test
            
        Returns:
            Tuple of (results_list, output_directory)
        """
        print(f"=== IPM MMD Systematic Experiment: {transformation.get_name().upper()} ===")
        
        # Create output directory
        output_dir = self.create_output_directory(transformation.get_name())
        print(f"Output directory: {output_dir}")
        
        # Create original point cloud
        print("Creating original point cloud...")
        from src.point_cloud.PointCloud import PointCloud
        pc_original = PointCloud()
        pc_original.generate_gaussian(**self.point_cloud_config)
        print(f"Original point cloud: {pc_original}")
        
        # Get parameter ranges
        param_ranges = transformation.get_param_ranges()
        print(f"Parameter ranges: {param_ranges}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        total_experiments = len(param_combinations)
        print(f"Total experiments: {total_experiments}")
        print()
        
        # Store results
        results = []
        
        # Run experiments
        for i, params in enumerate(param_combinations, 1):
            print(f"Experiment {i}/{total_experiments}: {params}")
            
            # Apply transformation
            pc_transformed = transformation.apply(pc_original, **params)
            
            # Compute MMD
            mmd_distance = self.mmd_calc.compare_point_clouds(pc_original, pc_transformed)
            
            # Create visualization
            param_str = transformation.format_params(**params)
            filename = f"comparison_{param_str}.html"
            filepath = os.path.join(output_dir, filename)
            
            PointCloud.save_multiple_html(
                [pc_original, pc_transformed],
                filepath,
                labels=["Original", f"{transformation.get_name().title()} {param_str}"],
                colors=["blue", "red"],
                title=f"Point Cloud Comparison: {transformation.get_name().title()} ({param_str})"
            )
            
            # Store result
            result = {
                'experiment_id': i,
                'transformation': transformation.get_name(),
                'mmd_distance': mmd_distance,
                'filename': filename,
                **params  # Add all parameters
            }
            results.append(result)
        
        # Save results
        self._save_results(results, output_dir, transformation)
        
        return results, output_dir
    
    def _generate_param_combinations(self, param_ranges: Dict[str, np.ndarray]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _save_results(self, results: List[Dict], output_dir: str, transformation: Transformation):
        """Save experiment results."""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save detailed results
        detailed_csv = os.path.join(output_dir, 'experiment_results.csv')
        df.to_csv(detailed_csv, index=False)
        print(f"Detailed results saved to: {detailed_csv}")
        
        # Create pivot table if applicable
        if len(df.columns) >= 4:  # transformation, mmd_distance, + at least 2 params
            param_cols = [col for col in df.columns if col not in ['experiment_id', 'transformation', 'mmd_distance', 'filename']]
            if len(param_cols) >= 2:
                try:
                    pivot_table = df.pivot_table(
                        values='mmd_distance', 
                        index=param_cols[0], 
                        columns=param_cols[1:] if len(param_cols) > 1 else param_cols[1],
                        aggfunc='mean'
                    )
                    
                    print(f"\n=== MMD Distance Results Table ===")
                    print(f"Transformation: {transformation.get_name().title()}")
                    print(pivot_table.round(6))
                    
                    pivot_csv = os.path.join(output_dir, 'results_pivot.csv')
                    pivot_table.to_csv(pivot_csv)
                    print(f"Pivot table saved to: {pivot_csv}")
                except Exception as e:
                    print(f"Could not create pivot table: {e}")
        
        # Save configuration
        config_file = os.path.join(output_dir, 'experiment_config.txt')
        with open(config_file, 'w') as f:
            f.write("=== Experiment Configuration ===\n")
            f.write(f"Transformation: {transformation.get_name()}\n")
            f.write(f"Point Cloud Config: {self.point_cloud_config}\n")
            f.write(f"Parameter Ranges: {transformation.get_param_ranges()}\n")
            f.write(f"Total Experiments: {len(results)}\n")
            f.write(f"MMD Kernel: {self.mmd_kernel} with gamma={self.mmd_gamma}\n")
        print(f"Configuration saved to: {config_file}")
        
        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Total experiments: {len(results)}")
        print(f"MMD range: {df['mmd_distance'].min():.6f} to {df['mmd_distance'].max():.6f}")
        print(f"Mean MMD: {df['mmd_distance'].mean():.6f}")
        print(f"Std MMD: {df['mmd_distance'].std():.6f}")


# Convenience function for quick experiments
def run_jitter_experiment(point_cloud_config: Dict[str, Any], 
                         data_std: float = 1.0,
                         **kwargs) -> Tuple[List[Dict], str]:
    """Run jitter experiment with data-relative scaling."""
    experiment = MMDExperiment(point_cloud_config, **kwargs)
    transformation = JitterTransformation(data_std=data_std)
    return experiment.run_systematic_experiment(transformation)

def run_translation_experiment(point_cloud_config: Dict[str, Any],
                              data_std: float = 1.0,
                              **kwargs) -> Tuple[List[Dict], str]:
    """Run translation experiment."""
    experiment = MMDExperiment(point_cloud_config, **kwargs)
    transformation = TranslationTransformation(data_std=data_std)
    return experiment.run_systematic_experiment(transformation)

def run_scaling_experiment(point_cloud_config: Dict[str, Any],
                          **kwargs) -> Tuple[List[Dict], str]:
    """Run scaling experiment."""
    experiment = MMDExperiment(point_cloud_config, **kwargs)
    transformation = ScalingTransformation()
    return experiment.run_systematic_experiment(transformation)
