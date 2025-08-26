"""
Multi-metric experiment framework that extends the existing MMD framework
Author: Chengyi Ma
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from .experiment_framework import Transformation
from .metrics import get_available_metrics, DistanceMetric


class MultiMetricExperiment:
    """
    Multi-metric experiment framework that runs multiple distance metrics
    on the same transformations and reports results separately.
    """
    
    def __init__(self, 
                 point_cloud_config: Dict[str, Any],
                 metrics: Optional[Dict[str, DistanceMetric]] = None,
                 output_base_dir: str = "output"):
        """
        Initialize multi-metric experiment framework.
        
        Args:
            point_cloud_config: Configuration for original point cloud
            metrics: Dictionary of metric_name -> DistanceMetric instances
            output_base_dir: Base directory for outputs
        """
        self.point_cloud_config = point_cloud_config
        self.output_base_dir = output_base_dir
        
        # Use provided metrics or get all available ones
        if metrics is None:
            self.metrics = get_available_metrics()
        else:
            self.metrics = metrics
        
        print(f"Initialized with {len(self.metrics)} metrics: {list(self.metrics.keys())}")
    
    def create_output_directory(self, transformation_name: str) -> str:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base_dir, f"multi_{transformation_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def run_systematic_experiment(self, transformation: Transformation) -> Tuple[Dict[str, List[Dict]], str]:
        """
        Run systematic experiment with multiple metrics.
        
        Args:
            transformation: Transformation to test
            
        Returns:
            Tuple of (results_dict_by_metric, output_directory)
        """
        print(f"=== Multi-Metric Systematic Experiment: {transformation.get_name().upper()} ===")
        
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
        print(f"Metrics to test: {list(self.metrics.keys())}")
        print()
        
        # Store results by metric
        results_by_metric = {metric_name: [] for metric_name in self.metrics.keys()}
        
        # Run experiments
        for i, params in enumerate(param_combinations, 1):
            print(f"Experiment {i}/{total_experiments}: {params}")
            
            # Apply transformation once
            pc_transformed = transformation.apply(pc_original, **params)
            
            # Test all metrics on the same transformation
            metric_distances = {}
            for metric_name, metric in self.metrics.items():
                try:
                    distance = metric.compare_point_clouds(pc_original, pc_transformed)
                    metric_distances[metric_name] = distance
                    print(f"  {metric_name}: {distance:.6f}")
                except Exception as e:
                    print(f"  {metric_name}: ERROR - {e}")
                    metric_distances[metric_name] = None
            
            # Create visualization (only once)
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
            
            # Store results for each metric
            for metric_name, distance in metric_distances.items():
                if distance is not None:
                    result = {
                        'experiment_id': i,
                        'transformation': transformation.get_name(),
                        'metric_distance': distance,  # Renamed to avoid confusion with jitter distance parameter
                        'filename': filename,
                        **params  # Add all parameters
                    }
                    results_by_metric[metric_name].append(result)
        
        # Save results for each metric separately
        self._save_multi_metric_results(results_by_metric, output_dir, transformation)
        
        return results_by_metric, output_dir
    
    def _generate_param_combinations(self, param_ranges: Dict[str, np.ndarray]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _save_multi_metric_results(self, results_by_metric: Dict[str, List[Dict]], 
                                  output_dir: str, transformation: Transformation):
        """Save experiment results for each metric separately."""
        
        # Save individual metric results
        for metric_name, results in results_by_metric.items():
            if not results:  # Skip if no valid results
                continue
                
            print(f"\n=== {metric_name.upper()} Results ===")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Save detailed results
            metric_dir = os.path.join(output_dir, metric_name)
            os.makedirs(metric_dir, exist_ok=True)
            
            detailed_csv = os.path.join(metric_dir, f'{metric_name}_results.csv')
            df.to_csv(detailed_csv, index=False)
            print(f"{metric_name} results saved to: {detailed_csv}")
            
            # Create pivot table if applicable
            self._create_pivot_table(df, metric_dir, metric_name, transformation)
            
            # Print summary statistics
            print(f"  Total experiments: {len(results)}")
            print(f"  Distance range: {df['distance'].min():.6f} to {df['distance'].max():.6f}")
            print(f"  Mean distance: {df['distance'].mean():.6f}")
            print(f"  Std distance: {df['distance'].std():.6f}")
        
        # Save combined summary
        self._save_combined_summary(results_by_metric, output_dir, transformation)
        
        # Save configuration
        self._save_experiment_config(output_dir, transformation)
        
        # Create comprehensive visualizations
        try:
            from .plotting import create_comprehensive_plots
            plot_dir = create_comprehensive_plots(results_by_metric, output_dir)
            print(f"📊 Comprehensive plots created in: {plot_dir}")
        except ImportError as e:
            print(f"⚠️  Plotting requires additional libraries: {e}")
            print("    Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
    
    def _create_pivot_table(self, df: pd.DataFrame, metric_dir: str, 
                           metric_name: str, transformation: Transformation):
        """Create pivot table for a single metric."""
        if len(df.columns) >= 4:  # transformation, distance, + at least 2 params
            param_cols = [col for col in df.columns if col not in ['experiment_id', 'transformation', 'distance', 'filename']]
            if len(param_cols) >= 2:
                try:
                    pivot_table = df.pivot_table(
                        values='distance', 
                        index=param_cols[0], 
                        columns=param_cols[1:] if len(param_cols) > 1 else param_cols[1],
                        aggfunc='mean'
                    )
                    
                    print(f"  {metric_name} Distance Table:")
                    print(pivot_table.round(6))
                    
                    pivot_csv = os.path.join(metric_dir, f'{metric_name}_pivot.csv')
                    pivot_table.to_csv(pivot_csv)
                    print(f"  Pivot table saved to: {pivot_csv}")
                except Exception as e:
                    print(f"  Could not create pivot table for {metric_name}: {e}")
    
    def _save_combined_summary(self, results_by_metric: Dict[str, List[Dict]], 
                              output_dir: str, transformation: Transformation):
        """Save a combined summary comparing all metrics."""
        print(f"\n=== COMBINED SUMMARY ===")
        
        # Create combined DataFrame for comparison
        combined_data = []
        for metric_name, results in results_by_metric.items():
            for result in results:
                combined_result = result.copy()
                combined_result['metric'] = metric_name
                combined_data.append(combined_result)
        
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            
            # Save combined results
            combined_csv = os.path.join(output_dir, 'combined_results.csv')
            combined_df.to_csv(combined_csv, index=False)
            print(f"Combined results saved to: {combined_csv}")
            
            # Create comparison table
            comparison_summary = combined_df.groupby('metric')['distance'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(6)
            
            print("Metric Comparison Summary:")
            print(comparison_summary)
            
            comparison_csv = os.path.join(output_dir, 'metric_comparison.csv')
            comparison_summary.to_csv(comparison_csv)
            print(f"Metric comparison saved to: {comparison_csv}")
    
    def _save_experiment_config(self, output_dir: str, transformation: Transformation):
        """Save experiment configuration."""
        config_file = os.path.join(output_dir, 'experiment_config.txt')
        with open(config_file, 'w') as f:
            f.write("=== Multi-Metric Experiment Configuration ===\n")
            f.write(f"Transformation: {transformation.get_name()}\n")
            f.write(f"Point Cloud Config: {self.point_cloud_config}\n")
            f.write(f"Parameter Ranges: {transformation.get_param_ranges()}\n")
            f.write(f"Metrics Used: {list(self.metrics.keys())}\n")
            f.write("\n=== Metric Configurations ===\n")
            
            for metric_name, metric in self.metrics.items():
                f.write(f"\n{metric_name}:\n")
                config_info = metric.get_config_info()
                for key, value in config_info.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"Configuration saved to: {config_file}")
