# Main program of Integral Probability Metrics (IPM) with Maximum Mean Discrepancy (MMD) 
# For 3D point clouds and event data
# Author: Chengyi Ma

import numpy as np
from src.experiment.experiment_framework import (
    MMDExperiment, 
    JitterTransformation, 
    TranslationTransformation, 
    ScalingTransformation,
    run_jitter_experiment,
    run_translation_experiment, 
    run_scaling_experiment,
    run_multi_metric_jitter_experiment,
    run_multi_metric_translation_experiment,
    run_multi_metric_scaling_experiment
)

# Config
experiments_to_run = ["jitter"]  # Options: ["jitter", "translation", "scaling"] or any combination

point_cloud_config = {
    "n_points": 1000,
    "mean": 0,
    "std": 1,
    "seed": 42
}


def run_experiment(experiments_to_run: list):
    """
    Run specified experiments with flexible configuration.
    
    Args:
        experiments_to_run: List of experiment types to run 
                           Options: ["jitter", "translation", "scaling"]
        
    Returns:
        Dictionary with results for each experiment type
    """
    
    # Validate experiment types
    valid_experiments = ["jitter", "translation", "scaling"]
    for exp_type in experiments_to_run:
        if exp_type not in valid_experiments:
            raise ValueError(f"Unknown experiment type '{exp_type}'. Valid options: {valid_experiments}")
    
    # Setup
    data_std = point_cloud_config["std"]
    results = {}
    
    # Print header
    print("=== Multi-Metric IPM Comprehensive Experiment Suite ===\n")
    
    # Check available metrics
    from src.experiment.metrics import get_available_metrics
    available_metrics = get_available_metrics()
    print(f"Available metrics: {list(available_metrics.keys())}")

    
    print(f"Running experiments: {experiments_to_run}\n")
    
    # Run each requested experiment
    for exp_type in experiments_to_run:
        print(f"Running {exp_type.title()} Experiment...")
        
        try:
            # Run multi-metric experiment
            if exp_type == "jitter":
                exp_results, output_dir = run_multi_metric_jitter_experiment(
                    point_cloud_config, data_std=data_std
                )
            elif exp_type == "translation":
                exp_results, output_dir = run_multi_metric_translation_experiment(
                    point_cloud_config, data_std=data_std
                )
            elif exp_type == "scaling":
                exp_results, output_dir = run_multi_metric_scaling_experiment(
                    point_cloud_config
                )
            
            results[exp_type] = (exp_results, output_dir)
            print(f"{exp_type.title()} experiment complete: {output_dir}\n")
            
        except Exception as e:
            print(f"Error in {exp_type} experiment: {e}\n")
            results[exp_type] = (None, None)
    
    # Print summary
    print("=== Experiment Summary ===")
    print(f"Completed experiments: {len([k for k, v in results.items() if v[0] is not None])}/{len(experiments_to_run)}")
    print("Results saved in:")
    
    for exp_type, (exp_results, output_dir) in results.items():
        if output_dir:
            print(f"  - {exp_type.title()}: {output_dir}")
        else:
            print(f"  - {exp_type.title()}: Failed")
    
    
    return results

if __name__ == "__main__":
    # Run experiments based on configuration
    print("Starting IPM MMD Experiment Suite...\n")
    
    # Run the specified experiments
    results = run_experiment(experiments_to_run)
    
    # Print final summary
    successful_experiments = [exp for exp, (res, _) in results.items() if res is not None]
    failed_experiments = [exp for exp, (res, _) in results.items() if res is None]
    
    print(f"\n=== Final Results ===")
    print(f"Successful experiments: {successful_experiments}")
    if failed_experiments:
        print(f"Failed experiments: {failed_experiments}")
    
    print(f"\nAll experiment files are saved in the 'output/' directory.")
    
    # Optional: Return results for further analysis
    if successful_experiments:
        print(f"\nTo analyze results, check the HTML files in each experiment directory.")
        if any(results[exp][0] is not None for exp in successful_experiments):
            print(f"Multi-metric comparison plots and analysis are available.")