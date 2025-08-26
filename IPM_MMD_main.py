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
    run_scaling_experiment
)

# Config
point_cloud_config = {
    "n_points": 1000,
    "mean": 0,
    "std": 1,
    "seed": 42
}

def run_all_experiments():
    """Run systematic experiments for all transformations."""
    
    print("=== IPM MMD Comprehensive Experiment Suite ===\n")
    
    data_std = point_cloud_config["std"]
    
    # 1. Jitter Experiment (with data-relative scaling)
    print("🔄 Running Jitter Experiment...")
    jitter_results, jitter_dir = run_jitter_experiment(
        point_cloud_config, 
        data_std=data_std
    )
    print(f" Jitter experiment complete: {jitter_dir}\n")
    
    # 2. Translation Experiment  
    print(" Running Translation Experiment...")
    translation_results, translation_dir = run_translation_experiment(
        point_cloud_config,
        data_std=data_std
    )
    print(f" Translation experiment complete: {translation_dir}\n")
    
    # 3. Scaling Experiment
    print(" Running Scaling Experiment...")
    scaling_results, scaling_dir = run_scaling_experiment(
        point_cloud_config
    )
    print(f" Scaling experiment complete: {scaling_dir}\n")
    
    print(" All experiments completed!")
    print(f"Results saved in:")
    print(f"  - Jitter: {jitter_dir}")
    print(f"  - Translation: {translation_dir}")
    print(f"  - Scaling: {scaling_dir}")
    
    return {
        'jitter': (jitter_results, jitter_dir),
        'translation': (translation_results, translation_dir), 
        'scaling': (scaling_results, scaling_dir)
    }

def run_single_experiment(experiment_type: str = "jitter"):
    """Run a single experiment type."""
    
    data_std = point_cloud_config["std"]
    
    if experiment_type == "jitter":
        return run_jitter_experiment(point_cloud_config, data_std=data_std)
    elif experiment_type == "translation":
        return run_translation_experiment(point_cloud_config, data_std=data_std)
    elif experiment_type == "scaling":
        return run_scaling_experiment(point_cloud_config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

if __name__ == "__main__":
    # Choose experiment mode
    mode = "single"  # Change to "all" to run all experiments
    
    if mode == "all":
        # Run all experiments
        all_results = run_all_experiments()
    else:
        # Run single experiment (change experiment_type as needed)
        experiment_type = "translation"  # Options: "jitter", "translation", "scaling"
        results, output_dir = run_single_experiment(experiment_type)
        print(f"\n🎉 {experiment_type.title()} experiment completed!")
        print(f"Results saved in: {output_dir}")