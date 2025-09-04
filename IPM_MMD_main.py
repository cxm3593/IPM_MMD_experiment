# Main program of Integral Probability Metrics (IPM) with Maximum Mean Discrepancy (MMD) 
# For 3D point clouds and event data
# Author: Chengyi Ma

import numpy as np
import os
from datetime import datetime
from src.experiments.Experiment import Experiment

experiment_config:dict = {
    "data"              : {
        "mean"      : 0.0,
        "std"       : 1.0,
        "seed"      : 42,
        "n_points"  : 1000,
    },
    "metrics"           : {},
    "visualize"         : True,
    "output_dir"       : None,
    "data_path"        : "data/selected_events_clean.txt",
    "modification"     : {
        "jitter"   : {
            "percentages": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #"distances": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "distances": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "seed": 42
        }
    }
}

def main():
    """
    Main entry point for IPM MMD experiments.
    
    """
    print("Running IPM MMD Experiment")
    # Step 1: set up the output environment
    # Create an unique output subdirectory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output path in config
    experiment_config["output_dir"] = output_dir

    # Step 2: Initialize and run the experiment
    experiment = Experiment(config=experiment_config)
    experiment.execute()


if __name__ == "__main__":
    main()