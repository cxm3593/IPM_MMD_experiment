# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

from src.point_cloud.PointCloud import PointCloud
from src.data_loader.DataLoader import load_event_data
import os


class Experiment:
    def __init__(self, config: dict):
        self.config = config
        self.base_point_cloud_list: list = []

    def execute(self):
        '''Start the experiment'''
        # Step 1: Data preparation. Set up the point cloud from generation or loaded event data
        self.prepare_data(path=self.config.get('data_path', None)) 

        # Step 1.1: Visualization (Optional)
        # Visualize base point cloud
        if self.config.get('visualize', True):
            print(f"Generating visualization for base point clouds...")
            output_dir = self.config.get('output_dir', "")

            pt_vis_dir = os.path.join(output_dir, "base_point_cloud")

            if pt_vis_dir:
                os.makedirs(pt_vis_dir, exist_ok=True)

            for i, point_cloud in enumerate(self.base_point_cloud_list):
                pt_vis_path = os.path.join(pt_vis_dir, f"base_point_cloud_{i}.html")
                point_cloud.save_html(pt_vis_path, title=f"Base Point Cloud {i}")
        pass
    
    def prepare_data(self, path=None):
        '''
        Prepare the data for the experiment
        path: str - path to the point cloud data. Will generate new data if not provided
        '''
        if path: 
            # Load event data and convert to point cloud
            print(f"Loading event data from: {path}")
            data_config = self.config.get('data', {})
            max_events = data_config.get('max_events', 1000)
            
            event_loader = load_event_data(path, max_events=max_events, print_summary=False)
            base_point_cloud_list = event_loader.data_segmentation(segment_size=2000) # Set for 2000 for now
        else:
            # Generate synthetic point cloud
            print(f"Generating synthetic point cloud")
            data_config = self.config.get('data', {})
            n_points = data_config.get('n_points', 1000)
            mean = data_config.get('mean', 0.0)
            std = data_config.get('std', 1.0)
            seed = data_config.get('seed', None)
            
            base_point_cloud = PointCloud()
            base_point_cloud.generate_gaussian(n_points=n_points, mean=mean, std=std, seed=seed)
            base_point_cloud_list.append(base_point_cloud)

        self.base_point_cloud_list = base_point_cloud_list

        return base_point_cloud_list
