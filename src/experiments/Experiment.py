# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

from src.point_cloud.PointCloud import PointCloud
from src.data_loader.DataLoader import load_event_data


class Experiment:
    def __init__(self, config: dict):
        self.config = config

    def execute(self):
        '''Start the experiment'''
        # Step 1: Data preparation. Set up the point cloud from generation or loaded event data
        self.prepare_data(path=self.config.get('data_path', None)) 

        # Step 1.1: Visualization (Optional)
        # Visualize base point cloud
        if self.config.get('visualize', True):
            output_path = self.config.get('output_path', 'base_point_cloud.html')
            self.base_point_cloud.save_html(output_path, title="Base Point Cloud")

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
            self.base_point_cloud = event_loader.convert_to_point_cloud()
        else:
            # Generate synthetic point cloud
            print(f"Generating synthetic point cloud")
            data_config = self.config.get('data', {})
            n_points = data_config.get('n_points', 1000)
            mean = data_config.get('mean', 0.0)
            std = data_config.get('std', 1.0)
            seed = data_config.get('seed', None)
            
            self.base_point_cloud = PointCloud()
            self.base_point_cloud.generate_gaussian(n_points=n_points, mean=mean, std=std, seed=seed)
        
        return self.base_point_cloud
