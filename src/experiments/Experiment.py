# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

# Experiment.py
# This module contains the class that setup and execute the complete experiment framework
# Author: Chengyi Ma

from src.point_cloud.PointCloud import PointCloud
from src.data_loader.DataLoader import load_event_data
from src.ipm.MMD import MMD
import os
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo


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
                # z-score normalization
                point_cloud.z_score_normalize()
                pt_vis_path = os.path.join(pt_vis_dir, f"base_point_cloud_{i}.html")
                point_cloud.save_html(pt_vis_path, title=f"Base Point Cloud {i}")
        
        # Step 2 DataTransform
        # Modify the base point cloud for further comparison
        
        # Jitter
        print(f"Generating jittered point clouds...")
        modification_config = self.config.get('modification', {})
        jitter_config = modification_config.get('jitter', {})
        seed = jitter_config.get('seed', 42)
        
        # Define parameter ranges for systematic exploration
        percentages = jitter_config.get('percentages', [1.0])
        distances = jitter_config.get('distances', [1.0])

        print(f"Creating {len(percentages)}x{len(distances)} = {len(percentages) * len(distances)} jitter combinations per point cloud")
        
        # Store jittered results for later use (MMD calculation, visualization, etc.)
        self.jittered_results = {}
        self.jitter_parameters = {
            'percentages': percentages,
            'distances': distances,
            'seed': seed
        }
        
        # Generate all jittered versions once
        for i, base_pc in enumerate(self.base_point_cloud_list):
            print(f"Generating jittered versions for point cloud {i} with {base_pc.n_points} points...")
            
            # Debug: only process the first few for now
            if i > 0:
                break

            self.jittered_results[i] = {}
            
            for percentage in percentages:
                for distance in distances:
                    # Create jittered version (only called once)
                    jittered_pc = PointCloud.jitter(base_pc, percentage=percentage, distance=distance, seed=seed)
                    
                    # Store jittered result for later use
                    self.jittered_results[i][(percentage, distance)] = jittered_pc

        # Save combined visualizations: base + all jittered variants in one plot
        if self.config.get('visualize', True):
            self._save_combined_point_cloud_variants()

        # Optional: Save comparison visualizations using stored results
        if self.config.get('visualize', True) and self.config.get('save_comparisons', False):
            self._save_jitter_comparisons()

        # Step 3 Metrics
        # For every point cloud, compute the metrics

        # MMD
        print(f"Computing MMD metrics using jittered results from previous step...")
        
        # Initialize MMD calculator
        mmd_calculator = MMD(kernel='linear', gamma=1.0)
        
        # Use stored jitter parameters
        percentages = self.jitter_parameters['percentages']
        distances = self.jitter_parameters['distances']
        
        # Process each point cloud with all parameter combinations
        for i, base_pc in enumerate(self.base_point_cloud_list):
            print(f"Computing MMD for point cloud {i} with {base_pc.n_points} points...")
            
            # Debug: Only process the first few for now
            if i > 0:
                break
            
            # Check if we have jittered results for this point cloud
            if i not in self.jittered_results:
                print(f"Warning: No jittered results found for point cloud {i}")
                continue
            
            # Initialize arrays to store MMD values for surface plot
            mmd_matrix = np.zeros((len(percentages), len(distances)))
            
            # Calculate MMD values using stored jittered versions
            for p_idx, percentage in enumerate(percentages):
                for d_idx, distance in enumerate(distances):
                    # Get the pre-computed jittered version
                    jittered_pc = self.jittered_results[i][(percentage, distance)]
                    
                    # Calculate MMD between base and jittered point cloud
                    mmd_value = mmd_calculator.compare_point_clouds(base_pc, jittered_pc)
                    mmd_matrix[p_idx, d_idx] = mmd_value
                    
                    # print(f"  p={percentage}, d={distance}: MMD={mmd_value:.6f}")
            
            # Create 3D surface plot
            output_dir = self.config.get('output_dir', "")
            surface_dir = os.path.join(output_dir, "mmd_surfaces")
            os.makedirs(surface_dir, exist_ok=True)
            
            # Create surface plot
            fig = go.Figure(data=[go.Surface(
                x=distances,
                y=percentages, 
                z=mmd_matrix,
                colorscale='Viridis',
                colorbar=dict(title="MMD Value")
            )])
            
            fig.update_layout(
                title=f'MMD Surface Plot - Point Cloud {i}',
                scene=dict(
                    xaxis_title='Distance',
                    yaxis_title='Percentage',
                    zaxis_title='MMD Value',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=800,
                height=600
            )
            
            # Save the surface plot
            surface_path = os.path.join(surface_dir, f"mmd_surface_point_cloud_{i}.html")
            fig.write_html(surface_path, include_plotlyjs='inline')
            print(f"Saved MMD surface plot to: {surface_path}")
            
            # Also save the MMD matrix as CSV for further analysis
            csv_path = os.path.join(surface_dir, f"mmd_matrix_point_cloud_{i}.csv")
            np.savetxt(csv_path, mmd_matrix, delimiter=',', 
                      header=f"MMD values for point cloud {i} - rows: percentages {percentages}, cols: distances {distances}")
            print(f"Saved MMD matrix to: {csv_path}")




        

    
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
            base_point_cloud_list = [base_point_cloud]

        self.base_point_cloud_list = base_point_cloud_list

        return base_point_cloud_list

    def _save_jitter_comparisons(self):
        """
        Save comparison visualizations using pre-computed jittered results.
        This method uses stored jittered point clouds to avoid recomputation.
        """
        print(f"Saving comparison visualizations using stored jittered results...")
        output_dir = self.config.get('output_dir', "")
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        percentages = self.jitter_parameters['percentages']
        distances = self.jitter_parameters['distances']
        
        # Process each point cloud with all parameter combinations
        for i, base_pc in enumerate(self.base_point_cloud_list):
            print(f"Creating comparison visualization for point cloud {i}...")
            
            # Debug: Only process the first one for now
            if i > 0:
                break
            
            # Check if we have jittered results for this point cloud
            if i not in self.jittered_results:
                print(f"Warning: No jittered results found for point cloud {i}")
                continue
            
            # Collect all jittered versions using stored results
            all_point_clouds = [base_pc]  # Start with base point cloud
            all_labels = [f"Base Point Cloud {i}"]
            all_colors = ['blue']
            
            # Use pre-computed jittered versions
            for percentage in percentages:
                for distance in distances:
                    jittered_pc = self.jittered_results[i][(percentage, distance)]
                    all_point_clouds.append(jittered_pc)
                    all_labels.append(f"Jittered p={percentage} d={distance}")
                    all_colors.append('red')
            
            # Create single comparison visualization with all modifications
            comparison_path = os.path.join(comparison_dir, f"point_cloud_{i}_all_modifications.html")
            PointCloud.save_multiple_html(
                all_point_clouds, 
                comparison_path,
                labels=all_labels,
                colors=all_colors,
                title=f"Point Cloud {i}: Base vs All Jitter Modifications (10x10 grid)"
            )

    def _save_combined_point_cloud_variants(self):
        """
        Save each base point cloud together with all its jittered variants in a single combined plot.
        Each point cloud gets its own HTML file showing the base (blue) + all jittered variants (red).
        """
        print(f"Saving combined visualizations: base + all jittered variants per point cloud...")
        output_dir = self.config.get('output_dir', "")
        combined_dir = os.path.join(output_dir, "combined_variants")
        os.makedirs(combined_dir, exist_ok=True)
        
        percentages = self.jitter_parameters['percentages']
        distances = self.jitter_parameters['distances']
        
        # Process each point cloud
        for i, base_pc in enumerate(self.base_point_cloud_list):
            print(f"Creating combined visualization for point cloud {i}...")
            
            # Check if we have jittered results for this point cloud
            if i not in self.jittered_results:
                print(f"Warning: No jittered results found for point cloud {i}")
                continue
            
            # Collect base + all jittered versions
            all_point_clouds = [base_pc]  # Start with base point cloud
            all_labels = [f"Base Point Cloud {i}"]
            all_colors = ['blue']
            
            # Add all jittered variants
            for percentage in percentages:
                for distance in distances:
                    jittered_pc = self.jittered_results[i][(percentage, distance)]
                    all_point_clouds.append(jittered_pc)
                    all_labels.append(f"Jittered p={percentage} d={distance}")
                    all_colors.append('red')
            
            # Create combined visualization with base + all variants
            combined_path = os.path.join(combined_dir, f"point_cloud_{i}_with_all_variants.html")
            PointCloud.save_multiple_html(
                all_point_clouds, 
                combined_path,
                labels=all_labels,
                colors=all_colors,
                title=f"Point Cloud {i}: Base + All Jittered Variants ({len(percentages)}x{len(distances)} combinations)"
            )
            
            print(f"  Saved combined visualization with base + {len(percentages) * len(distances)} variants")
        
        print(f"All combined visualizations saved to: {combined_dir}")
