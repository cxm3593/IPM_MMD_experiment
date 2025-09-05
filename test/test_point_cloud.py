# Test Point Cloud
# This module contain test functions related to Point Cloud class

from src.point_cloud.PointCloud import PointCloud
from src.data_loader.DataLoader import EventDataLoader, load_event_data
import numpy as np
import plotly
import plotly.io as pio

test_data_path = "C:\\Users\\cxm3593\\Academic\\Workspace\\IPM_MMD_experiment\\data\\selected_events_clean.txt"

def test_main():
    """
    The main test function for PointCloud class.
    """
    # Render to browser directly
    pio.renderers.default = 'browser'

    # Test 1: Load data and generate point cloud
    event_loader = load_event_data(test_data_path)

    base_point_cloud_list = event_loader.data_segmentation(segment_size=2000)
    print(f"Number of point clouds generated: {len(base_point_cloud_list)}")

    # visualize the first point cloud for testing sample 
    pc = base_point_cloud_list[0]

    jittered_pc = PointCloud.jitter_image_space(pc, 0.5, 2, 50, seed=42)

    # Plot original and jittered point clouds
    PointCloud.show_multiple(
        point_clouds=[pc, jittered_pc],
        labels=['Base Cloud', 'Translated Cloud'],
        colors=['blue', 'red'],
        title="Comparing Two Point Clouds"
    )

if __name__ == "__main__":
    test_main()