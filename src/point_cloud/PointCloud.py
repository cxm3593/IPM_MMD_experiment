"""
PointCloud class for generating 3D point clouds with Gaussian distributions
Author: Chengyi Ma
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Union


class PointCloud:
    """
    A simple class for generating 3D point clouds with Gaussian distributions.
    """
    
    def __init__(self, points: Optional[np.ndarray] = None):
        """
        Initialize PointCloud object.
        
        Args:
            points: Optional numpy array of shape (n, 3) representing 3D points
        """
        self.points = points if points is not None else np.empty((0, 3))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PointCloud':
        """
        Alternate constructor: Creates a PointCloud instance from a pandas DataFrame.
        """
        if not all(col in df.columns for col in ['x', 'y', 'timestamp']):
            raise ValueError("DataFrame must contain 'x', 'y', and 'timestamp' columns.")
            
        # Convert DataFrame to NumPy array
        numpy_array = df[['x', 'y', 'timestamp']].to_numpy()

        # Call the primary constructor to create the instance
        return cls(numpy_array)
        
    @property
    def n_points(self) -> int:
        """Return the number of points in the cloud."""
        return self.points.shape[0]
    
    def generate_gaussian(self, 
                         n_points: int, 
                         mean: Union[float, np.ndarray] = 0.0,
                         std: Union[float, np.ndarray] = 1.0,
                         seed: Optional[int] = None) -> 'PointCloud':
        """
        Generate a 3D point cloud following a Gaussian distribution.
        
        Args:
            n_points: Number of points to generate
            mean: Mean of the distribution. Can be scalar or array of shape (3,)
            std: Standard deviation. Can be scalar or array of shape (3,)
            seed: Random seed for reproducibility
            
        Returns:
            Self for method chaining
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Handle mean parameter
        if np.isscalar(mean):
            mean = np.array([mean, mean, mean])
        else:
            mean = np.array(mean)
        
        # Handle std parameter
        if np.isscalar(std):
            std = np.array([std, std, std])
        else:
            std = np.array(std)
        
        # Generate points
        self.points = np.random.normal(mean, std, (n_points, 3))
        return self
    
    def visualize(self, 
                  title: str = "3D Point Cloud",
                  color: Union[str, np.ndarray] = 'blue',
                  size: Union[float, np.ndarray] = 3,
                  opacity: float = 0.7,
                  show_axes: bool = True,
                  width: int = 800,
                  height: int = 600) -> go.Figure:
        """
        Create an interactive 3D visualization of the point cloud using Plotly.
        
        Args:
            title: Title for the plot
            color: Color of points (string or array for color mapping)
            size: Size of points (scalar or array for size mapping)
            opacity: Opacity of points (0-1)
            show_axes: Whether to show axis labels and grid
            width: Width of the plot in pixels
            height: Height of the plot in pixels
            
        Returns:
            Plotly Figure object
        """
        if self.n_points == 0:
            raise ValueError("No points to visualize")
        
        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=self.points[:, 0],
            y=self.points[:, 1], 
            z=self.points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                colorscale='Viridis' if isinstance(color, np.ndarray) else None,
                showscale=isinstance(color, np.ndarray),
                colorbar=dict(title="Value") if isinstance(color, np.ndarray) else None
            ),
            text=[f'Point {i}' for i in range(self.n_points)],
            hovertemplate='<b>Point %{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        )])
        
        # Configure layout for GPU acceleration and interactivity
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title='X axis',
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                yaxis=dict(
                    title='Y axis', 
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                zaxis=dict(
                    title='Z axis',
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                camera=dict(
                    eye=dict(x=0, y=0, z=-1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=1, z=0),
                ),
                aspectmode='cube'
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=50),
            # Enable GPU acceleration
            uirevision='constant'
        )
        
        # Configure for better performance with large datasets
        fig.update_traces(
            marker_line_width=0,  # Remove marker borders for better performance
        )
        
        return fig
    
    def show(self, **kwargs):
        """
        Display the interactive 3D plot.
        
        Args:
            **kwargs: Arguments passed to visualize() method
        """
        fig = self.visualize(**kwargs)
        fig.show()

    def save_html(self, filename: str, **kwargs):
        """
        Save the interactive 3D plot as an HTML file.
        
        Args:
            filename: Output HTML filename
            **kwargs: Arguments passed to visualize() method
        """
        fig = self.visualize(**kwargs)
        fig.write_html(filename)
        # print(f"Interactive plot saved as {filename}")
    
    @staticmethod
    def translate(point_cloud: 'PointCloud', translation: Union[float, np.ndarray]) -> 'PointCloud':
        """
        Create a translated copy of the point cloud.
        
        Args:
            point_cloud: Input PointCloud object
            translation: Translation vector. Can be scalar or array of shape (3,)
            
        Returns:
            New PointCloud object with translated points
        """
        if point_cloud.n_points == 0:
            return PointCloud()
        
        # Handle translation parameter
        if np.isscalar(translation):
            translation = np.array([translation, translation, translation])
        else:
            translation = np.array(translation)
        
        # Create copy and translate
        new_points = point_cloud.points.copy() + translation
        return PointCloud(new_points)
    
    @staticmethod
    def scale(point_cloud: 'PointCloud', scale_factor: Union[float, np.ndarray]) -> 'PointCloud':
        """
        Create a scaled copy of the point cloud.
        
        Args:
            point_cloud: Input PointCloud object
            scale_factor: Scale factor. Can be scalar or array of shape (3,)
            
        Returns:
            New PointCloud object with scaled points
        """
        if point_cloud.n_points == 0:
            return PointCloud()
        
        # Handle scale parameter
        if np.isscalar(scale_factor):
            scale_factor = np.array([scale_factor, scale_factor, scale_factor])
        else:
            scale_factor = np.array(scale_factor)
        
        # Create copy and scale
        new_points = point_cloud.points.copy() * scale_factor
        return PointCloud(new_points)
    
    @staticmethod
    def jitter(point_cloud: 'PointCloud', 
               percentage: float = 0.1,
               distance: float = 0.1,
               seed: Optional[int] = None) -> 'PointCloud':
        """
        Create a jittered copy of the point cloud by shifting a percentage of points
        in random directions with uniform distance.
        
        Args:
            point_cloud: Input PointCloud object
            percentage: Percentage of points to jitter (0.0 to 1.0)
            distance: Uniform distance to shift selected points
            seed: Random seed for reproducibility
            
        Returns:
            New PointCloud object with jittered points
        """
        if point_cloud.n_points == 0:
            return PointCloud()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create copy of points
        new_points = point_cloud.points.copy()
        
        # Calculate number of points to jitter
        n_jitter = int(point_cloud.n_points * percentage)
        
        if n_jitter > 0:
            # Randomly select points to jitter
            jitter_indices = np.random.choice(point_cloud.n_points, n_jitter, replace=False)
            
            # Generate random unit directions for each selected point
            random_directions = np.random.randn(n_jitter, 3)
            # Normalize to unit vectors
            random_directions = random_directions / np.linalg.norm(random_directions, axis=1, keepdims=True)
            
            # Apply uniform distance shift
            shifts = random_directions * distance
            new_points[jitter_indices] += shifts
        
        return PointCloud(new_points)
    
    @staticmethod
    def plot_multiple(point_clouds: list,
                     labels: Optional[list] = None,
                     colors: Optional[list] = None,
                     sizes: Optional[list] = None,
                     title: str = "Multiple 3D Point Clouds",
                     opacity: float = 0.7,
                     show_axes: bool = True,
                     width: int = 800,
                     height: int = 600,
                     show_legend: bool = True) -> go.Figure:
        """
        Create an interactive 3D visualization of multiple point clouds using Plotly.
        
        Args:
            point_clouds: List of PointCloud objects to plot
            labels: List of labels for each point cloud (optional)
            colors: List of colors for each point cloud (optional)
            sizes: List of point sizes for each point cloud (optional)
            title: Title for the plot
            opacity: Opacity of points (0-1)
            show_axes: Whether to show axis labels and grid
            width: Width of the plot in pixels
            height: Height of the plot in pixels
            show_legend: Whether to show legend
            
        Returns:
            Plotly Figure object
        """
        if not point_clouds:
            raise ValueError("No point clouds provided")
        
        # Default colors if not provided
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        if colors is None:
            colors = default_colors[:len(point_clouds)]
        
        # Default labels if not provided
        if labels is None:
            labels = [f"Point Cloud {i+1}" for i in range(len(point_clouds))]
        
        # Default sizes if not provided
        if sizes is None:
            sizes = [3] * len(point_clouds)
        
        # Create figure
        fig = go.Figure()
        
        # Add each point cloud as a separate trace
        for i, pc in enumerate(point_clouds):
            if pc.n_points == 0:
                continue
                
            color = colors[i % len(colors)]
            size = sizes[i % len(sizes)]
            label = labels[i % len(labels)]
            
            fig.add_trace(go.Scatter3d(
                x=pc.points[:, 0],
                y=pc.points[:, 1],
                z=pc.points[:, 2],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0)  # Remove marker borders for better performance
                ),
                name=label,
                text=[f'{label} - Point {j}' for j in range(pc.n_points)],
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>'
            ))
        
        # Configure layout for GPU acceleration and interactivity
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title='X axis',
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                yaxis=dict(
                    title='Y axis',
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                zaxis=dict(
                    title='Z axis',
                    visible=show_axes,
                    showgrid=show_axes,
                    zeroline=show_axes
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=show_legend,
            # Enable GPU acceleration
            uirevision='constant'
        )
        
        return fig
    
    @staticmethod
    def show_multiple(point_clouds: list, **kwargs):
        """
        Display multiple point clouds in an interactive 3D plot.
        
        Args:
            point_clouds: List of PointCloud objects to plot
            **kwargs: Arguments passed to plot_multiple() method
        """
        fig = PointCloud.plot_multiple(point_clouds, **kwargs)
        fig.show()
    
    @staticmethod
    def save_multiple_html(point_clouds: list, filename: str, **kwargs):
        """
        Save multiple point clouds as an interactive HTML file.
        
        Args:
            point_clouds: List of PointCloud objects to plot
            filename: Output HTML filename
            **kwargs: Arguments passed to plot_multiple() method
        """
        fig = PointCloud.plot_multiple(point_clouds, **kwargs)
        fig.write_html(filename)
        # print(f"Multiple point clouds plot saved as {filename}")
    
    def __repr__(self) -> str:
        """String representation of the PointCloud."""
        return f"PointCloud({self.n_points} points)"
