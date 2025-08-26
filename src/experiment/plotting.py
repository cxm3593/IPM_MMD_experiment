"""
Visualization module for multi-metric experiment results using Plotly
Author: Chengyi Ma
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional
import os


class MultiMetricPlotter:
    """
    Creates comprehensive interactive visualizations for multi-metric experiment results using Plotly.
    """
    
    def __init__(self, results_by_metric: Dict[str, List[Dict]], output_dir: str):
        """
        Initialize plotter with results data.
        
        Args:
            results_by_metric: Dictionary mapping metric names to result lists
            output_dir: Directory to save plots
        """
        self.results_by_metric = results_by_metric
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Create combined DataFrame for easier plotting
        self.combined_df = self._create_combined_dataframe()
        
        # Color palette for metrics
        self.colors = px.colors.qualitative.Set1
    
    def _create_combined_dataframe(self) -> pd.DataFrame:
        """Create a combined DataFrame from all metrics."""
        combined_data = []
        for metric_name, results in self.results_by_metric.items():
            for result in results:
                combined_result = result.copy()
                combined_result['metric'] = metric_name
                combined_data.append(combined_result)
        
        return pd.DataFrame(combined_data)
    
    def create_all_plots(self):
        """Create all visualization types."""
        print("Creating comprehensive interactive visualizations...")
        
        # 1. 3D Surface plots for each metric
        self.plot_3d_surfaces()
        
        # 2. Interactive heatmaps for each metric
        self.plot_interactive_heatmaps()
        
        # 3. Comparative line plots
        self.plot_comparative_lines()
        
        # 4. Metric correlation analysis
        self.plot_metric_correlations()
        
        # 5. Statistical distribution comparison
        self.plot_distributions()
        
        # 6. Interactive dashboard
        self.plot_interactive_dashboard()
        
        # 7. Combined 3D visualization
        self.plot_combined_3d()
        
        print(f"All interactive plots saved in: {self.plot_dir}")
    
    def plot_3d_surfaces(self):
        """Create interactive 3D surface plots for each metric."""
        print("Creating 3D surface plots...")
        
        # Get parameter ranges
        percentages = sorted(self.combined_df['percentage'].unique())
        jitter_distances = sorted(self.combined_df['distance'].unique())  # Renamed for clarity
        
        metrics = list(self.results_by_metric.keys())
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=len(metrics),
            specs=[[{'type': 'surface'} for _ in metrics]],
            subplot_titles=[f'{metric.upper()}' for metric in metrics],
            horizontal_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            # Get metric data
            metric_data = self.combined_df[self.combined_df['metric'] == metric]
            
            # Create pivot table for surface
            pivot_table = metric_data.pivot_table(
                values='metric_distance',  # Updated column name
                index='percentage', 
                columns='distance',
                aggfunc='mean'
            )
            
            # Create surface plot
            surface = go.Surface(
                z=pivot_table.values,
                x=jitter_distances,  # X corresponds to columns (distance values)
                y=percentages,       # Y corresponds to rows (percentage values)  
                colorscale='viridis',
                name=metric.upper(),
                showscale=(i == len(metrics))  # Only show colorbar for last plot
            )
            
            fig.add_trace(surface, row=1, col=i)
        
        fig.update_layout(
            title_text="3D Surface Plots - Metric Response to Jitter Parameters",
            height=600,
        )

        for i in range(len(metrics)):
            scene_name = 'scene' if i == 0 else f'scene{i+1}'
            fig.update_layout(**{
                scene_name: dict(
                    xaxis_title="Jitter Distance",
                    yaxis_title="Percentage", 
                    zaxis_title="Metric Distance"
                )
            })

        fig.write_html(os.path.join(self.plot_dir, '3d_surfaces.html'))
    
    def plot_interactive_heatmaps(self):
        """Create interactive heatmap visualizations for each metric."""
        print("Creating interactive heatmaps...")
        
        metrics = list(self.results_by_metric.keys())
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[f'{metric.upper()}' for metric in metrics],
            horizontal_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            # Create pivot table for heatmap
            metric_data = self.combined_df[self.combined_df['metric'] == metric]
            pivot_table = metric_data.pivot_table(
                values='metric_distance',  # Updated column name
                index='percentage', 
                columns='distance',
                aggfunc='mean'
            )
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=pivot_table.values,
                x=[f'{d:.3f}' for d in pivot_table.columns],
                y=[f'{p:.1f}' for p in pivot_table.index],
                colorscale='viridis',
                name=metric.upper(),
                showscale=(i == len(metrics)),
                hovertemplate=f'<b>{metric.upper()}</b><br>' +
                             'Percentage: %{y}<br>' +
                             'Jitter Distance: %{x}<br>' +
                             'Metric Distance: %{z:.6f}<extra></extra>'
            )
            
            fig.add_trace(heatmap, row=1, col=i)
        
        fig.update_layout(
            title_text="Interactive Heatmaps - Metric Response Patterns",
            height=500
        )
        
        # Update axes labels
        for i in range(len(metrics)):
            fig.update_xaxes(title_text="Jitter Distance", row=1, col=i+1)
            fig.update_yaxes(title_text="Percentage", row=1, col=i+1)
        
        fig.write_html(os.path.join(self.plot_dir, 'interactive_heatmaps.html'))
    
    def plot_comparative_lines(self):
        """Create interactive line plots comparing metrics across parameters."""
        print("Creating comparative line plots...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Metric Response vs Percentage (avg over jitter distances)',
                'Metric Response vs Jitter Distance (avg over percentages)', 
                'Normalized Metric Comparison',
                'Box Plot Distribution'
            ],
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Plot 1: Distance vs Percentage (averaged over jitter distances)
        percentage_avg = self.combined_df.groupby(['metric', 'percentage'])['metric_distance'].mean().reset_index()
        
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = percentage_avg[percentage_avg['metric'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['percentage'],
                    y=metric_data['metric_distance'],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(color=self.colors[i % len(self.colors)]),
                    hovertemplate=f'<b>{metric.upper()}</b><br>' +
                                 'Percentage: %{x}<br>' +
                                 'Avg Distance: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Distance vs Jitter Distance (averaged over percentages)
        distance_avg = self.combined_df.groupby(['metric', 'distance'])['metric_distance'].mean().reset_index()
        
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = distance_avg[distance_avg['metric'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['distance'],
                    y=metric_data['metric_distance'],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(color=self.colors[i % len(self.colors)]),
                    showlegend=False,
                    hovertemplate=f'<b>{metric.upper()}</b><br>' +
                                 'Jitter Distance: %{x:.3f}<br>' +
                                 'Avg Distance: %{y:.6f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Normalized comparison
        normalized_df = self.combined_df.copy()
        
        for metric in self.results_by_metric.keys():
            metric_mask = normalized_df['metric'] == metric
            metric_values = normalized_df.loc[metric_mask, 'metric_distance']
            min_val, max_val = metric_values.min(), metric_values.max()
            if max_val > min_val:
                normalized_df.loc[metric_mask, 'metric_distance'] = (metric_values - min_val) / (max_val - min_val)
        
        percentage_norm = normalized_df.groupby(['metric', 'percentage'])['metric_distance'].mean().reset_index()
        
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = percentage_norm[percentage_norm['metric'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['percentage'],
                    y=metric_data['metric_distance'],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(color=self.colors[i % len(self.colors)]),
                    showlegend=False,
                    hovertemplate=f'<b>{metric.upper()}</b><br>' +
                                 'Percentage: %{x}<br>' +
                                 'Normalized Distance: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Box plots
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            fig.add_trace(
                go.Box(
                    y=metric_data,
                    name=metric.upper(),
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Comparative Analysis of Metric Responses",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Average Distance", row=1, col=1)
        fig.update_xaxes(title_text="Jitter Distance", row=1, col=2)
        fig.update_yaxes(title_text="Average Distance", row=1, col=2)
        fig.update_xaxes(title_text="Percentage", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Distance", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Distance Value", row=2, col=2)
        
        fig.write_html(os.path.join(self.plot_dir, 'comparative_analysis.html'))
    
    def plot_metric_correlations(self):
        """Plot correlation analysis between metrics."""
        print("Creating correlation analysis...")
        
        # Create wide format for correlation
        wide_df = self.combined_df.pivot_table(
            values='metric_distance',  # Updated column name
            index=['percentage', 'distance'],
            columns='metric',
            aggfunc='mean'
        ).reset_index()
        
        metrics = list(self.results_by_metric.keys())
        correlation_matrix = wide_df[metrics].corr()
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Correlation Matrix', 'Pairwise Scatter Plot'],
            specs=[[{}, {}]]
        )
        
        # Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 12},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Scatter plot for first two metrics (if available)
        if len(metrics) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=wide_df[metrics[0]],
                    y=wide_df[metrics[1]],
                    mode='markers',
                    name=f'{metrics[0]} vs {metrics[1]}',
                    marker=dict(size=8, opacity=0.6),
                    hovertemplate=f'{metrics[0].upper()}: %{{x:.6f}}<br>' +
                                 f'{metrics[1].upper()}: %{{y:.6f}}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add correlation coefficient annotation
            corr_coef = correlation_matrix.iloc[0, 1]
            fig.add_annotation(
                x=0.95, y=0.95,
                xref="x2 domain", yref="y2 domain",
                text=f"r = {corr_coef:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        fig.update_layout(
            title_text="Metric Correlation Analysis",
            height=500
        )
        
        # Update axes
        fig.update_xaxes(title_text=f'{metrics[0].upper()}', row=1, col=2)
        fig.update_yaxes(title_text=f'{metrics[1].upper()}', row=1, col=2)
        
        fig.write_html(os.path.join(self.plot_dir, 'correlations.html'))
    
    def plot_distributions(self):
        """Plot statistical distributions of metric values."""
        print("Creating distribution plots...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Histogram Comparison',
                'Violin Plot Distribution',
                'Density Distribution', 
                'Log-scale Distribution'
            ]
        )
        
        # Histogram comparison
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            fig.add_trace(
                go.Histogram(
                    x=metric_data,
                    name=metric.upper(),
                    opacity=0.7,
                    nbinsx=20,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=1, col=1
            )
        
        # Violin plots
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            fig.add_trace(
                go.Violin(
                    y=metric_data,
                    name=metric.upper(),
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.colors[i % len(self.colors)],
                    opacity=0.6,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Density plots (using histogram with histnorm='probability density')
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            fig.add_trace(
                go.Histogram(
                    x=metric_data,
                    name=metric.upper(),
                    opacity=0.7,
                    nbinsx=20,
                    histnorm='probability density',
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Log scale comparison
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            # Filter out zero values for log scale
            log_data = np.log10(metric_data[metric_data > 0])
            if len(log_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=log_data,
                        name=metric.upper(),
                        opacity=0.7,
                        nbinsx=20,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Statistical Distribution Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Distance Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Distance Value", row=1, col=2)
        fig.update_xaxes(title_text="Distance Value", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_xaxes(title_text="log10(Distance Value)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        fig.write_html(os.path.join(self.plot_dir, 'distributions.html'))
    
    def plot_interactive_dashboard(self):
        """Create a comprehensive interactive dashboard."""
        print("Creating interactive dashboard...")
        
        # Create the main dashboard figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Metric Trends vs Percentage',
                'Parameter Response Heatmap',
                'Distribution Comparison',
                'Correlation Matrix',
                'Normalized Comparison',
                'Summary Statistics'
            ],
            specs=[
                [{}, {}],
                [{}, {}], 
                [{}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Metric trends
        percentage_avg = self.combined_df.groupby(['metric', 'percentage'])['metric_distance'].mean().reset_index()
        
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = percentage_avg[percentage_avg['metric'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['percentage'],
                    y=metric_data['metric_distance'],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(color=self.colors[i % len(self.colors)])
                ),
                row=1, col=1
            )
        
        # 2. Combined heatmap (using first metric as example)
        first_metric = list(self.results_by_metric.keys())[0]
        metric_data = self.combined_df[self.combined_df['metric'] == first_metric]
        pivot_table = metric_data.pivot_table(
            values='metric_distance',  # Updated column name
            index='percentage', 
            columns='distance',
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=[f'{d:.3f}' for d in pivot_table.columns],
                y=[f'{p:.1f}' for p in pivot_table.index],
                colorscale='viridis',
                showscale=False
            ),
            row=1, col=2
        )
        
        # 3. Box plots
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
            fig.add_trace(
                go.Box(
                    y=metric_data,
                    name=metric.upper(),
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Correlation matrix
        wide_df = self.combined_df.pivot_table(
            values='metric_distance',  # Updated column name
            index=['percentage', 'distance'],
            columns='metric',
            aggfunc='mean'
        ).reset_index()
        
        metrics = list(self.results_by_metric.keys())
        correlation_matrix = wide_df[metrics].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ),
            row=2, col=2
        )
        
        # 5. Normalized comparison
        normalized_df = self.combined_df.copy()
        
        for metric in self.results_by_metric.keys():
            metric_mask = normalized_df['metric'] == metric
            metric_values = normalized_df.loc[metric_mask, 'metric_distance']
            min_val, max_val = metric_values.min(), metric_values.max()
            if max_val > min_val:
                normalized_df.loc[metric_mask, 'metric_distance'] = (metric_values - min_val) / (max_val - min_val)
        
        percentage_norm = normalized_df.groupby(['metric', 'percentage'])['metric_distance'].mean().reset_index()
        
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = percentage_norm[percentage_norm['metric'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['percentage'],
                    y=metric_data['metric_distance'],
                    mode='lines+markers',
                    name=metric.upper(),
                    line=dict(color=self.colors[i % len(self.colors)]),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 6. Summary statistics table
        summary_stats = self.combined_df.groupby('metric')['metric_distance'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric'] + list(summary_stats.columns),
                           fill_color='lightblue'),
                cells=dict(values=[summary_stats.index] + [summary_stats[col] for col in summary_stats.columns],
                          fill_color='white')
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text="Multi-Metric Experiment Dashboard",
            height=1200,
            showlegend=True
        )
        
        fig.write_html(os.path.join(self.plot_dir, 'dashboard.html'))
    
    def plot_combined_3d(self):
        """Create a combined 3D visualization showing all metrics together."""
        print("Creating combined 3D visualization...")
        
        # Get parameter ranges
        percentages = sorted(self.combined_df['percentage'].unique())
        distances = sorted(self.combined_df['distance'].unique())
        
        fig = go.Figure()
        
        # Add each metric as a separate surface with offset
        for i, metric in enumerate(self.results_by_metric.keys()):
            metric_data = self.combined_df[self.combined_df['metric'] == metric]
            
            # Create pivot table
            pivot_table = metric_data.pivot_table(
                values='metric_distance',  # Updated column name
                index='percentage', 
                columns='distance',
                aggfunc='mean'
            )
            
            # Add surface with Z-offset to separate metrics visually
            z_offset = i * 0.1  # Small offset to separate surfaces
            
            fig.add_trace(
                go.Surface(
                    z=pivot_table.values + z_offset,
                    x=distances,      # X corresponds to columns (distance values)
                    y=percentages,    # Y corresponds to rows (percentage values)
                    name=metric.upper(),
                    colorscale='viridis',  # Use named colorscale instead of color string
                    opacity=0.8,
                    showscale=False
                )
            )
        
        fig.update_layout(
            title="Combined 3D Visualization - All Metrics",
            scene=dict(
                xaxis_title="Jitter Distance",  # X now corresponds to jitter distance
                yaxis_title="Percentage",       # Y now corresponds to percentage  
                zaxis_title="Metric Distance",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        fig.write_html(os.path.join(self.plot_dir, 'combined_3d.html'))
    
    def save_summary_report(self):
        """Save a text summary of the analysis."""
        report_path = os.path.join(self.plot_dir, 'analysis_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== Multi-Metric Experiment Analysis Summary ===\n\n")
            
            # Basic statistics
            f.write("SUMMARY STATISTICS:\n")
            summary_stats = self.combined_df.groupby('metric')['metric_distance'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            f.write(str(summary_stats))
            f.write("\n\n")
            
            # Correlation analysis
            f.write("CORRELATION ANALYSIS:\n")
            wide_df = self.combined_df.pivot_table(
                values='metric_distance',  # Updated column name
                index=['percentage', 'distance'],
                columns='metric',
                aggfunc='mean'
            ).reset_index()
            
            metrics = list(self.results_by_metric.keys())
            correlation_matrix = wide_df[metrics].corr()
            f.write(str(correlation_matrix))
            f.write("\n\n")
            
            # Key insights
            f.write("KEY INSIGHTS:\n")
            for metric in metrics:
                metric_data = self.combined_df[self.combined_df['metric'] == metric]['metric_distance']
                f.write(f"- {metric.upper()}: Range [{metric_data.min():.6f}, {metric_data.max():.6f}], "
                       f"Mean = {metric_data.mean():.6f}, Std = {metric_data.std():.6f}\n")
            
            f.write("\n=== VISUALIZATION FILES ===\n")
            f.write("- 3d_surfaces.html: Interactive 3D surface plots for each metric\n")
            f.write("- interactive_heatmaps.html: Interactive heatmaps showing parameter response\n")
            f.write("- comparative_analysis.html: Line plots and distributions comparing metrics\n")
            f.write("- correlations.html: Correlation analysis between metrics\n")
            f.write("- distributions.html: Statistical distribution analysis\n")
            f.write("- dashboard.html: Comprehensive interactive dashboard\n")
            f.write("- combined_3d.html: All metrics in single 3D visualization\n")
        
        print(f"Analysis summary saved to: {report_path}")


def create_comprehensive_plots(results_by_metric: Dict[str, List[Dict]], output_dir: str):
    """
    Create comprehensive interactive visualizations for multi-metric results using Plotly.
    
    Args:
        results_by_metric: Dictionary mapping metric names to result lists
        output_dir: Directory where plots will be saved
    """
    try:
        plotter = MultiMetricPlotter(results_by_metric, output_dir)
        plotter.create_all_plots()
        plotter.save_summary_report()
        
        return plotter.plot_dir
    except ImportError:
        print("⚠️  Plotly not available. Install with: pip install plotly")
        return None
    except Exception as e:
        print(f"⚠️  Error creating plots: {e}")
        return None
