import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import torch
from datetime import datetime
import json
from pathlib import Path

# Absolute imports
from trading_bot.trainer import TradingTrainer
from trading_bot.model_builder import ModelBuilder

class TrainingVisualizer:
    """Comprehensive training metrics visualization"""
    
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List], save_plot: bool = True) -> go.Figure:
        """Plot comprehensive training history"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Learning Rate Schedule', 'Training Speed', 'Validation Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss curves
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Learning rate schedule
        if 'learning_rates' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['learning_rates'],
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
        
        # Training speed (epoch times)
        if 'epoch_times' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['epoch_times'],
                    mode='lines+markers',
                    name='Epoch Time (s)',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # Validation metrics (if available)
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_accuracy'],
                    mode='lines',
                    name='Validation Accuracy',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Training History Overview",
            height=600,
            showlegend=True
        )
        
        if save_plot:
            fig.write_html(self.save_dir / "training_history.html")
        
        return fig
    
    def plot_loss_landscape(self, model: torch.nn.Module, data_loader, save_plot: bool = True) -> go.Figure:
        """Visualize loss landscape around current parameters"""
        # This is a simplified 2D projection of the loss landscape
        model.eval()
        
        # Get current parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Create parameter perturbation grid
        n_points = 20
        alpha_range = np.linspace(-0.5, 0.5, n_points)
        beta_range = np.linspace(-0.5, 0.5, n_points)
        
        # Random directions for perturbation
        direction1 = [torch.randn_like(p) for p in original_params]
        direction2 = [torch.randn_like(p) for p in original_params]
        
        # Normalize directions
        for i, (d1, d2) in enumerate(zip(direction1, direction2)):
            direction1[i] = d1 / torch.norm(d1)
            direction2[i] = d2 / torch.norm(d2)
        
        loss_surface = np.zeros((n_points, n_points))
        
        criterion = torch.nn.MSELoss()
        
        for i, alpha in enumerate(alpha_range):
            for j, beta in enumerate(beta_range):
                # Perturb parameters
                with torch.no_grad():
                    for k, (param, d1, d2, orig) in enumerate(zip(model.parameters(), direction1, direction2, original_params)):
                        param.data = orig + alpha * d1 + beta * d2
                
                # Calculate loss
                total_loss = 0
                num_batches = 0
                with torch.no_grad():
                    for batch in data_loader:
                        if num_batches >= 5:  # Limit to 5 batches for speed
                            break
                        
                        price_data = batch['price_data']
                        targets = batch['target']
                        
                        predictions = model(price_data)
                        loss = criterion(predictions.squeeze(), targets)
                        total_loss += loss.item()
                        num_batches += 1
                
                loss_surface[i, j] = total_loss / max(num_batches, 1)
        
        # Restore original parameters
        with torch.no_grad():
            for param, orig in zip(model.parameters(), original_params):
                param.data = orig
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                z=loss_surface,
                x=alpha_range,
                y=beta_range,
                colorscale='Viridis'
            )
        ])
        
        fig.update_layout(
            title="Loss Landscape (2D Projection)",
            scene=dict(
                xaxis_title="Parameter Direction 1",
                yaxis_title="Parameter Direction 2",
                zaxis_title="Loss Value"
            ),
            height=600
        )
        
        if save_plot:
            fig.write_html(self.save_dir / "loss_landscape.html")
        
        return fig
    
    def plot_gradient_flow(self, model: torch.nn.Module, save_plot: bool = True) -> go.Figure:
        """Visualize gradient flow through the network"""
        layers = []
        avg_grads = []
        max_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layers.append(name)
                avg_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        fig = go.Figure()
        
        # Average gradients
        fig.add_trace(
            go.Bar(
                x=layers,
                y=avg_grads,
                name='Average Gradient Magnitude',
                marker_color='blue',
                opacity=0.7
            )
        )
        
        # Maximum gradients
        fig.add_trace(
            go.Bar(
                x=layers,
                y=max_grads,
                name='Maximum Gradient Magnitude',
                marker_color='red',
                opacity=0.7
            )
        )
        
        fig.update_layout(
            title="Gradient Flow Analysis",
            xaxis_title="Layer",
            yaxis_title="Gradient Magnitude",
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        
        if save_plot:
            fig.write_html(self.save_dir / "gradient_flow.html")
        
        return fig
    
    def plot_weight_distribution(self, model: torch.nn.Module, save_plot: bool = True) -> go.Figure:
        """Plot weight distribution across layers"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weight Histograms', 'Weight Norms by Layer', 'Bias Distribution', 'Parameter Statistics'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # Collect all weights and biases
        all_weights = []
        all_biases = []
        layer_names = []
        weight_norms = []
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.flatten().cpu().numpy()
                all_weights.extend(weights)
                layer_names.append(name)
                weight_norms.append(np.linalg.norm(weights))
            elif 'bias' in name:
                biases = param.data.flatten().cpu().numpy()
                all_biases.extend(biases)
        
        # Weight histogram
        fig.add_trace(
            go.Histogram(
                x=all_weights,
                nbinsx=50,
                name='Weight Distribution',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Weight norms by layer
        fig.add_trace(
            go.Bar(
                x=layer_names,
                y=weight_norms,
                name='Weight Norms',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Bias histogram
        if all_biases:
            fig.add_trace(
                go.Histogram(
                    x=all_biases,
                    nbinsx=30,
                    name='Bias Distribution',
                    opacity=0.7,
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # Parameter statistics table
        stats_data = {
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Median'],
            'Weights': [
                f"{np.mean(all_weights):.6f}",
                f"{np.std(all_weights):.6f}",
                f"{np.min(all_weights):.6f}",
                f"{np.max(all_weights):.6f}",
                f"{np.median(all_weights):.6f}"
            ],
            'Biases': [
                f"{np.mean(all_biases):.6f}" if all_biases else "N/A",
                f"{np.std(all_biases):.6f}" if all_biases else "N/A",
                f"{np.min(all_biases):.6f}" if all_biases else "N/A",
                f"{np.max(all_biases):.6f}" if all_biases else "N/A",
                f"{np.median(all_biases):.6f}" if all_biases else "N/A"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys())),
                cells=dict(values=list(stats_data.values()))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Weight Analysis",
            height=700,
            showlegend=True
        )
        
        if save_plot:
            fig.write_html(self.save_dir / "weight_distribution.html")
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, 
                              save_plot: bool = True) -> go.Figure:
        """Plot feature importance analysis"""
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        fig = go.Figure()
        
        # Horizontal bar chart for better readability
        fig.add_trace(
            go.Bar(
                x=sorted_scores[:20],  # Top 20 features
                y=sorted_features[:20],
                orientation='h',
                marker_color=px.colors.sequential.Viridis,
                text=sorted_scores[:20],
                texttemplate='%{text:.3f}',
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title="Feature Importance Analysis (Top 20)",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            margin=dict(l=150)
        )
        
        if save_plot:
            fig.write_html(self.save_dir / "feature_importance.html")
        
        return fig
    
    def create_training_report(self, history: Dict[str, List], model: torch.nn.Module, 
                             feature_names: List[str], feature_importance: np.ndarray,
                             save_report: bool = True) -> str:
        """Generate comprehensive training report"""
        report = []
        report.append("# Advanced Trading Bot - Training Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Training Summary")
        
        # Training metrics
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
        best_val_loss = min(history['val_loss']) if history['val_loss'] else 0
        
        report.append(f"- **Final Training Loss**: {final_train_loss:.6f}")
        report.append(f"- **Final Validation Loss**: {final_val_loss:.6f}")
        report.append(f"- **Best Validation Loss**: {best_val_loss:.6f}")
        report.append(f"- **Total Epochs**: {len(history['train_loss'])}")
        
        if 'epoch_times' in history:
            avg_epoch_time = np.mean(history['epoch_times'])
            total_training_time = sum(history['epoch_times'])
            report.append(f"- **Average Epoch Time**: {avg_epoch_time:.2f} seconds")
            report.append(f"- **Total Training Time**: {total_training_time/60:.1f} minutes")
        
        # Model statistics
        report.append("\n## Model Statistics")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        report.append(f"- **Total Parameters**: {total_params:,}")
        report.append(f"- **Trainable Parameters**: {trainable_params:,}")
        report.append(f"- **Model Size (MB)**: {total_params * 4 / (1024**2):.2f}")
        
        # Top features
        report.append("\n## Top 10 Most Important Features")
        sorted_indices = np.argsort(feature_importance)[::-1]
        for i in range(min(10, len(feature_names))):
            idx = sorted_indices[i]
            report.append(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        # Training recommendations
        report.append("\n## Training Analysis & Recommendations")
        
        # Overfitting check
        if len(history['train_loss']) > 10:
            recent_train_trend = np.polyfit(range(-10, 0), history['train_loss'][-10:], 1)[0]
            recent_val_trend = np.polyfit(range(-10, 0), history['val_loss'][-10:], 1)[0]
            
            if recent_val_trend > 0 and recent_train_trend < 0:
                report.append("⚠️ **Potential Overfitting Detected**: Validation loss increasing while training loss decreasing")
                report.append("   - Consider reducing model complexity or increasing regularization")
                report.append("   - Add more dropout or weight decay")
            elif abs(recent_train_trend) < 0.001 and abs(recent_val_trend) < 0.001:
                report.append("✅ **Training Converged**: Both training and validation losses stabilized")
            else:
                report.append("📈 **Training in Progress**: Model still learning")
        
        # Learning rate analysis
        if 'learning_rates' in history and len(history['learning_rates']) > 1:
            lr_reduction = history['learning_rates'][0] / history['learning_rates'][-1]
            if lr_reduction > 10:
                report.append("📉 **Significant LR Decay**: Learning rate reduced significantly during training")
                report.append("   - This suggests the scheduler is working effectively")
        
        report_text = "\n".join(report)
        
        if save_report:
            with open(self.save_dir / "training_report.md", 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_training_artifacts(self, history: Dict, model: torch.nn.Module, 
                              model_builder: ModelBuilder, symbol: str):
        """Save all training artifacts for future reference"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        with open(self.save_dir / f"history_{symbol}_{timestamp}.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, (list, np.ndarray)):
                    json_history[key] = [float(x) for x in value]
                else:
                    json_history[key] = value
            json.dump(json_history, f, indent=2)
        
        # Save model configuration
        model_builder.save_config(self.save_dir / f"model_config_{symbol}_{timestamp}.json")
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'symbol': symbol
        }, self.save_dir / f"model_{symbol}_{timestamp}.pth")

def visualize_training_session(history: Dict, model: torch.nn.Module, 
                             model_builder: ModelBuilder, feature_names: List[str],
                             feature_importance: np.ndarray, symbol: str = "UNKNOWN"):
    """Complete training visualization workflow"""
    visualizer = TrainingVisualizer()
    
    # Generate all plots
    print("Generating training visualizations...")
    
    visualizer.plot_training_history(history)
    visualizer.plot_weight_distribution(model)
    visualizer.plot_feature_importance(feature_names, feature_importance)
    
    # Generate comprehensive report
    report = visualizer.create_training_report(
        history, model, feature_names, feature_importance
    )
    
    # Save artifacts
    visualizer.save_training_artifacts(history, model, model_builder, symbol)
    
    print(f"Training visualization complete. Files saved to: {visualizer.save_dir}")
    
    return report