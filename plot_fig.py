from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os

from utils import load_test_dataset, analyze_intra_condition_similarity
from test_acc_topo import process_morans_i_results, process_ed_results


def process_baseline_results(file_path):
    acc_data = np.load(file_path)
    control_acc = acc_data[: 10]  # Control condition accuracies
    as_acc = acc_data[10: 70].reshape(6, 10)  # Activation smoothing accuracies
    ws_acc = acc_data[70:].reshape(6, 10)  # Weight smoothing accuracies
    lambdas = [0.1, 0.3, 0.5, 1, 2, 3]

    # Calculate mean and SD for control models
    control_stats = {
        "mean": np.mean(control_acc),
        "std": np.std(control_acc)
    }   
    # Calculate mean and SD for activation and weight smoothing models by lambda
    activation_smooth_stats = {
        lambdas[i]: {
            "mean": np.mean(as_acc[i]),
            "std": np.std(as_acc[i])
        }
        for i in range(6)
    }
    weight_smooth_stats = {
        lambdas[i]: {
            "mean": np.mean(ws_acc[i]),
            "std": np.std(ws_acc[i])
        }
        for i in range(6)
    }

    # Combine everything into a single dictionary
    baseline_summary = {
        "control": control_stats,
        "as": activation_smooth_stats,
        "ws": weight_smooth_stats
    }
    return baseline_summary


def process_noise_results(file_path, baseline_summary):
    """
    Process the noise results and store them in a pandas DataFrame.

    Args:
        file_path (str): Path to the .npy file containing noise results.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """
    # Step 1: Read the file and reshape to 5 x 13 x 10
    data = np.load(file_path)
    reshaped_data = data.reshape(5, 13, 10)  # 5 noise levels, 13 conditions, 10 trials

    # Step 2: Create a DataFrame
    noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
    lambdas = [0.1, 0.3, 0.5, 1, 2, 3]
    conditions = ['control'] + [f'as_{i}' for i in range(1, 7)] + [f'ws_{i}' for i in range(1, 7)]
    trials = list(range(1, 11))

    # Flatten the data for DataFrame creation
    rows = []
    for noise_idx, noise_level in enumerate(noise_levels):
        for condition_idx, condition in enumerate(conditions):
            for trial_idx, trial in enumerate(trials):
                # Determine lambda value for 'as' and 'ws' conditions
                if condition.startswith('as_'):
                    lambda_value = lambdas[int(condition.split('_')[1]) - 1]
                elif condition.startswith('ws_'):
                    lambda_value = lambdas[int(condition.split('_')[1]) - 1]
                else:
                    lambda_value = None  # No lambda for 'control'

                rows.append({
                    'Noise Level': noise_level,
                    'Condition': condition.split('_')[0],  # Replace underscores with spaces for better readability
                    'Lambda': lambda_value,
                    'Trial': trial,
                    'Accuracy': reshaped_data[noise_idx, condition_idx, trial_idx],
                })

    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    # Filter activation_smooth and weight_smooth
    sem_df = df[df['Condition'].isin(['as', 'ws'])]
    # Compute mean and SEM for activation_smooth and weight_smooth
    sem_summary = sem_df.groupby(['Condition', 'Lambda', 'Noise Level']).agg(
        accuracy_mean=('Accuracy', 'mean'),
        accuracy_sem=('Accuracy', 'sem')  # Compute SEM
    ).reset_index()
    # Compute mean and SEM for control condition
    control_summary = df[df['Condition'] == 'control'].groupby('Noise Level').agg(
        accuracy_mean=('Accuracy', 'mean'),
        accuracy_sem=('Accuracy', 'sem')  # Compute SEM
    ).reset_index()

    # Asjusted accuracy
    def _adjust_accuracy(row):
        # Extract model type and lambda value
        model_type = row['Condition']
        lambda_val = row['Lambda']    
        # Get the baseline mean
        baseline_mean = baseline_summary[model_type][lambda_val]["mean"]     
        # Subtract the baseline mean from accuracy_mean
        return row['accuracy_mean'] - baseline_mean
    sem_summary['adjusted_accuracy_mean'] = sem_summary.apply(_adjust_accuracy, axis=1)

    return sem_summary, control_summary


def plot_noise_results(sem_summary, control_summary, baseline_summary, save_path=None):
    noise_levels = sem_summary['Noise Level'].unique()
    lambda_values = sem_summary['Lambda'].unique()
    # Define subplot dimensions
    subplot_size = 0.2 * 8.27  # Each subplot is a square 20% of A4 width

    # Set up the figure and axes with adjusted size
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(subplot_size * len(noise_levels), subplot_size), sharey=False)

    # Loop through each noise level to create a subplot
    for i, noise_level in enumerate(noise_levels):
        plot_data = sem_summary[sem_summary['Noise Level'] == noise_level].copy()  # Make a copy to avoid warning
        plot_data['Lambda'] = plot_data['Lambda'].astype(str)
        lambda_values_str = [f'{x:.1f}' for x in lambda_values]


        # Multiply Y values by 100 (for percentage presentation)
        plot_data['adjusted_accuracy_mean'] *= 100
        plot_data['accuracy_sem'] *= 100
        
        # Plotting activation_smooth and weight_smooth with lambda as categorical
        sns.lineplot(
            data=plot_data,
            x='Lambda', 
            y='adjusted_accuracy_mean', 
            hue='Condition', 
            marker='o', 
            ax=axes[i], 
            palette={"as": "blue", "ws": "orange"},
            legend=False,
            errorbar=None,
            markersize=4,
            linewidth=0.7
        )

        # Add SEM shading for activation_smooth and weight_smooth
        for model_type in plot_data['Condition'].unique():
            model_data = plot_data[plot_data['Condition'] == model_type]
            axes[i].fill_between(
                model_data['Lambda'],
                model_data['adjusted_accuracy_mean'] - model_data['accuracy_sem'],  # Lower bound
                model_data['adjusted_accuracy_mean'] + model_data['accuracy_sem'],  # Upper bound
                alpha=0.2
            )

        # Plot control condition as a straight line
        control_data = control_summary[control_summary['Noise Level'] == noise_level]
        if not control_data.empty:  # Check if there is control data for this noise level
            control_mean = control_data['accuracy_mean'].values[0]  # Get the control mean for the noise level
            
            # Adjust the control mean if necessary (subtract baseline if required)
            if 'control' in baseline_summary:
                baseline_mean = baseline_summary['control'].get('mean', 0)
                adjusted_control_mean = control_mean - baseline_mean
            else:
                adjusted_control_mean = control_mean  # No adjustment if no baseline
            
            adjusted_control_mean *= 100

            # Plot control condition as a horizontal line
            axes[i].axhline(y=adjusted_control_mean, color='red', linestyle='--', linewidth=0.7)

        # Set the title for each subplot
        if i == 0:
            axes[i].set_title(f'Noise Level: {noise_level}', fontsize=9)  # Title in font size 9
        else:
            axes[i].set_title(f'{noise_level}', fontsize=9)
        axes[i].tick_params(axis='both', which='major', labelsize=6)  # Tick labels in font size 6
        # Set x-ticks to show all lambda values evenly spaced
        axes[i].set_xticks(range(len(lambda_values)))
        axes[i].set_xticklabels(lambda_values_str, fontsize=6)

        # Add labels and legend only for the leftmost plot
        if i == 0:
            axes[i].set_xlabel('$\lambda$', fontsize=8)  # X label in font size 9
            axes[i].set_ylabel('Change (%)', fontsize=8)  # Y label in font size 9
            # axes[i].legend(loc='lower left', fontsize=6)  # Add legend only for the first subplot
        else:
            axes[i].set_xlabel('')  # Remove X label for other plots
            axes[i].set_ylabel('')  # Remove Y label for other plots

    # Tight layout for better spacing
    plt.tight_layout()  
    # Save and display the plots
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")    
    plt.show()


def plot_noise_examples(dataset_name, n_samples=5, save_path=None):
    """
    Plot examples of different noise types on randomly selected images.
    Original images on top, followed by noisy versions.
    """
    # Load dataset without normalization
    data_loader = load_test_dataset(dataset_name, use_norm=False)
    
    # Get random indices
    all_images = next(iter(data_loader))[0]
    random_indices = torch.randperm(len(all_images))[:n_samples]
    selected_images = all_images[random_indices]
    
    # Set up noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
    noise_types = ['white', 'pink', 'salt_pepper']
    
    # Create figure with 4 rows (original + 3 noise types)
    fig, axes = plt.subplots(4, n_samples, figsize=(2*n_samples, 8))
    
    # Plot original images in the first row
    for c in range(n_samples):
        img = selected_images[c]
        if img.shape[0] == 1:  # MNIST
            axes[0, c].imshow(img.squeeze().numpy(), cmap='gray')
        else:  # CIFAR
            axes[0, c].imshow(img.permute(1, 2, 0).numpy())
        axes[0, c].set_xticks([])
        axes[0, c].set_yticks([])
    
    # Loop through noise types and samples
    for r, noise_type in enumerate(noise_types):
        for c in range(n_samples):
            img = selected_images[c]
            noise_level = noise_levels[c]
            n_channels = img.shape[0]
            
            # Apply noise
            if noise_type == 'white':
                noisy_img = img + noise_level * torch.randn_like(img)
                noisy_img = torch.clamp(noisy_img, 0, 1)
            elif noise_type == 'pink':
                noisy_img = torch.zeros_like(img)
                for channel in range(n_channels):
                    f_white = np.fft.fft2(torch.randn_like(img[channel]).numpy())
                    f_x = np.fft.fftfreq(img.shape[-1])
                    f_y = np.fft.fftfreq(img.shape[-2])
                    f_XX, f_YY = np.meshgrid(f_x, f_y)
                    f_dist = np.sqrt(f_XX**2 + f_YY**2)
                    f_dist[0, 0] = 1.0
                    pink_f = f_white / f_dist
                    pink = np.real(np.fft.ifft2(pink_f))
                    pink = (pink - pink.mean()) / pink.std()
                    noisy_img[channel] = img[channel] + noise_level * torch.tensor(pink).float()

                    # from scipy.signal import lfilter
                    #  # Generate white noise
                    # white_noise = torch.randn_like(img[channel]).numpy()
                    # # Apply a low-pass filter to create pink noise
                    # # Filter coefficients
                    # b = [1]  # Numerator coefficients (feedforward)
                    # a = [1, -0.9]  # Denominator coefficients (feedback)
                    # # Use lfilter to apply the filter to the white noise
                    # pink_noise = lfilter(b, a, white_noise)
                    # # Normalize the pink noise to have the same standard deviation as specified
                    # pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
                    # noisy_img[channel] = img[channel] + noise_level * torch.tensor(pink_noise).float()

                noisy_img = torch.clamp(noisy_img, 0, 1)
            else:  # salt_pepper
                noisy_img = img.clone()
                n_pixels = int(img.shape[-1] * img.shape[-2] * noise_level)
                for _ in range(n_pixels):
                    pixel_row = torch.randint(0, img.shape[-2], (1,))
                    pixel_col = torch.randint(0, img.shape[-1], (1,))
                    if torch.rand(1) < 0.5:
                        noisy_img[:, pixel_row, pixel_col] = 0.0
                    else:
                        noisy_img[:, pixel_row, pixel_col] = 1.0
            
            # Display image
            if n_channels == 1:
                axes[r+1, c].imshow(noisy_img.squeeze().numpy(), cmap='gray')
            else:
                axes[r+1, c].imshow(noisy_img.permute(1, 2, 0).numpy())
            axes[r+1, c].set_xticks([])
            axes[r+1, c].set_yticks([])
            
    # Add noise level labels at the bottom
    for c in range(n_samples):
        noise_level = noise_levels[c]
        axes[-1, c].set_xlabel(f'{noise_level}', fontsize=12)
    axes[-1, 0].set_xlabel('Noise Level 0.01', fontsize=12)
    
    # Define noise type labels
    noise_labels = {
        'original': 'No noise',
        'white': 'White noise',
        'pink': 'Pink noise',
        'salt_pepper': 'Salt & Pepper'
    }
    
    # Add y-axis labels with proper positioning
    for r, noise_type in enumerate(['original'] + noise_types):
        axes[r, 0].set_ylabel(noise_labels[noise_type], fontsize=14)

    # Adjust layout
    plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.05)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_noise_comparison(dataset_name, baseline_summary, save_path=None):
    """
    Plot noise comparison results in a 3x5 grid (3 noise types, 5 noise levels).
    Each subplot shows lambda vs accuracy change for a specific noise type and level.
    """
    noise_types = ['white', 'pink', 'salt_pepper']
    noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    # Create a 3x5 grid of subplots with increased width and adjusted height
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))  # Increased height to 10
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.85, wspace=0.3, hspace=0.3)  # Adjusted top margin

    # Add dataset name at the top center with increased spacing and size
    # dataset_display = "CIFAR-10" if dataset_name == "cifar" else dataset_name.upper()
    dataset_display = dataset_name.upper()
    fig.suptitle(dataset_display, fontsize=24, y=0.95)  # Increased font size and y position
    
    # Process and plot each noise type
    for row, noise_type in enumerate(noise_types):
        # Load and process noise results
        noise_results_path = f'./results/acc/{dataset_name}_noise_{noise_type}.npy'
        sem_summary, control_summary = process_noise_results(noise_results_path, baseline_summary)
        
        # Use default matplotlib colors
        colors = {
            'as': '#1f77b4',    # Default matplotlib blue
            'ws': '#ff7f0e'     # Default matplotlib orange
        }
        
        # Plot each noise level
        for col, noise_level in enumerate(noise_levels):
            ax = axes[row, col]
            
            # Get data for this noise level
            plot_data = sem_summary[sem_summary['Noise Level'] == noise_level].copy()
            # Convert Lambda to string for categorical x-axis
            plot_data['Lambda'] = plot_data['Lambda'].astype(str)
            plot_data['adjusted_accuracy_mean'] *= 100
            plot_data['accuracy_sem'] *= 100
            
            # Plot activation and weight smoothing lines
            sns.lineplot(
                data=plot_data,
                x='Lambda',
                y='adjusted_accuracy_mean',
                hue='Condition',
                marker='o',
                ax=ax,
                palette=colors,
                legend=False if (row > 0 or col > 0) else True,
                errorbar=None,
                markersize=4,
                linewidth=2
            )
            
            # Add SEM shading with matching colors
            for condition in plot_data['Condition'].unique():
                condition_data = plot_data[plot_data['Condition'] == condition]
                ax.fill_between(
                    range(len(condition_data)),
                    condition_data['adjusted_accuracy_mean'] - condition_data['accuracy_sem'],
                    condition_data['adjusted_accuracy_mean'] + condition_data['accuracy_sem'],
                    alpha=0.2,
                    color=colors[condition]
                )
            
            # Plot control line in darker red
            control_data = control_summary[control_summary['Noise Level'] == noise_level]
            if not control_data.empty:
                control_mean = control_data['accuracy_mean'].values[0]
                baseline_mean = baseline_summary['control']['mean']
                adjusted_control_mean = (control_mean - baseline_mean) * 100
                ax.axhline(y=adjusted_control_mean, color='red', linestyle='--', linewidth=1.5)
            
            # Customize subplot with smaller tick labels
            lambda_values = ['0.1', '0.3', '0.5', '1.0', '2.0', '3.0']
            ax.set_xticks(range(len(lambda_values)))
            ax.set_xticklabels(lambda_values, fontsize=10)  # Decreased from 14
            ax.tick_params(axis='both', which='major', labelsize=10)  # Decreased from 14
            
            # Add labels only where needed
            if col == 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('')
            
            if row == 2:
                ax.set_xlabel('λ', fontsize=18)  # Keep lambda symbol large
            else:
                ax.set_xlabel('')
                
            # Add noise level only to first row
            if row == 0:
                ax.set_title(f'Noise Level: {noise_level}', fontsize=18)  # Changed format and kept size
    
    # Modify y-axis labels with larger font size
    noise_type_labels = {
        'white': 'WHITE NOISE\n\nChange (%)',  # Added extra newline for better spacing
        'pink': 'PINK NOISE\n\nChange (%)',
        'salt_pepper': 'SALT & PEPPER\n\nChange (%)'
    }
    
    # Update labels with larger font size
    for row, noise_type in enumerate(noise_types):
        for col in range(5):
            ax = axes[row, col]
            if col == 0:
                ax.set_ylabel(noise_type_labels[noise_type], fontsize=16, labelpad=25)  # Increased labelpad
    
    # Create custom lines with increased thickness for legend
    as_line = plt.Line2D([0], [0], color=colors['as'], linewidth=2.0, marker='o', markersize=12)
    ws_line = plt.Line2D([0], [0], color=colors['ws'], linewidth=2.0, marker='o', markersize=12)
    control_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5)
    
    # Add legend with better text alignment
    fig.legend([as_line, ws_line, control_line], ['AS', 'WS', 'Control'], 
              loc='center', 
              bbox_to_anchor=(0.5, 0.08),
              ncol=3,
              fontsize=20,  # Increased font size
              frameon=False,
              handlelength=3,
              handleheight=1,
              handletextpad=0.5,    # Adjusted padding between line and text
              columnspacing=1.5,    # Adjusted spacing between columns
              labelspacing=0.2,
              borderaxespad=0.0)
    
    # Remove the original legend from the first subplot
    if axes.size > 0:
        axes[0, 0].get_legend().remove()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_morans_i_results(sem_summary, control_summary, dataset_name, save_path=None):
    """
    Plot Moran's I results in a single plot with three lines (AS, WS, Control).
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15, left=0.15, top=0.85)  # Adjusted top margin
    
    # Add dataset name at the top
    # dataset_display = "CIFAR-10" if dataset_name == "cifar" else dataset_name.upper()
    dataset_display = dataset_name.upper()
    fig.suptitle(dataset_display, fontsize=20, y=0.95)
    
    # Use default matplotlib colors
    colors = {
        'as': '#1f77b4',    # Default matplotlib blue
        'ws': '#ff7f0e'     # Default matplotlib orange
    }
    
    # Convert Lambda to string for categorical x-axis
    sem_summary['Lambda'] = sem_summary['Lambda'].astype(str)
    
    # Plot AS and WS lines
    for condition in ['as', 'ws']:
        condition_data = sem_summary[sem_summary['Condition'] == condition]
        
        # Plot main line
        sns.lineplot(
            data=condition_data,
            x='Lambda',
            y='moran_i_mean',
            color=colors[condition],
            marker='o',
            markersize=8,
            linewidth=2,
            label=condition.upper(),
            ax=ax,
        )
        
        # Add SEM shading
        ax.fill_between(
            range(len(condition_data)),
            condition_data['moran_i_mean'] - condition_data['moran_i_sem'],
            condition_data['moran_i_mean'] + condition_data['moran_i_sem'],
            alpha=0.2,
            color=colors[condition]
        )
    
    # Add control line
    control_mean = control_summary['moran_i_mean'].iloc[0]
    control_sem = control_summary['moran_i_sem'].iloc[0]
    ax.axhline(y=control_mean, color='red', linestyle='--', linewidth=2, label='Control')
    ax.fill_between(
        ax.get_xlim(),
        [control_mean - control_sem] * 2,
        [control_mean + control_sem] * 2,
        color='red',
        alpha=0.2,
    )
    
    # Customize plot
    lambda_values = ['0.1', '0.3', '0.5', '1.0', '2.0', '3.0']
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels(lambda_values, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Labels and title
    ax.set_xlabel('λ', fontsize=18)
    ax.set_ylabel("Moran's I", fontsize=18)
    
    # Create custom lines with increased thickness for legend
    as_line = plt.Line2D([0], [0], color=colors['as'], linewidth=2.0, marker='o', markersize=8)
    ws_line = plt.Line2D([0], [0], color=colors['ws'], linewidth=2.0, marker='o', markersize=8)
    control_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2.0)
    
    # Add legend with better text alignment
    fig.legend([as_line, ws_line, control_line], ['AS', 'WS', 'Control'], 
              loc='center', 
              bbox_to_anchor=(0.5, 0.0),
              ncol=3,
              fontsize=18,  
              frameon=False,
              handlelength=3,
              handleheight=1,
              handletextpad=0.5,
              columnspacing=1.5,
              labelspacing=0.2,
              borderaxespad=0.0)
    
    # Remove the original legend from the plot
    ax.get_legend().remove()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ed_results(ed_weight_avg, ed_weight_sem, ed_act_avg, ed_act_sem,
                    dataset_name, save_path=None):
    """
    Plot Effective Dimensionality results for weight and activation smoothing.
    """
    lambda_labels = ['0.1', '0.3', '0.5', '1', '2', '3']
    x = np.arange(len(lambda_labels))  # categorical positions: 0..5

    # CTRL (constant across lambda)
    ctrl_weight_mean_val, ctrl_weight_sem_val = ed_weight_avg[0], ed_weight_sem[0]
    ctrl_act_mean_val, ctrl_act_sem_val = ed_act_avg[0], ed_act_sem[0]
    ctrl_weight_mean = np.full_like(x, ctrl_weight_mean_val, dtype=float)
    ctrl_weight_sem  = np.full_like(x, ctrl_weight_sem_val, dtype=float)
    ctrl_act_mean    = np.full_like(x, ctrl_act_mean_val, dtype=float)
    ctrl_act_sem     = np.full_like(x, ctrl_act_sem_val, dtype=float)

    # AS
    as_weight_mean, as_weight_sem = ed_weight_avg[1:7], ed_weight_sem[1:7]
    as_act_mean, as_act_sem  = ed_act_avg[1:7], ed_act_sem[1:7]

    # WS
    ws_weight_mean, ws_weight_sem = ed_weight_avg[7:], ed_weight_sem[7:]
    ws_act_mean, ws_act_sem  = ed_act_avg[7:], ed_act_sem[7:]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=130, sharex=True)

    color_ctrl = 'red'
    color_ws   = 'orange'
    color_as   = 'blue'

    def _plot_with_shade(ax, x, y, sem, color, label, linestyle='-'):
        ax.plot(x, y, linestyle=linestyle, color=color, label=label)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2)

    # CTRL
    # _plot_with_shade(ax, x, ctrl_weight_mean, ctrl_weight_sem, color_ctrl, 'Ctrl weight', '-')
    # _plot_with_shade(ax, x, ctrl_act_mean,   ctrl_act_sem,     color_ctrl, 'Ctrl act', '--')

    # # AS
    # _plot_with_shade(ax, x, as_weight_mean, as_weight_sem, color_as, 'AS weight', '-')
    # _plot_with_shade(ax, x, as_act_mean,   as_act_sem,     color_as, 'AS act', '--')

    # # WS
    # _plot_with_shade(ax, x, ws_weight_mean, ws_weight_sem, color_ws, 'WS weight', '-')
    # _plot_with_shade(ax, x, ws_act_mean,   ws_act_sem,     color_ws, 'WS act', '--')

    # # Axes, labels, etc.
    # ax.set_xticks(x)
    # ax.set_xticklabels(lambda_labels)
    # ax.set_xlabel(r'$\lambda$', fontsize=18)
    # ax.set_ylabel('Effective Dimensionality', fontsize=18)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_title(dataset_name.upper(), fontsize=22)
    # ax.grid(True, linestyle=':', linewidth=0.5)
    # # ax.legend(ncol=2, fontsize=16)

    # Weights
    _plot_with_shade(ax1, x, ctrl_weight_mean, ctrl_weight_sem, color_ctrl, 'Control',  '--')
    _plot_with_shade(ax1, x, as_weight_mean, as_weight_sem, color_as, 'AS')
    _plot_with_shade(ax1, x, ws_weight_mean, ws_weight_sem, color_ws, 'WS')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambda_labels)
    ax1.set_xlabel(r'$\lambda$', fontsize=16)
    ax1.set_ylabel('Effective Dimensionality', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_title('Weight matrix', fontsize=18)
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.legend(fontsize=16)

    # Activations
    _plot_with_shade(ax2, x, ctrl_act_mean, ctrl_act_sem, color_ctrl, 'Control', '--')
    _plot_with_shade(ax2, x, as_act_mean, as_act_sem, color_as, 'AS',)
    _plot_with_shade(ax2, x, ws_act_mean, ws_act_sem, color_ws, 'WS',)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lambda_labels)
    ax2.set_xlabel(r'$\lambda$', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_title('Activation matrix', fontsize=18)
    ax2.grid(True, linestyle=':', linewidth=0.5)
    # ax2.legend(fontsize=16)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_intra_condition_similarity(dataset_name, save_path=None):
    """
    For each of the predefined RSA files, plot intra-condition similarity with the
    same visual style as used in plot_ed_results. Concatenate the five plots in a single row.
    """
    file_names = [f'rsa_weight_{dataset_name}.npy',
                  f'rsa_pre_relu_{dataset_name}.npy',
                  f'rsa_avg_pre_relu_{dataset_name}.npy',
                  f'rsa_post_relu_{dataset_name}.npy',
                  f'rsa_avg_post_relu_{dataset_name}.npy']
    subtitles = ['Weight', 'Pre-ReLU', 'Avg Pre-ReLU', 'Post-ReLU', 'Avg Post-ReLU']

    lambda_labels = ['0.1', '0.3', '0.5', '1.0', '2.0', '3.0']
    x = list(range(len(lambda_labels)))

    color_ctrl = 'red'
    color_as = 'blue'
    color_ws = 'orange'

    n_plots = len(file_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(3.8 * n_plots, 3.8), squeeze=False)
    axes = axes[0]
    fig.suptitle(dataset_name.upper(), fontsize=22)

    def _plot_with_shade(ax, x, y, sem, color, label, linestyle='-'):
        ax.plot(x, y, linestyle=linestyle, color=color, label=label)
        ax.fill_between(x, np.array(y) - np.array(sem), np.array(y) + np.array(sem), color=color, alpha=0.2)

    for i, file_name in enumerate(file_names):
        ax = axes[i]
        file_path = os.path.join('./results/stat/', file_name)
        if not os.path.isfile(file_path):
            ax.text(0.5, 0.5, 'missing file', ha='center', va='center', fontsize=12)
            ax.set_title(subtitles[i], fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(lambda_labels, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=':', linewidth=0.5)
            continue

        intra_condition_data = np.load(file_path, allow_pickle=True)
        res = np.array(analyze_intra_condition_similarity(intra_condition_data))
        # CTRL (constant across lambda)
        ctrl_mean, ctrl_sem = res[0][0], res[0][1]
        ctrl_mean = np.full_like(x, ctrl_mean, dtype=float)
        ctrl_sem  = np.full_like(x, ctrl_sem, dtype=float)
        # AS
        as_mean, as_sem = res[1:7][:, 0], res[1:7][:, 1]
        # WS
        ws_mean, ws_sem = res[7:][:, 0], res[7:][:, 1]
        # print(ctrl_mean, ctrl_sem, as_mean, as_sem, ws_mean, ws_sem)

        if ctrl_mean is not None:
            _plot_with_shade(ax, x, ctrl_mean, ctrl_sem, color_ctrl, 'Control', linestyle='--')
        if as_mean is not None:
            _plot_with_shade(ax, x, as_mean, as_sem, color_as, 'AS')
        if ws_mean is not None:
            _plot_with_shade(ax, x, ws_mean, ws_sem, color_ws, 'WS')

        ax.set_xticks(x)
        ax.set_xticklabels(lambda_labels, fontsize=12)
        ax.set_xlabel(r'$\lambda$', fontsize=14)
        if i == 0:
            ax.set_ylabel('Similarity (Pearson $R^2$)', fontsize=14)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title(subtitles[i], fontsize=14)
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)

        if i == 0:
            ax.legend(fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
        

# def plot_inter_condition_similarity(dataset_name, save_path=None, subtitles=None):
#     """
#     For each of the predefined RSA files, plot inter-condition similarity as block averages.
#     Each subplot shows a 13x13 matrix (conditions) where each cell is the mean similarity
#     between two conditions (averaged over the 10x10 block).
#     """
#     import matplotlib
#     file_names = [f'rsa_weight_{dataset_name}.npy',
#                   f'rsa_pre_relu_{dataset_name}.npy',
#                   f'rsa_avg_pre_relu_{dataset_name}.npy',
#                   f'rsa_post_relu_{dataset_name}.npy',
#                   f'rsa_avg_post_relu_{dataset_name}.npy']
#     if subtitles is None:
#         subtitles = ['Weight', 'Pre-ReLU', 'Avg Pre-ReLU', 'Post-ReLU', 'Avg Post-ReLU']

#     conditions = [
#         'ctrl', 'as_0.1', 'as_0.3', 'as_0.5', 'as_1', 'as_2', 'as_3',
#         'ws_0.1', 'ws_0.3', 'ws_0.5', 'ws_1', 'ws_2', 'ws_3'
#     ]
#     n_cond = len(conditions)
#     block_size = 10
#     n_plots = len(file_names)
#     fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), squeeze=False)
#     axes = axes[0]

#     fig.suptitle(dataset_name.upper(), fontsize=22)

#     for i, file_name in enumerate(file_names):
#         ax = axes[i]
#         file_path = os.path.join('./results/stat/', file_name)
#         if not os.path.isfile(file_path):
#             ax.text(0.5, 0.5, 'missing file', ha='center', va='center', fontsize=12)
#             ax.set_title(subtitles[i], fontsize=14)
#             ax.axis('off')
#             continue

#         rsa = np.load(file_path, allow_pickle=True)
#         block_avg = np.zeros((n_cond, n_cond))
#         for j in range(n_cond):
#             for k in range(n_cond):
#                 # Define the slice for the current block
#                 row_start, row_end = j * block_size, (j + 1) * block_size
#                 col_start, col_end = k * block_size, (k + 1) * block_size
#                 # Extract the block
#                 block = rsa[row_start:row_end, col_start:col_end]
#                 # If it's a diagonal block (e.g., comparing 'ctrl' with 'ctrl'),
#                 # average only the off-diagonal values.
#                 if j == k:
#                     # Extract the values off the main diagonal
#                     off_diagonal_values = block[~np.eye(block_size, dtype=bool)]
#                     block_avg[j, k] = np.mean(off_diagonal_values)
#                 else:
#                     # If it's an off-diagonal block, average all values in the block
#                     block_avg[j, k] = np.mean(block)

#         im = ax.imshow(block_avg, cmap='viridis', vmin=0, vmax=1)
#         ax.set_title(subtitles[i], fontsize=14)
#         ax.set_xticks(range(n_cond))
#         ax.set_yticks(range(n_cond))
#         ax.set_xticklabels(conditions, rotation=90, fontsize=9)
#         ax.set_yticklabels(conditions, fontsize=9)
#         ax.tick_params(axis='both', which='major', labelsize=9)
#         ax.grid(False)
#         # Colorbar only for last subplot
#         if i == n_plots - 1:
#             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Mean similarity (Pearson $R^2$)')

#     plt.tight_layout()
#     if save_path is not None:
#         fig.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()


def plot_grouped_condition_similarity(dataset_name, save_path=None):
    """
    For each RSA file, plots the 3x3 inter-condition similarity, grouped into Ctrl, AS, and WS.
    Each cell shows the mean similarity between two major experimental groups.
    """
    # Define file names and plot titles
    file_names = [f'rsa_weight_{dataset_name}.npy',
                  f'rsa_pre_relu_{dataset_name}.npy',
                  f'rsa_avg_pre_relu_{dataset_name}.npy',
                  f'rsa_post_relu_{dataset_name}.npy',
                  f'rsa_avg_post_relu_{dataset_name}.npy']
    subtitles = ['Weight', 'Pre-ReLU', 'Avg Pre-ReLU', 'Post-ReLU', 'Avg Post-ReLU']

    # --- Define the 3 main groups and their boundaries ---
    group_labels = ['Ctrl', 'AS', 'WS']
    group_slices = {
        'Ctrl': slice(0, 10),    # The first condition (10 items)
        'AS':   slice(10, 70),   # The next 6 conditions (60 items)
        'WS':   slice(70, 130)   # The final 6 conditions (60 items)
    }
    slices_list = [group_slices[label] for label in group_labels]
    n_groups = len(group_labels)
    
    # --- Create the plot ---
    n_plots = len(file_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)
    axes = axes.ravel() # Flatten the axes array for easy iteration

    fig.suptitle(f'{dataset_name.upper()}', fontsize=18, y=1.05)

    for i, file_name in enumerate(file_names):
        ax = axes[i]
        file_path = os.path.join('./results/stat/', file_name)

        # Handle missing files gracefully
        if not os.path.isfile(file_path):
            ax.text(0.5, 0.5, 'missing file', ha='center', va='center', fontsize=12)
            ax.set_title(subtitles[i], fontsize=14)
            ax.axis('off')
            continue

        # Load the full 130x130 RDM
        # NOTE: Assuming similarity (1 is high sim), not dissimilarity.
        # If your matrix is dissimilarity, the interpretation of colors will be reversed.
        full_rdm = np.load(file_path, allow_pickle=True)
        
        # --- Perform the 3x3 block averaging ---
        grouped_rdm = np.zeros((n_groups, n_groups))
        for j in range(n_groups):
            for k in range(n_groups):
                slice1, slice2 = slices_list[j], slices_list[k]
                block = full_rdm[slice1, slice2]
                print(block.shape)
                
                if j == k: # For diagonal blocks (e.g., AS vs AS)
                    mask = ~np.eye(block.shape[0], dtype=bool)
                    grouped_rdm[j, k] = np.mean(block[mask])
                else: # For off-diagonal blocks (e.g., AS vs WS)
                    grouped_rdm[j, k] = np.mean(block)

        # --- Plotting with Seaborn for better annotations ---
        # Note: Set vmin/vmax based on your data's expected range
        sns.heatmap(
            grouped_rdm,
            ax=ax,
            annot=True,
            fmt=".2f",           # Display 2 decimal places
            cmap='RdBu',       # Blue-Orange colormap (Red-Blue reversed)
            # vmin=0.0, vmax=1.0,  # Assuming similarity from 0 to 1
            square=True,
            linewidths=.5,
            xticklabels=group_labels,
            yticklabels=group_labels if i == 0 else False, # Only label y-axis on the first plot
            cbar=i == n_plots - 1, # Only draw colorbar on the last plot
            cbar_kws={
                'label': 'Mean similarity (Pearson $R^2$)',
                'fraction': 0.046, # Adjust fraction to control colorbar size relative to plot
                'shrink': 0.8 # Adjust shrink to control colorbar height relative to plot
            }
        )
        ax.set_title(subtitles[i], fontsize=14)

        # --- Manually adjust the colorbar label on the last plot ---
        if i == n_plots - 1:
            cbar = ax.collections[0].colorbar
            cbar.set_label('Mean similarity (Pearson $R^2$)', size=14) # Set label and font size here
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.tight_layout() # Adjust layout to make space for suptitle
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # for dataset_name in ['mnist', 'cifar']:
    #     for type_noise in ['poisson']:#, 'pink', 'salt_pepper']:
    #         baseline_summary = process_baseline_results(f'./results/acc/{dataset_name}.npy')
    #         noise_as_ws_summary, noise_control_summary = process_noise_results(f'./results/acc/{dataset_name}_noise_{type_noise}.npy', baseline_summary)
    #         plot_noise_results(noise_as_ws_summary, noise_control_summary, baseline_summary,
    #                         save_path=f'./results/fig/{dataset_name}_noise_{type_noise}.png')

    # for dataset_name in ['mnist', 'cifar']:
    #     baseline_summary = process_baseline_results(f'./results/acc/{dataset_name}.npy')
    #     plot_noise_comparison(dataset_name, baseline_summary, save_path=f'./results/fig/{dataset_name}_noise_comparison.png')

    # for dataset_name in ['mnist', 'cifar']:
    #     plot_noise_examples(dataset_name=dataset_name, save_path=f'./results/fig/{dataset_name}_noise_examples.png')

    # # Example usage for Moran's I results
    # for dataset_name in ['mnist', 'cifar']:
    #     morans_i_sem_summary, morans_i_control_summary = process_morans_i_results(file_path=f"./results/stat/{dataset_name}_morans_i_batch_fill.npy")
    #     plot_morans_i_results(morans_i_sem_summary, morans_i_control_summary, dataset_name, save_path=f'./results/fig/{dataset_name}_morans_i.png')


    # for dataset_name in ['mnist', 'cifar']:
    #     ed_weight_avg, ed_weight_sem, ed_act_avg, ed_act_sem = process_ed_results(dataset_name)
    #     plot_ed_results(ed_weight_avg, ed_weight_sem, ed_act_avg, ed_act_sem,
    #                     dataset_name, save_path=f'./results/fig/{dataset_name}_ed.png')

    for dataset_name in ['mnist', 'cifar']:
        plot_intra_condition_similarity(dataset_name, save_path=f'./results/fig/{dataset_name}_intra_condition_similarity_new.png')
        # plot_inter_condition_similarity(dataset_name)
        # plot_grouped_condition_similarity(dataset_name, save_path=f'./results/fig/{dataset_name}_grouped_condition_similarity.png')
