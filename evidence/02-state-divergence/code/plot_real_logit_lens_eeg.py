import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# ---- Plot config ----
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_eeg_figure():
    _HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_HERE, "..", "data", "real_logit_lens_data.json")
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    def get_mean_trajectory(condition):
        if condition not in data or not data[condition]:
            return np.zeros(29)
        all_probs = [item["layer_probs"] for item in data[condition]]
        min_len = min(len(p) for p in all_probs)
        truncated_probs = [p[:min_len] for p in all_probs]
        return np.mean(truncated_probs, axis=0)
        
    ot_trajectory = get_mean_trajectory("over_thinking")
    nr_trajectory = get_mean_trajectory("normal_thinking")
    ut_trajectory = get_mean_trajectory("under_thinking")
    
    layers = np.arange(len(ot_trajectory))
    num_layers = len(layers)
    
    # We will also create some variations around the mean for the 3D aesthetic
    # to represent the 'bundle' of traces in the EEG metaphor
    
    # Style via plot_config
    setup_style({"font.size": 7})
    bg_color = '#FFFFFF'
    grid_color = '#E0E0E0'

    fig = plt.figure(figsize=(18, 6), dpi=300)
    
    # Okabe-Ito three-state colors (CLAUDE.md spec)
    conditions = [
        ('Premature convergence\n(Wait, think.)', ut_trajectory, OI["vermillion"], '#E67E22'),
        ('Delayed commitment\n(Let\'s reason through this step by step.)', nr_trajectory, OI["bluish_green"], '#33B893'),
        ('Persistent oscillation\n(Let\'s think step by step.)', ot_trajectory, OI["blue"], '#3399CC')
    ]
    
    for i, (title, mean_traj, color_main, color_glow) in enumerate(conditions):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_facecolor(bg_color)
        
        # Plot multiple traces around the mean to create the EEG effect
        num_traces = 15
        
        for j in range(num_traces):
            # add noise for visual thickness
            noise = np.random.normal(0, 0.02, len(mean_traj))
            # smooth the noise
            noise = smooth(noise, 3)
            trace = np.clip(mean_traj + noise, 0, 1)
            
            # Z is probability, X is layer, Y is just a slight offset for 3D width
            y_offset = np.random.normal(0, 0.5)
            
            alpha = np.random.uniform(0.1, 0.4)
            linewidth = np.random.uniform(0.5, 1.5)
            
            ax.plot(layers, np.full(len(layers), y_offset), trace, 
                    color=color_glow, alpha=alpha, linewidth=linewidth)
            
        # Plot the main trace with same thickness as others
        ax.plot(layers, np.zeros(len(layers)), mean_traj, 
                color=color_main, alpha=1.0, linewidth=1.5, label='Mean Probability')
        
        ax.set_title(title, color='#000000', fontsize=14, pad=20, fontweight='bold')
        
        ax.set_xlim(0, len(layers)-1)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 1.0)
        
        ax.set_xlabel('\nModel Layers', color='#333333', fontsize=10, labelpad=10)
        ax.set_ylabel('\nOffset', color='#333333', fontsize=10, labelpad=10)
        ax.set_zlabel('\nToken Probability', color='#333333', fontsize=10, labelpad=10)
        
        # Set x-axis ticks to show only key layers (8, 16, 24, 32 if available)
        max_layer = len(layers) - 1
        xticks = []
        for tick in [8, 16, 24, 32]:
            if tick <= max_layer:
                xticks.append(tick)
        if 0 not in xticks:
            xticks.insert(0, 0)
        if max_layer not in xticks:
            xticks.append(max_layer)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=8)
        
        ax.set_yticks([])
        
        ax.view_init(elev=20, azim=-60)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(grid_color)
        ax.yaxis.pane.set_edgecolor(grid_color)
        ax.zaxis.pane.set_edgecolor(grid_color)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1)
    
    # 移除主标题
    # fig.suptitle('Logit Lens: Trajectory of Target Token Probability Across Layers (Real 1.5B Data)', 
    #              fontsize=18, color='#000000', fontweight='bold', y=0.98)
    
    out_dir = os.path.join(_HERE, "..", "figures")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_figure(fig, "Fig7_LogitLens_32runs_3d_eeg", out_dir)
    print(f"Saved figures to {out_dir}")

if __name__ == "__main__":
    plot_eeg_figure()
