#!/usr/bin/env python3 
"""
Figure and Charts for the paper on Dynamic subscription. 
Author: Dhananjay Singh Chauhan 
Date: July 2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os

# Configure matplotlib for publication-quality black and white figures
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'gray',
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
def create_output_directory():
    """Create output directory for figures"""
    if not os.path.exists('figures'):
        os.makedirs('figures')
    print("Output directory 'figures/' created")

def figure_1_theoretical_procrastination_window():
    """Figure 1: Procrastination Window Illustration"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parameters
    beta = 0.6
    delta = 0.9
    delta_w = 8
    
    cost_range = np.linspace(0, 12, 100)
    time_consistent_threshold = delta * delta_w
    naive_threshold = beta * delta * delta_w
    
    # Plot thresholds
    ax.axhline(y=time_consistent_threshold, color='black', linestyle='-', linewidth=2, 
               label=f'Time-consistent: δΔW = {time_consistent_threshold:.1f}')
    ax.axhline(y=naive_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Naive: βδΔW = {naive_threshold:.1f}')
    
    # Shade procrastination window
    ax.fill_between([0, 12], naive_threshold, time_consistent_threshold, 
                    alpha=0.3, color='gray', label='Procrastination Window')
    
    # Mark effective cost
    effective_cost = 5
    ax.axvline(x=effective_cost, color='black', linestyle=':', linewidth=2,
               label=f'Effective Cost = {effective_cost}')
    
    # Annotations
    ax.annotate('Naive stays', xy=(3, 3), fontsize=11, ha='center')
    ax.annotate('Time-consistent cancels', xy=(9, 8.5), fontsize=11, ha='center')
    
    ax.set_xlabel('Effective Cost: c̃ + (v - p_m)', fontsize=12)
    ax.set_ylabel('Decision Threshold', fontsize=12)
    iax.set_title('Procrastination Window: βδΔW ≤ c̃ + (v-p_m) < δΔW\n' + 
                 f'Parameters: β={beta}, δ={delta}, ΔW={delta_w}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('figures/figure_1_procrastination_window.png')
    plt.savefig('figures/figure_1_procrastination_window.pdf')
    plt.close()
    print("Generated Figure 1: Theoretical Procrastination Window")

