import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def uncertainty_plot(path, inputs, plot_structure):
    # Unpack plot_structure and inputs
    c0, c1, c2, plot_label, save_name, y_min, y_max, ystep, life_time = plot_structure
    colors, save_dir = inputs

    # Set general font size parameters
    plt.rcParams.update({
        'font.size': 16,      # General font size
        'axes.titlesize': 18, # Title font size
        'axes.labelsize': 16, # Axis labels font size
        'legend.fontsize': 14 # Legend font size
    })

    # Read the data from the specified path
    box_plot_df = pd.read_excel(path)
    df_bp = ((box_plot_df['Use pr day']).dropna()).to_frame()

    # Scale values based on the lifetime
    for col in df_bp.columns:
        df_bp[col] *= 365 * life_time

    # Create the figure and axes with the specified size
    fig, ax = plt.subplots(figsize=(7, 5))  # Updated figure size to (7, 5)

    # Create the boxplot
    boxplot = sns.boxplot(
        data=df_bp,
        flierprops=dict(marker='o', color='red', markerfacecolor='w', markersize=6),
        color=colors[c0],
        linewidth=0,  # General line width
        width=4  # Adjusted box width
    )

    # Customize the box and whisker elements
    for patch in boxplot.artists:
        patch.set_facecolor(colors[c0])  # Set box face color
        patch.set_edgecolor(colors[c0])  # Set box edge color
        patch.set_linewidth(3)  # Set box edge line width

    for i, line in enumerate(boxplot.lines):
        # Customize median lines
        if i % 6 == 4:
            line.set_color(colors[c1])  # Set median line color
            line.set_linewidth(2)  # Set median line width
        # Customize whiskers
        elif i % 6 == 0 or i % 6 == 1:
            line.set_color(colors[c0])  # Set whisker color
            line.set_linewidth(2)  # Set whisker line width
        # Customize caps
        elif i % 6 == 2 or i % 6 == 3:
            line.set_color(colors[c0])  # Set cap color
            line.set_linewidth(2)  # Set cap line width

    # Add the mean value as a scatter point
    mean_value = df_bp.mean().values[0]
    ax.scatter(x=0, y=mean_value, color=colors[c2], marker='D', label='Mean', zorder=2)

    # Add the legend with explanations
    legend_elements = [
        plt.Line2D([0], [0], color=colors[c0], lw=6, label='Q1 to Q3'),
        plt.Line2D([0], [0], color=colors[c1], lw=2, label='Median'),
        plt.Line2D([0], [0], color=colors[c2], marker='D', linestyle='None', markersize=6, label='Mean')
    ]

    plt.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),  # Adjust positioning below the plot
        ncol=3,
        frameon=False
    )

    # Customize labels, ticks, and limits
    plt.ylabel(plot_label)  # Set y-axis label
    ax.get_xaxis().set_visible(False)  # Hide x-axis since it's not meaningful here
    plt.yticks(np.arange(y_min, y_max + 0.001, step=ystep))  # Set y-axis ticks
    plt.ylim(y_min - 0.001, y_max + 0.005)  # Set y-axis limits

    # Save the plot to the specified directory
    output_file = os.path.join(save_dir, f'boxplot_{save_name}.png')
    plt.tight_layout()  # Adjust layout to fit elements
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')  # Save the plot
    plt.show()  # Display the plot
