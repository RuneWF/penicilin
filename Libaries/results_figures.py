import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc
import re
import bw2data as bd


# import life_cycle_assessment as lc
import lcia_results as lr
from standards import *

from lca import LCA

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

lca_init = LCA(path=path,matching_database=matching_database)

def save_totals_to_excel(method, df):
   
    df_tot, _ = lca_init.dataframe_element_scaling(df)
    df_tot_T = df_tot.T

    method_updated = []

    for m in method:
        method_updated.append(m[1:])

    df_tot_T.index = method_updated

    file_path_tot = r"C:\Users\ruw\Desktop\RA\penicilin\results\LCIA\penincillium_totals.xlsx"

    save_LCIA_results(df_tot_T, file_path_tot, "totals")

def data_set_up(sheet_name="penicillin"):

    lcia_method = lca_init.lcia_meth

    # Perform quick LCIA (Life Cycle Impact Assessment) and get the results
    df = lr.quick_LCIA(sheet_name)

    # Process the data
    df_res = lca_init.dataframe_results_handling(df)

    if type(df_res) is list:
        df_mid, df_endpoint = df_res
    # Scale the data for midpoint and endpoint
    df_tot_mid, df_scaled_mid = lca_init.dataframe_element_scaling(df_mid)
    
    if 'recipe' in lcia_method.lower():
        df_tot_end, df_scaled_end = lca_init.dataframe_element_scaling(df_endpoint)

    # Extract the GWP (Global Warming Potential) column
    df_col = [df_mid.columns[1]]
    df_GWP = df_mid[df_col]
  
    return [df_scaled_mid, df_scaled_end, df_GWP, df_tot_mid, df_tot_end]

def plot_text_size():
    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 10 # Legend font size
    }) 


def mid_end_legend_text(df):
    leg_idx = []
    for idx in df.index:
        txt = idx.replace(f", defined system", "")
        leg_idx.append(txt)

    return leg_idx

def mid_end_figure_title(recipe):
    lcia_method = lca_init.lcia_impact_method()
    title_txt = ""
    if "Midpoint" in recipe:
        temp = lcia_method[0][0]
        title_txt = temp.replace(" - no biogenic", "")
    elif "Endpoint" in recipe:
        temp = lcia_method[-1][0]
        title_txt = temp.replace(" - no biogenic", "")

    return title_txt

def midpoint_graph(df, recipe, plot_x_axis, folder):
    plot_text_size()
    recipe = 'Midpoint (H)'
    colors = color_range(colorname="coolwarm", color_quantity=2)

    # Extract columns and indices for plotting
    columns_to_plot = df.columns
    index_list = list(df.index.values)

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(7, 5))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df.index):
        values = df.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax.bar(index + i * bar_width, 
               values, bar_width, 
               label=process, 
               color=color,
            #    hatch="///",
               edgecolor="k",
               zorder=10)

    # Set title and labels
    ax.set_title(mid_end_figure_title(recipe))  
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax.set_yticks(np.arange(0, 1 + 0.001, step=0.1))

    x_pos = 0.94

    fig.legend(
        mid_end_legend_text(df),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    # Save the plot with high resolution
    output_file = join_path(
        folder,
        f'{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def endpoint_graph(df, recipe, plot_x_axis_end, folder):
    plot_text_size()
    colors = color_range(colorname="coolwarm", color_quantity=2)
    recipe = 'Endpoint (H)'

    plot_text_size()

    # for sc_idx, (sc, val) in enumerate(data.items()):
    # Extract columns and indices for plotting
    columns_to_plot = df.columns
    index_list = list(df.index.values)

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(7, 5))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df.index):
        values = df.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax.bar(index + i * bar_width, 
               values, bar_width, 
               label=process, 
               color=color, 
            #    hatch="//",
               edgecolor="k",
               zorder=10)
        
    # Set title and labels
    ax.set_title(mid_end_figure_title(recipe))  
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis_end, rotation=0)  # Added rotation here
    ax.set_yticks(np.arange(0, 1 + 0.001, step=0.1))


    # xlim(sc, ax, columns_to_plot)

    x_pos = 0.94

    
    fig.legend(
        mid_end_legend_text(df),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    # Save the plot with high resolution
    output_file = join_path(
        folder,
        f'{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def create_results_figures():
    # Set the current Brightway project
    bw_project = lca_init.bw_project
    bd.projects.set_current(bw_project)

    path_github = lca_init.path_github
    folder = results_folder(join_path(path_github,'results'), "figures")

    impact_categories = lca_init.lcia_impact_method()
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]
    
     # Extract the endpoint categories from the plot x-axis
    plot_x_axis_end = [
            "Ecosystem\n damage",
            "Human health\n damage",
            "Natural resources\n damage"
        ]
    
    # Extract the midpoint categories from the plot x-axis
    ic_mid = plot_x_axis_all[:-3]

    plot_x_axis_mid = []
    # Process each midpoint category to create a shortened version for the plot x-axis
    for ic in ic_mid:
        string = re.findall(r'\((.*?)\)', ic)
        if 'ODPinfinite' in string[0]:
            string[0] = 'ODP'
        elif '1000' in string[0]:
            string[0] = 'GWP'
        plot_x_axis_mid.append(string[0])
    
    data = data_set_up()
    
    midpoint_graph(data[0], 'recipe', plot_x_axis_mid, folder)
    endpoint_graph(data[1], 'recipe', plot_x_axis_end, folder)

    return data
