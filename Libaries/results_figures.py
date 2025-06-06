import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc
import re
import bw2data as bd
import brightway2 as bw


# import life_cycle_assessment as lc
import lcia_results as lr
from standards import *
import database_manipulation as dm

from lca import LCA

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

lca_init = LCA(path=path,matching_database=matching_database)

def endpoint_new_name():
    endpoints_new_name = [
            "Ecosystem damage",
            "Human health damage",
            "Natural resources damage"
        ]
    
    return endpoints_new_name

def obtain_impact_category_units():
    endpoints_new_name = endpoint_new_name()
    impact_cat_unit_dct = {}
    end_counter = 0
    for m in bw.methods:
        if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m):
            method = bw.Method(m)
            unit = method.metadata.get('unit', 'No unit found')
            impact_cat_unit_dct[str(m[2]).capitalize()] = unit
            # print(f"Method: {m[2]}, Unit: {unit}")
        elif 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m):
            method = bw.Method(m)
            unit = method.metadata.get('unit', 'No unit found')
            

            impact_cat_unit_dct[endpoints_new_name[end_counter]] = unit
            end_counter += 1
            
    return impact_cat_unit_dct

def organize_dataframe_index(df):
    df_tot_T = df.T
    ic_idx_lst = []
    for i in lca_init.lcia_impact_method():
        ic_idx_lst.append(str(i[2]).capitalize())

    endpoints_new_name = endpoint_new_name()

    for end in endpoints_new_name:
        for x, iil in enumerate(ic_idx_lst):
            iil = iil.replace(" quality", "")
            if iil.lower() in end.lower():
                ic_idx_lst[x] = end

    df_tot_T.index = ic_idx_lst
    
    return df_tot_T

def save_totals_to_excel(df):
   
    impact_cat_unit_dct = obtain_impact_category_units()
    df_tot_T = organize_dataframe_index(df)

    for idx in df_tot_T.index:
        df_tot_T.at[idx,"Unit"] = impact_cat_unit_dct[idx]

    file_path_tot = rf"{lca_init.results_path}\LCIA\penincillium_totals.xlsx"

    save_LCIA_results(df_tot_T, file_path_tot, "totals")

def print_min_max_val(scaled_df):
    min_val = None
    max_val = 0
    for col in scaled_df.columns:
        for idx, row in scaled_df.iterrows():
            val = row[col]
            if min_val is None or val < min_val:
                min_val = val
            elif val != 1 and val > max_val:
                max_val = val
    print(f"Mininum valúe : {min_val}, Maximum value : {max_val}")


def data_set_up(reload=False, calc=False):
    dm.database_setup(path, matching_database, reload=reload)
    lcia_method = lca_init.lcia_meth

    # Perform quick LCIA (Life Cycle Impact Assessment) and get the results
    df = lr.obtain_LCIA_results(calc)
    save_totals_to_excel(df)

    # Process the data
    df_res = lca_init.dataframe_results_handling(df)

    if type(df_res) is list:
        df_mid, df_endpoint = df_res
    # Scale the data for midpoint and endpoint
    df_scaled_mid = lca_init.dataframe_element_scaling(df_mid)
    
    if 'recipe' in lcia_method.lower():
        df_scaled_end = lca_init.dataframe_element_scaling(df_endpoint)

    # Extract the GWP (Global Warming Potential) column
    df_col = [df_mid.columns[1]]
    df_GWP = df_mid[df_col]
    
    print_min_max_val(df_scaled_mid)

    return [df_scaled_mid, df_scaled_end, df_GWP, df_mid, df_endpoint]

def plot_text_size():
    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 11 # Legend font size
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
    recipe = 'Midpoint (H)'
    colors = color_range(colorname="coolwarm", color_quantity=2)

    # Extract columns and indices for plotting
    columns_to_plot = df.columns
    index_list = list(df.index.values)

    
    # dpi = 400
    # width_in = 3543 / dpi
    # height_in = width_in * 0.6


    # Create the plot
    width_in, height_in, dpi = plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
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
    ax.set_title(mid_end_figure_title(recipe)+" results for 1 treatment")  
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax.set_yticks(np.arange(0, 1 + 0.001, step=0.1))
    
    y_ticks = plt.gca().get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in y_ticks])
    x_pos = 0.94

    fig.legend(
        mid_end_legend_text(df),
        loc='upper left',
        bbox_to_anchor=(0.975, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    # Save the plot with high resolution
    output_file = join_path(
        lca_init.path_github,
        f'figures\{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
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
        lca_init.path_github,
        f'{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def create_results_figures(reload=False, calc=False):
    # Set the current Brightway project
    bw_project = lca_init.bw_project
    bd.projects.set_current(bw_project)

    folder = results_folder(lca_init.path_github, "figures")

    impact_categories = lca_init.lcia_impact_method()
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]
    
    # # Extract the endpoint categories from the plot x-axis
    # plot_x_axis_end = [
    #         "Ecosystem\n damage",
    #         "Human health\n damage",
    #         "Natural resources\n damage"
    #     ]
    
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
    
    data = data_set_up(reload=reload, calc=calc)
    
    midpoint_graph(data[0], 'recipe', plot_x_axis_mid, folder)
    # endpoint_graph(data[1], 'recipe', plot_x_axis_end, folder)

    return data
