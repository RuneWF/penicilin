import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from copy import deepcopy as dc
import re
import bw2data as bd


import life_cycle_assessment as lc
import lcia_results as lr
from standards import *


def category_organization(database_name):

    if 'case1' in database_name :
        category_mapping = {
        "Raw mat. + prod.": ["Raw mat. + prod."],
        "Use": ["Disinfection", "Autoclave"],
        "Recycling": ["Recycling"],
        "EoL": ["Incineration", "Avoided energy prod.", "Avoided mat. prod." ],
        "Net impact": ["Net impact"]
        }
    
    elif 'case2' in database_name or 'model' in database_name:
        category_mapping = {
        "Raw mat. + prod.": ["Raw mat. + prod."],
        "Use": ["Use",  "Disinfection",  "Ster. consumables", "Ster. autoclave"],
        "EoL": ["Incineration", "Avoided energy prod."],
        "Net impact": ["Net impact"]
        }

    return category_mapping

# Function to update the flow name and simplify them
def flow_name_update(x, gwp, case):
    x_og = x

    if '1' in case:
        
        if 'H200' in x or 'H400' in x or 'alubox (small)' in x or 'alubox (large)' in x:
            # print(x)
            x = 'Raw mat. + prod.' 
        if 'market for polypropylene' in x or 'polyethylene, high density' in x and 'waste' not in x:
            # print(f'Plastic {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(f'recycling {x} = {gwp}')
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            # print(f'paper {x} = {gwp}')
            x = 'Recycling'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'
        if 'autoclave' in x:
            x = 'Autoclave'
        if 'transport' in x:
            x = 'Raw mat. + prod.'
        if 'polysulfone' in x:
            x = 'Raw mat. + prod.'
        if 'cast alloy' in x:
            # print(f'Cast alloy {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'
        if '18/8' in x:
            x = "Avoided mat. prod."

    elif '2' in case:
        if 'H200' in x:
            x = 'Ster. consumables' 
        if 'autoclave' in x.lower():
            x = 'Ster. autoclave' 
        if ('MUD' in x or 'SUD' in x) and 'eol' not in x or 'transport' in x or 'scalpel' in x:
            x = "Raw mat. + prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            x = 'Avoided mat. prod.'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'

        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'
        if 'eol' in x:
            x = 'Incineration'
        if 'use' in x:
            x = 'Use'
          

    return x, gwp

def process_categorizing(df_GWP, case, flow_legend, columns):
    x_axis = []
    GWP_value = []
    raw_dct = {}
    rec_dct = {}
    comp  = {}
    for col in df_GWP.columns:
        
        for i, row in df_GWP.iterrows():
            # print(f'idx = {i}')
            lst_x = []
            lst_GWP = []
            gwp_tot = 0
            for lst in row[col]:
                x = lst[0]
                gwp = lst[1]
                    
                x, gwp = flow_name_update(x, gwp, case)

                lst_x.append(x)
                lst_GWP.append(gwp)
                gwp_tot += gwp


            # print(gwp_tot, lst_GWP)
            lst_GWP.append(gwp_tot)
            lst_x.append('Net impact')
            x_axis.append(lst_x)
            GWP_value.append(lst_GWP)

    for key, item in raw_dct.items():
        comp[key] = rec_dct[key]/item*100

    # Create an empty dictionary to collect the data
    key_dic = {}

    # Collect the data into the dictionary
    for scenario, lst in enumerate(GWP_value):
        for idx, gwp in enumerate(lst):
            key = (flow_legend[scenario], x_axis[scenario][idx])
            # print(x_axis[i][a])
            if key in key_dic:
                key_dic[key] += gwp
            else:
                key_dic[key] = gwp

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(key_dic.items()), columns=['Category', 'Value'])

    # Separate 'Total' values from the rest
    totals_df = df[df['Category'].apply(lambda x: x[1]) == 'Net impact']
    df = df[df['Category'].apply(lambda x: x[1]) != 'Net impact']

    # Pivot the DataFrame to have a stacked format
    df_stacked = df.pivot_table(index=[df['Category'].apply(lambda x: x[0])], columns=[df['Category'].apply(lambda x: x[1])], values='Value', aggfunc='sum').fillna(0)

    # Create a DataFrame to store results

    df_stack_updated = pd.DataFrame(0, index=flow_legend, columns=columns[:-1], dtype=object)  # dtype=object to handle lists
    for col in df_stack_updated.columns:
        for inx, row in df_stack_updated.iterrows():
            # print(df_stacked[col])
            try:
                row[col] = df_stacked[col][inx]

            except KeyError:
                # print(f"keyerror at {inx}")
                pass

                
    return df_stack_updated, totals_df

def save_totals_to_excel(method, df):
   
    df_tot, _ = lc.dataframe_element_scaling(df)
    df_tot_T = df_tot.T

    method_updated = []

    for m in method:
        method_updated.append(m[1:])

    df_tot_T.index = method_updated

    file_path_tot = r"C:\Users\ruw\Desktop\RA\penicilin\results\sensitivity\penincillium_totals.xlsx"

    save_LCIA_results(df_tot_T, file_path_tot, "totals")

def data_set_up(path, matching_database, database, lcia_method, bw_project):
    # Initialize the life cycle assessment and get file information
    file_info, initialization = lc.initilization(path, matching_database, database, lcia_method, bw_project)
    _, file_name, file_name_unique_process = file_info 

    # Perform quick LCIA (Life Cycle Impact Assessment) and get the results
    df, plot_x_axis_all, impact_categories = lr.quick_LCIA(initialization, file_name, file_name_unique_process, "penicillin")

    # Separate the scenarios from the results
    # data_df = scenario_seperation(df_all)
    database_name = initialization[1]

    # data = []

    # Process each scenario's data
    # for sc, df in data_df.items():
    df_res, plot_x_axis_lst = lc.dataframe_results_handling(df, database_name, plot_x_axis_all, initialization[3])

    if type(df_res) is list:
        df_mid, df_endpoint = df_res
    # Scale the data for midpoint and endpoint
    df_tot_mid, df_scaled_mid = lc.dataframe_element_scaling(df_mid)
    
    if 'recipe' in initialization[3].lower():
        df_tot_end, df_scaled_end = lc.dataframe_element_scaling(df_endpoint)

    # Extract the GWP (Global Warming Potential) column
    df_col = [df_mid.columns[1]]
    df_GWP = df_mid[df_col]

    # Store the processed data in the dictionary
    # data = [df_scaled_mid, df_scaled_end, df_GWP, df_tot_mid, df_tot_end]
  
    return [df_scaled_mid, df_scaled_end, df_GWP, df_tot_mid, df_tot_end]

def legend_text(text):
    if '1' in text:
        flow_leg = [
                        'H2I',
                        'H2R',
                        'ASC',
                        'ASW',
                        'H4I',
                        'H4R',
                        'ALC',
                        'ALW'
                        ]
        return flow_leg
    else:
        return ['SUD', 'MUD']
    
def xlim(case, ax1, ax2, columns_to_plot):
    if '2' in case:
        ax1.set_xlim(-0.35, len(columns_to_plot) -0.3)
        ax2.set_xlim(-0.35, len(columns_to_plot) -0.3)
    else:
        ax1.set_xlim(-0.2, len(columns_to_plot))
        ax2.set_xlim(-0.2, len(columns_to_plot))

def plot_text_size():
    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 10 # Legend font size
    }) 

def plot_title_text(lca_type):
    if 'consq' in lca_type:
        return 'Consequential'
    elif 'cut' in lca_type:
        return 'Allocation cut-off by Classification'
    else:
        return ''

def mid_end_legend_text(df):
    leg_idx = []
    for idx in df.index:
        txt = idx.replace(f", Defined daily dose", "")
        leg_idx.append(txt)

    return leg_idx

def midpoint_graph(df, recipe, plot_x_axis, folder):
    plot_text_size()
    recipe = 'Midpoint (H)'
    colors = color_range(colorname="Greys", color_quantity=2)

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
               hatch="///",
               edgecolor="k")

    # Set title and labels
    ax.set_title(f"{recipe}")  
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax.set_yticks(np.arange(0, 1 + 0.001, step=0.1))


    # xlim(sc, ax, columns_to_plot)

    x_pos = 0.94

    # leg_idx = mid_end_legend_text(df, sc)


    fig.legend(
        mid_end_legend_text(df),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )

    # Save the plot with high resolution
    output_file = os.path.join(
        folder,
        f'{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def endpoint_graph(df, recipe, plot_x_axis_end, folder):
    plot_text_size()
    colors = color_range(colorname="Greys", color_quantity=2)
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
               hatch="//",
               edgecolor="k")
        
    # Set title and labels
    ax.set_title(f"{recipe}")  
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

    # Save the plot with high resolution
    output_file = os.path.join(
        folder,
        f'{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def create_results_figures(path, matching_database, database, lcia_method, bw_project):
    # Set the current Brightway project
    bd.projects.set_current(bw_project)

    path_github, ecoinevnt_paths, system_path = data_paths(path)
    folder = results_folder(join_path(path_github,'results'), "figures")
    impact_categories = lc.lcia_impact_method()
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]
    
     # Extract the endpoint categories from the plot x-axis
    plot_x_axis_end = plot_x_axis_all[-3:]
    
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
    
    data = data_set_up(path, matching_database, database, lcia_method, bw_project)
    
    midpoint_graph(data[0], 'recipe', plot_x_axis_mid, folder)
    endpoint_graph(data[1], 'recipe', plot_x_axis_end, folder)

    return data
