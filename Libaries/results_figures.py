import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from copy import deepcopy as dc
import re

import life_cycle_assessment as lc
import lcia_results as lr
import standards as s

def color_range():
    cmap = plt.get_cmap('Accent')
    return [cmap(i) for i in np.linspace(0, 1, 9)]

def join_path(path1, path2):
    return os.path.join(path1, path2)

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
    for i, p in enumerate(GWP_value):
        for a, b in enumerate(p):
            key = (flow_legend[i], x_axis[i][a])
            # print(x_axis[i][a])

            if key in key_dic:
                key_dic[key] += b
            else:
                key_dic[key] = b

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

def results_dataframe(initialization, file_name, file_name_unique_process, sheet_name):
    df, plot_x_axis_all, impact_categories, unique = {}, {}, {}, {}
    for key, item in initialization.items():
        print(f"Perfoming LCA for {key}")
        # print(file_name_unique_process[key])
        df[key], plot_x_axis_all[key], impact_categories[key] = lr.quick_LCIA(item, file_name[key], file_name_unique_process[key], sheet_name[key])
        print()
    
    return df, plot_x_axis_all, impact_categories, unique

def data_set_up(path, lcia_method, ecoinevnt_paths, system_path):
    data = {
    'case1' : {},
    'case2' : {}
    }
    flow_legend, file_name, sheet_name, _, initialization, file_name_unique_process = lc.initilization(path, lcia_method, ecoinevnt_paths, system_path)
    df, plot_x_axis_all, impact_categories, unique = results_dataframe(initialization, file_name, file_name_unique_process, sheet_name)

    for key, item in initialization.items():

        database_name = item[1]
        if 'apos' not in database_name:
            if '1' in database_name:
                df_res, plot_x_axis_lst = lc.dataframe_results_handling(df[key], database_name, plot_x_axis_all[key], item[3])
                if type(df_res) is list:
                    df_mid, df_endpoint = df_res
                    plot_x_axis, plot_x_axis_end = plot_x_axis_lst


                _, df_scaled = lc.dataframe_element_scaling(df_mid)
                df_col = [df_mid.columns[1]]
                df_GWP = df_mid[df_col]

                if 'recipe' in item[3].lower():
                    _, df_scaled_e = lc.dataframe_element_scaling(df_endpoint)
                
                # inputs = [flow_legend[key], colors, save_dir[key], item[4], database_name]
                columns = lc.unique_elements_list(database_name)
                df_be, ignore = process_categorizing(df_GWP, database_name, flow_legend[key], columns)

                data['case1'].update({key : [df_scaled, df_scaled_e, df_GWP, df_be]})
            elif '2' in database_name:
                df_res, plot_x_axis_lst = lc.dataframe_results_handling(df[key], database_name, plot_x_axis_all[key], item[3])
                if type(df_res) is list:
                    df_mid, df_endpoint = df_res
                    plot_x_axis, plot_x_axis_end = plot_x_axis_lst


                _, df_scaled = lc.dataframe_element_scaling(df_mid)
                df_col = [df_mid.columns[1]]
                df_GWP = df_mid[df_col]

                if 'recipe' in item[3].lower():
                    _, df_scaled_e = lc.dataframe_element_scaling(df_endpoint)
                
                columns = lc.unique_elements_list(database_name)
                df_be, ignore = process_categorizing(df_GWP, database_name, flow_legend[key], columns)

                # inputs = [flow_legend[key], colors, save_dir[key], item[4], database_name]
                data['case2'].update({key : [df_scaled, df_scaled_e, df_GWP, df_be]})

    return data

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

def midpoint_graph(data, case, recipe, plot_x_axis, initialization, folder):
    plot_text_size()
    colors = color_range()
    df1 = data[case][f'{case}_cut_off'][0]
    df2 = data[case][f'{case}_consq'][0]

    # Extract columns and indices for plotting
    columns_to_plot = df1.columns
    index_list = list(df1.index.values)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df1.index):
        values = df1.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax1.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    for i, process in enumerate(df2.index):
        values = df2.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax2.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    # Format impact category string

    # Set title and labels
    ax1.set_title(f"{plot_title_text('cut')} - {recipe}")  
    ax1.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax1.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax1.set_yticks(np.arange(-1, 1 + 0.001, step=0.2))

    ax2.set_title(f"{plot_title_text('consq')} - {recipe}")
    ax2.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax2.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax2.set_yticks(np.arange(-1, 1 + 0.001, step=0.2))

    xlim(case, ax1, ax2, columns_to_plot)

    x_pos = 0.97


    fig.legend(
        legend_text(initialization[f'{case}_consq'][1]),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )

    # Save the plot with high resolution
    output_file = os.path.join(
        folder,
        f'{recipe}_{case}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def endpoint_graph(data, case, recipe, plot_x_axis_end, initialization, folder):
    plot_text_size()
    colors = color_range()
    recipe = 'endpoint (H)'
    plt.rcParams.update({
        'font.size': 12,      # General font size
        'axes.titlesize': 14, # Title font size
        'axes.labelsize': 12, # Axis labels font size
        'legend.fontsize': 10 # Legend font size
        }) 

    df1 = data[case][f'{case}_cut_off'][1]
    df2 = data[case][f'{case}_consq'][1]

    # Extract columns and indices for plotting
    columns_to_plot = df1.columns
    index_list = list(df1.index.values)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df1.index):
        values = df1.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax1.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    for i, process in enumerate(df2.index):
        values = df2.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax2.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    # Format impact category string

    # Set title and labels
    ax1.set_title(f"{plot_title_text('cut')} - {recipe}")  
    ax1.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax1.set_xticklabels(plot_x_axis_end, rotation=0)  # Added rotation here

    ax2.set_title(f"{plot_title_text('consq')} - {recipe}")
    ax2.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax2.set_xticklabels(plot_x_axis_end, rotation=0)  # Added rotation here


    xlim(case, ax1, ax2, columns_to_plot)

    x_pos = 0.97


    fig.legend(
        legend_text(initialization[f'{case}_consq'][1]),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )


    # Save the plot with high resolution
    output_file = os.path.join(
        folder,
        f'{recipe}_{case}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def gwp_figure_setup(data, case, path, initialization, flow_legend):
    df1gwp = data[case][f'{case}_cut_off'][2]
    df2gwp = data[case][f'{case}_consq'][2]

    flow_legend = legend_text(case)

    folder = s.results_folder(join_path(path,'results'), case)


    _, database_name1, _, _, tp = initialization[f'{case}_cut_off']
    columns1 = lc.unique_elements_list(database_name1)
    
    df1s, totals_df1 = process_categorizing(df1gwp, case, flow_legend, columns1)

    _, database_name2, _, _, tp = initialization[f'{case}_consq']
    columns2 = lc.unique_elements_list(database_name2)
    df2s, totals_df2 = process_categorizing(df2gwp, case, flow_legend, columns2)

    

    return folder, df1s, totals_df1, df2s, totals_df2, columns1

def y_min_max(case):
    if '1' in case:
        y_min1 = -0.6
        y_max1 = 1.8
        y_min2 = -0.6
        y_max2 = 1.8

    else:
        y_min1 = -0.4
        y_max1 = 1.6
        y_min2 = -0.4
        y_max2 = 1.6

    return y_min1, y_max1, y_min2, y_max2

def gwp_figure(data, case, path, initialization):
    flow_legend = legend_text(case)
    folder, df1s, totals_df1, df2s, totals_df2, columns1 = gwp_figure_setup(data, case, path, initialization, flow_legend)
    plot_text_size()
    colors = color_range()
    

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    marker_color = 'k'
    # cut off
    df1s.plot(kind='bar', stacked=True, ax=ax1, color=colors, zorder=2, legend=False)
    ax1.axhline(y=0, color='k', linestyle='-', zorder=0, linewidth=0.5)
    ax1.set_ylabel('Global Warming Potential [kg CO$_2$e/FU]',  fontsize=12)

    # Plotting 'Total' values as dots and including it in the legend
    for idx, row in totals_df1.iterrows():
        unit = row['Category'][0]
        total = row['Value']
        ax1.plot(unit, total, 'D', color=marker_color, markersize=4, mec='k', label='Net impact' if idx == 0 else "")
        # Add the data value
        ax1.text(
            unit, total - 0.12, f"{total:.2f}", 
            ha='center', va='bottom', fontsize=10, 
            color=marker_color)

    # Custom legend with 'Total' included
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_color, mec='k', markersize=4, label='Net impact')
    )

    # CONSQ
    df2s.plot(kind='bar', stacked=True, ax=ax2, color=colors, zorder=2, legend=False)
    ax2.axhline(y=0, color='k', linestyle='-', zorder=0, linewidth=0.5)
    ax2.set_ylabel('Global Warming Potential [kg CO$_2$e/FU]',  fontsize=12)

    # Plotting 'Total' values as dots and including it in the legend
    for idx, row in totals_df2.iterrows():
        unit = row['Category'][0]
        total = row['Value']
        ax2.plot(unit, total, 'D', color=marker_color, markersize=4, mec='k', label='Net impact' if idx == 0 else "")
        # Add the data value
        ax2.text(
            unit, total - 0.15, f"{total:.2f}", 
            ha='center', va='bottom', fontsize=10, 
            color=marker_color)

    # Custom legend with 'Total' included
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_color, mec='k', markersize=4, label='Net impact')
    )

    fig.legend(
        labels=columns1,
        handles=handles,
        loc="upper center",  # Place legend at the bottom center
        bbox_to_anchor=(1.13, 0.97),  # Adjust position to below the x-axis
        ncol=1,  # Display legend entries in 3 columns
        fontsize=10.5,
        frameon=False  # Remove the legend box
    )

    y_min1, y_max1, y_min2, y_max2 = y_min_max(case)

    # Set title, y-ticks, y-limits, and x-tick labels for ax1
    ax1.set_title(plot_title_text('cut'), fontsize=14)
    ax1.set_yticks(np.arange(y_min1,y_max1 + 0.01, step=0.2))
    ax1.set_ylim(y_min1 - 0.05, y_max1 + 0.05)
    ax1.set_xticklabels(legend_text(initialization[f'{case}_cut_off'][1]), rotation=0, fontsize=11)




    # Set title, y-ticks, y-limits, and x-tick labels for ax2
    ax2.set_title(plot_title_text('consq'), fontsize=14)
    ax2.set_yticks(np.arange(y_min2, y_max2 + 0.01, step=0.2))
    ax2.set_ylim(y_min2 - 0.05, y_max2 + 0.05)
    ax2.set_xticklabels(legend_text(initialization[f'{case}_consq'][1]), rotation=0, fontsize=11)

    plt.tight_layout()

    filename = join_path(folder, f'gwp_{case}.png')
    plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
    plt.show()

def break_even_orginization(df_be, case):
    df_be_copy = dc(df_be)
    if '1' in case:
        wipe_small_container = df_be.at['ASW', 'Disinfection']
        wipe_large_container = df_be.at['ALW', 'Disinfection']

        # Avoided energy
        cabinet_small_avoided_energy = df_be.at['ASC', 'Avoided energy prod.']
        wipe_small_avoided_energy = df_be.at['ASW', 'Avoided energy prod.']

        allocate_avoided_energy_S = wipe_small_avoided_energy - cabinet_small_avoided_energy

        cabinet_large_avoided_energy = df_be.at['ALC', 'Avoided energy prod.']
        wipe_large_avoided_energy = df_be.at['ALW', 'Avoided energy prod.']

        allocate_avoided_energy_L = wipe_large_avoided_energy - cabinet_large_avoided_energy



        # Incineration
        cabinet_small_inc = df_be.at['ASC', 'Incineration']
        wipe_small_inc = df_be.at['ASW', 'Incineration']

        allocate_inc_S = wipe_small_inc - cabinet_small_inc

        cabinet_large_inc = df_be.at['ALC', 'Incineration']
        wipe_large_inc = df_be.at['ALW', 'Incineration']

        allocate_inc_L = wipe_large_inc - cabinet_large_inc

        # Recycling
        cabinet_small_rec = df_be.at['ASC', 'Recycling']
        wipe_small_rec = df_be.at['ASW', 'Recycling']

        allocate_rec_S = wipe_small_rec - cabinet_small_rec

        cabinet_large_rec = df_be.at['ALC', 'Recycling']
        wipe_large_rec = df_be.at['ALW', 'Recycling']

        allocate_rec_L = wipe_large_rec - cabinet_large_rec

        # Calculating the new sums

        wipe_small_container_new = wipe_small_container + allocate_avoided_energy_S + allocate_inc_S + allocate_rec_S

        wipe_large_container_new = wipe_large_container + allocate_avoided_energy_L  + allocate_inc_L +allocate_rec_L


        df_be_copy.at['ASW', 'Avoided energy prod.'] = cabinet_small_avoided_energy
        df_be_copy.at['ALW', 'Avoided energy prod.'] = cabinet_large_avoided_energy

        df_be_copy.at['ASW', 'Incineration'] = cabinet_small_inc
        df_be_copy.at['ALW', 'Incineration'] = cabinet_large_inc

        df_be_copy.at['ASW', 'Disinfection'] = wipe_small_container_new
        df_be_copy.at['ALW', 'Disinfection'] = wipe_large_container_new

        df_be_copy.at['ASW', 'Recycling'] = cabinet_small_rec
        df_be_copy.at['ALW', 'Recycling'] = cabinet_large_rec

    return df_be_copy

def break_even_setup(data, case, initialization, path):
    df1be = data[case][f'{case}_cut_off'][3]
    df2be = data[case][f'{case}_consq'][3]

    df1be.index = legend_text(initialization[f'{case}_cut_off'][1])
    df2be.index = legend_text(initialization[f'{case}_consq'][1])

    folder = s.results_folder(join_path(path,'results'), case)

    return df1be, df2be, folder

def be_case1_setup(df1be, case):
    amount_of_uses =513
    
    df_be_copy = break_even_orginization(df1be, case)
    # Split index into small and large based on criteria
    small_idx = [idx for idx in df_be_copy.index if '2' in idx or 'AS' in idx]
    large_idx = [idx for idx in df_be_copy.index if idx not in small_idx]

    # Create empty DataFrames for each scenario
    scenarios = {
        'small': pd.DataFrame(0, index=small_idx, columns=df_be_copy.columns, dtype=object),
        'large': pd.DataFrame(0, index=large_idx, columns=df_be_copy.columns, dtype=object)
    }
    dct = {}
    # Fill scenarios with data
    for sc_idx, (scenario_name, scenario_df) in enumerate(scenarios.items()):
        scenario_df.update(df_be_copy.loc[scenario_df.index])

        use_cycle, production = {}, {}

        for idx, row in scenario_df.iterrows(): 
            use, prod = 0, 0
            for col in df_be_copy.columns:
                if ('autoclave' in col.lower() or 'disinfection' in col.lower()) and 'H' not in idx:
                    use_cycle[idx] = row[col] + use
                    use += row[col]
                elif 'A' in idx:
                    # print(idx, col ,(row[col] + prod) * amount_of_uses)
                    production[idx] = (row[col] + prod) * amount_of_uses
                    prod += row[col]
                    
                else:
                    production[idx] = row[col] + prod
                    prod += row[col]
        
        # Calculate break-even values
        be_dct = {}
        for key, usage in production.items():
                be_dct[key] = []
                for u in range(1, amount_of_uses + 1):
                    # if u == 1:
                    #     be_dct[key].append(usage)
                    # else:
                    be_dct[key].append(use_cycle.get(key, usage) * u + usage)
                    
        dct[scenario_name] = be_dct
    return dct

def be1_figure(data, case, initialization, path):
    plot_text_size()
    colors = color_range()
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    color_idxs = [0, 1, 2, 4]
    color_idxl = [3, 5, 6, 7]
    
    df1be, df2be, folder = break_even_setup(data, case, initialization, path)

    be1 = be_case1_setup(df1be, case)
    s1 = be1['small']
    l1 = be1['large']

    

    # cut off
    for idx, (key, value) in enumerate(s1.items()):
        try:
            if 'H' in key:
                ax1.plot(value, label=key, linestyle='dashed', color=colors[color_idxs[idx] % len(colors)], linewidth=3)
            else:
                ax1.plot(value, label=key, color=colors[color_idxs[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idxs[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    for idx, (key, value) in enumerate(l1.items()):
        try:
            if 'H' in key:
                ax2.plot(value, label=key, linestyle='dashed', color=colors[color_idxl[idx] % len(colors)], linewidth=3)
            else:
                ax2.plot(value, label=key, color=colors[color_idxl[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idxl[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    # consq
    be2 = be_case1_setup(df2be, case)
    s2 = be2['small']
    l2 = be2['large']

    for idx, (key, value) in enumerate(s2.items()):
        try:
            if 'H' in key:
                ax3.plot(value, label=key, linestyle='dashed', color=colors[color_idxs[idx] % len(colors)], linewidth=3)
            else:
                ax3.plot(value, label=key, color=colors[color_idxs[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idxs[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    for idx, (key, value) in enumerate(l2.items()):
        try:
            if 'H' in key:
                ax4.plot(value, label=key, linestyle='dashed', color=colors[color_idxl[idx] % len(colors)], linewidth=3)
            else:
                ax4.plot(value, label=key, color=colors[color_idxl[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idxl[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    # Create custom legend handles and labels
    handles = []
    labels = []

    for idx, key in enumerate(s1.keys()):
        handles.append(plt.Line2D([0], [0], color=colors[color_idxs[idx]], linewidth=3, linestyle='dashed' if 'H' in key else 'solid'))
        labels.append(key)

    for idx, key in enumerate(l1.keys()):
        handles.append(plt.Line2D([0], [0], color=colors[color_idxl[idx]], linewidth=3, linestyle='dashed' if 'H' in key else 'solid'))
        labels.append(key)

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0), ncol=8, fontsize=10.5, frameon=False)

    ax1.set_title(f"{plot_title_text('cut')} - Small")
    ax2.set_title(f"{plot_title_text('cut')} - Large")
    ax3.set_title(f"{plot_title_text('consq')} - Small")
    ax4.set_title(f"{plot_title_text('consq')} - Large")

    ax1.set_xlabel('Cycle(s)')
    ax2.set_xlabel('Cycle(s)')
    ax3.set_xlabel('Cycle(s)')
    ax4.set_xlabel('Cycle(s)')

    ax1.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
    ax2.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
    ax3.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
    ax4.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')

    ax1.set_xlim(0, 513)
    ax2.set_xlim(0, 513)
    ax3.set_xlim(0, 513)
    ax4.set_xlim(0, 513)
    y1 = 350
    y2 = 800
    ax1.set_ylim(0, y1)
    ax2.set_ylim(0, y2)
    ax3.set_ylim(0, y1)
    ax4.set_ylim(0, y2)

    ax1.set_yticks(range(0, y1 + 5, 50))
    ax2.set_yticks(range(0, y2 + 5, 100))
    ax3.set_yticks(range(0, y1 + 5, 50))
    ax4.set_yticks(range(0, y2 + 5, 100))

    # Adjust layout to add a 0.5 cm gap between the figures
    plt.subplots_adjust(hspace=0.6 / 2.54)  # 0.5 cm gap converted to inches

    plt.tight_layout()

    # Save and display plot
    filename = join_path(folder, f'break_even_{case}.png')
    plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
    plt.show()

def be_case2_setup(df_be):
    amount_of_uses = 250
    multi_use, production = {}, {}
    for idx, row in df_be.iterrows(): 
        use, prod = 0, 0
        for col in df_be.columns:
            if 'Disinfection' in col and 'SUD' not in idx:
                multi_use[idx] = row[col] + use
                use += row[col]
            elif 'MUD' in idx:

                production[idx] = (row[col] + prod) * amount_of_uses
                prod += row[col]
            else:
                production[idx] = row[col] + prod
                prod += row[col]

    # Calculate break-even values
    be_dct = {}
    for key, usage in production.items():
        be_dct[key] = []
        for u in range(1, amount_of_uses + 1):
            # if u == 1:
            #     be_dct[key].append(usage)
            # else:
            be_dct[key].append(multi_use.get(key, usage) * u + usage)
        
    return be_dct

def be2_figure(data, case, initialization, path):
    color_idx = [0, 1, 2, 4]
    df1be = data[case][f'{case}_cut_off'][3]
    df2be = data[case][f'{case}_consq'][3]

    df1be, df2be, folder = break_even_setup(data, case, initialization, path)

    dfbe1 = be_case2_setup(df1be)
    dfbe2 = be_case2_setup(df2be)

    

    plot_text_size()
    colors = color_range()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

    for idx, (key, value) in enumerate(dfbe1.items()):
        try:
            if 'RMD' in key or 'SUD' in key:
                ax1.plot(value, label=key, linestyle='dashed', color=colors[color_idx[idx] % len(colors)], linewidth=3)
            else:
                ax1.plot(value, label=key, color=colors[color_idx[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idx[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    for idx, (key, value) in enumerate(dfbe2.items()):
        try:
            if 'RMD' in key or 'SUD' in key:
                ax2.plot(value, label=key, linestyle='dashed', color=colors[color_idx[idx] % len(colors)], linewidth=3)
            else:
                ax2.plot(value, label=key, color=colors[color_idx[idx]], linewidth=3)
        except IndexError:
            print(f'Color index of {color_idx[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')

    # Create custom legend handles and labels
    handles = []
    labels = []

    for idx, key in enumerate(dfbe1.keys()):
        handles.append(plt.Line2D([0], [0], color=colors[color_idx[idx]], linewidth=3, linestyle='dashed' if 'H' in key else 'solid'))
        labels.append(key)

    # Customize plot
    fig.legend(
        handles,
        labels,
        loc="upper center",  # Place legend at the bottom center
        bbox_to_anchor=(0.97, 0.89),  # Adjust position to below the x-axis
        ncol=1,  # Display legend entries in 3 columns
        fontsize=10.5,
        frameon=False  # Remove the legend box
    )

    ax1.set_title(f"{plot_title_text('cut')}")
    ax2.set_title(f"{plot_title_text('consq')}")
    ax1.set_xlabel('Cycle(s)\n')
    ax2.set_xlabel('Cycle(s)')

    ax1.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
    ax2.set_ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')

    ax1.set_xlim(0, 250)
    ax2.set_xlim(0, 250)

    ax1.set_ylim(0, 355)
    ax2.set_ylim(0, 355)

    ax1.set_yticks(range(0, 400 + 5, 50))
    ax2.set_yticks(range(0, 400 + 5, 50))


    # Adjust layout to add a 0.5 cm gap between the figures
    plt.subplots_adjust(hspace=0.6 / 2.54)  # 0.5 cm gap converted to inches

    # Save and display plot
    filename = join_path(folder, f'break_even_{case}.png')
    plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
    plt.show()

def be_figure(case, data, initialization, path):
    if '1' in case:
        be1_figure( data, case, initialization, path)
    else:
        be2_figure(data, case, initialization, path)

def create_results_figures(case, path, ecoinevnt_paths, system_path):
    _, initialization = lc.get_all_flows(path, 'recipe')
    folder = s.results_folder(join_path(path,'results'), case)
    impact_categories = lc.lcia_impact_method('recipe')
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]
    
     # Extract the endpoint categories from the plot x-axis
    plot_x_axis_end = plot_x_axis_all[-3:]
    
    # Extract the midpoint categories from the plot x-axis
    ic_mid = plot_x_axis_all[:-3]

    plot_x_axis = []
    # Process each midpoint category to create a shortened version for the plot x-axis
    for ic in ic_mid:
        string = re.findall(r'\((.*?)\)', ic)
        if 'ODPinfinite' in string[0]:
            string[0] = 'ODP'
        elif '1000' in string[0]:
            string[0] = 'GWP'
        plot_x_axis.append(string[0])
    
    data = data_set_up(path, 'recipe', ecoinevnt_paths, system_path)

    midpoint_graph(data, case, 'recipe', plot_x_axis, initialization, folder)
    endpoint_graph(data, case, 'recipe', plot_x_axis_end, initialization, folder)
    gwp_figure(data, case, path, initialization)
    be_figure(case, data, initialization, path)