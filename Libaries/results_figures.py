import matplotlib.pyplot as plt
import numpy as np
import re
import os
import bw2data as bd
import brightway2 as bw
import bw2calc as bc
import pandas as pd
from copy import deepcopy as dc

from sensitivity import treatment_quantity

import main as m

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

init = m.main(path=path,matching_database=matching_database)


def extract_func_unit():
    db = init.db
    func_unit = []
    idx_lst = []
    func_unit_keys = []
    for act in db:
        if "defined system" in act["name"]:
            idx_lst.append(act["name"])
            func_unit_keys.append(act)
    func_unit_keys.sort()
    func_unit_keys.reverse()
    idx_lst.sort()

    for key in func_unit_keys:
        func_unit.append({key : 1})
        
    return func_unit, idx_lst

def perform_LCIA():
    """
    Calculate Life Cycle Impact Assessment (LCIA) results for given processes and impact categories.

    Parameters:
    unique_process_index (list): List of unique process indices.
    func_unit (list): List of functional units.
    impact_categories (list or tuple): List or tuple of impact categories.
    file_name_unique (str): Name of the file to save results.
    sheet_name (str): Name of the sheet to save results.
    case (str): Case identifier for the calculation setup.

    Returns:
    pd.DataFrame: DataFrame containing LCIA results.
    """

    impact_categories = init.lcia_impact_method()

    # Ensure impact categories is a list
    if isinstance(impact_categories, tuple):
        impact_categories = list(impact_categories)
    func_unit, idx_lst = extract_func_unit()
    # Initialize DataFrame to store results
    df = pd.DataFrame(0, index=idx_lst, columns=impact_categories, dtype=object)

    print(f'Total amount of calculations: {len(impact_categories) * len(idx_lst)}')
    # Set up and perform the LCA calculation
    bd.calculation_setups[f'1 treatment'] = {'inv': func_unit, 'ia': impact_categories}
     
    mylca = bc.MultiLCA(f'1 treatment')
    res = mylca.results
    
    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df.iat[col, row] = val

    # Save results to file
    init.save_LCIA_results(df, init.file_name)

    return df

def obtain_LCIA_results(calc):
    
    file_name = init.file_name

    # Check if the file exists
    if os.path.isfile(file_name):
        try:
            if calc:
                df = perform_LCIA()
        
        except (ValueError, KeyError, TypeError) as e:
            print(e)
            # Perform LCIA if there is an error in importing results
            df = perform_LCIA()
    
    else:
        print(f"{file_name} do not exist, but will be created now")
        # Perform LCIA if file does not exist
        df = perform_LCIA()

    # Import LCIA results if user chooses 'n'
    if calc is False:
        df = init.import_LCIA_results(file_name, init.lcia_impact_method())
        
 
    return df

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
    for i in init.lcia_impact_method():
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

    file_path_tot = rf"{init.results_path}\LCIA\penincillium_totals.xlsx"

    init.save_LCIA_results(df_tot_T, file_path_tot)

def print_min_max_val(scaled_df):
    min_val = None
    max_val = 0
    for col in scaled_df.columns:
        for _, row in scaled_df.iterrows():
            val = row[col]
            if min_val is None or val < min_val:
                min_val = val
            elif val != 1 and val > max_val:
                max_val = val
    print(f"Mininum valúe : {min_val}, Maximum value : {max_val}")

def data_set_up(reload=False, calc=False, sensitivity=False):
    init.database_setup(reload=reload, sensitivty=sensitivity)

    # Perform quick LCIA (Life Cycle Impact Assessment) and get the results
    df = obtain_LCIA_results(calc)
    if calc:
        save_totals_to_excel(df)

    # Process the data
    df_res = init.dataframe_results_handling(df)

    if type(df_res) is list:
        df_mid, df_endpoint = df_res
    # Scale the data for midpoint and endpoint
    df_scaled_mid = init.dataframe_cell_scaling(df_mid)
    
    # if 'recipe' in lcia_method.lower():
    df_scaled_end = init.dataframe_cell_scaling(df_endpoint)

    # Extract the GWP (Global Warming Potential) column
    df_col = [df_mid.columns[1]]
    df_GWP = df_mid[df_col]
    
    print_min_max_val(df_scaled_mid)

    return [df_scaled_mid, df_scaled_end, df_GWP, df_mid, df_endpoint]

def mid_end_legend_text(df):
    leg_idx = []
    for idx in df.index:
        if "G" in idx:
            # txt = idx.replace(f", defined system", "")
            txt = "IV"
            leg_idx.append(txt)
        else:
            txt = "Oral"
            leg_idx.append(txt)

    return leg_idx


def results_normalization(calc):
    df = obtain_LCIA_results(calc)
    df_T = df.T
    idx_lst = df_T.index

    df_T = df_T.drop(idx_lst[-3:])
    idx_lst = df_T.index
    
    nf_excel_path = init.join_path(init.path_github,r"data\ReCiPe_2016_Normalization_Factors.xlsx")

    nf_df = pd.read_excel(io=nf_excel_path, index_col=0)
    nf_df.index = idx_lst

    df_T_nf = dc(df_T)
    for col in df_T.columns:
        for idx, row in df_T_nf.iterrows():
            row[col] /= nf_df.at[idx, "Value"]
            row[col] *= pow(10,3)

            if "HTPc" in str(idx[2]):
                print(f"{idx[2]} = {row[col]}")

    df_nf = df_T_nf.T

    return df_nf

def midpoint_graph(df, plot_x_axis):
    recipe = 'Midpoint (H)'
    colors = init.color_range(colorname="coolwarm", color_quantity=2)

    # Extract columns and indices for plotting
    columns_to_plot = df.columns
    index_list = list(df.index.values)

    # Create the plot
    width_in, height_in, dpi = init.plot_dimensions()
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
    output_file = init.join_path(
        init.path_github,
        f'figures\{recipe}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

def midpoint_normalized_graph(calc, plot_x_axis):
    colors = init.color_range(colorname="coolwarm", color_quantity=2)

    df = results_normalization(calc)

    # Extract columns and indices for plotting
    columns_to_plot = df.columns
    index_list = list(df.index.values)

    # Create the plot
    width_in, height_in, dpi = init.plot_dimensions()

    fig, (ax, ax2) = plt.subplots(2,1, sharex=True, figsize=(width_in, height_in), dpi=dpi)
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))
    axes = [ax, ax2]
    # Plot each group of bars
    for a in axes:
        for i, process in enumerate(df.index):
            values = df.loc[process, columns_to_plot].values
            color = colors[i % len(colors)]  # Ensure color cycling
            a.bar(index + i * bar_width, 
                values, bar_width, 
                label=process, 
                color=color,
                edgecolor="k",
                zorder=10)
    
        a.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
        a.set_xticklabels(plot_x_axis, rotation=90)
    x_pos = 0.92  
    fig.legend(
        mid_end_legend_text(df),
        loc='upper left',
        bbox_to_anchor=(0.86, x_pos),
        ncol= 1,
        fontsize=10,
        frameon=False,
    )
    
    
    # Center y-labels for both axes
    ax.set_ylabel('miliPerson equivalent per treatment', labelpad=20, va='center')
    # ax2.set_ylabel('miliPerson equivalent per treatment', labelpad=20, va='center')
    ax.yaxis.set_label_coords(-0.08, 0)
    ax.set_title("Normalization results for 1 treatment")  
    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(0.8, 7.1)  # outliers only
    ax.set_yticks(np.arange(2, 8, 1))  # Set y-ticks from 1 to 7 in steps of 1

    ax2.set_ylim(0, 0.55)  # most of the data
    ax2.set_yticks(np.arange(0, 0.6, 0.1))

    # hide the spines between ax and ax2
    ax.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # plt.axhline(0.75, linestyle='--',zorder=10)


    # Minimize the distance of the slanted lines
    d = 0.5  # smaller value for less distance
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    kwargs2 = dict(markersize=2,
                  linestyle="--", color='k', mec='k', mew=0.5, clip_on=False, alpha=0.3)
    ax2.plot([0, 1], [1.07, 1.07], transform=ax2.transAxes, **kwargs2,zorder=20)

    output_file = init.join_path(
        init.path_github,
        r'figures\normalized_results_midpoint.png'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

def extract_fu_penG_contribution():
    db = init.db
    func_unit = {}
    for act in db:
        if "defined system" in act["name"] and "G" in act["name"]:
            func_unit[act["name"]] = {}
            for exc in act.exchanges():
                if "techno" in exc["type"]:
                    func_unit[act["name"]].update({exc.input : exc["amount"]})
    return func_unit

def extract_penG_actvitites():
    
    fu_all_dct = extract_fu_penG_contribution()
    fu_all = fu_all_dct["Penicillin G, defined system"]
    fu_sep = []
    for key, item in fu_all.items():
        fu_sep.append({key : item})

    fu_sep = []
    idx_lst = []

    scaling_dct = treatment_quantity()

    for key, item in fu_all.items():
        if "glass vials" in str(key):

            fu_sep.append({key : item})
            idx_lst.append(key)
            for act in init.db:
                if "penicillium G" in str(act):
                    fu_sep.append({act : scaling_dct["manufacturing of raw penicillium G"]})
                    idx_lst.append(str(act))
        else:
            fu_sep.append({key : item})
            idx_lst.append(str(key))

    return fu_sep, idx_lst

def substract_penGprod(df_contr):
    penG_idx = None
    vial_idx = None

    for idx in df_contr.index:
        if "packaging of glass vials with penicillin G" in str(idx):
            vial_idx = idx
        elif "manufacturing of raw penicillium G" in str(idx):
            penG_idx = idx

    for col in df_contr.columns:
        df_contr.at[vial_idx, col] = df_contr.at[vial_idx, col] - df_contr.at[penG_idx, col]
    
    return df_contr

def contribution(contr_excel_path):
    func_unit, idx_lst = extract_penG_actvitites()
    calc_setup_name = str("PenG contrinbution")
    bd.calculation_setups[calc_setup_name] = {'inv': func_unit, 'ia': init.lcia_impact_method()}
    mylca = bc.MultiLCA(calc_setup_name)
    res = mylca.results
    df_contr = pd.DataFrame(0, index=idx_lst, columns=init.lcia_impact_method(), dtype=object)

    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_contr.iat[col, row] = val

    df_contr = substract_penGprod(df_contr)

    df_contr_share = dc(df_contr)

    for col in df_contr_share.columns:
        tot = df_contr_share[col].to_numpy().sum()
        for _, row in df_contr_share.iterrows():
            row[col] /= tot

    init.save_LCIA_results(df_contr_share, contr_excel_path)

    return df_contr_share

def contribution_LCIA_calc(calc):
    contr_excel_path = init.join_path(init.results_path, r"LCIA\penG_contribution.xlsx")
    if os.path.isfile(contr_excel_path):
        # user_input = input("Select y for recalculate or n to import calculated results")
        if calc:
            df_contr_share = contribution(contr_excel_path)
        else:
            df_contr_share = init.import_LCIA_results(contr_excel_path, init.lcia_impact_method())
 
    else:
        df_contr_share = contribution(contr_excel_path)

    return df_contr_share

def act_to_string_simplification(text):
    if type(text) is not str:
        text = str(text)

    if "glass vials" in text.lower():
        text = "glass vial"
    if "wipe" in text.lower():
        text = "wet wipe"
    if "gloves" in text.lower():
        text = "gloves"
    if "incineration" in text.lower():
        text = "incineration"
    if "packaging paper" in text.lower():
        text = "packaging"
    if "iv bag" in text.lower():
        text = "IV bag"
    if "stopcock" in text.lower():
        text = "stopcock"
    if "water" in text.lower():
        text = "ultrapure water"
    if "medical connector" in text.lower():
        text = "medical connector"
    if "sodium chlorate" in text.lower():
        text = "sodium chlorate"
    if "penicillium g" in text.lower():
        text = "penicillium"

    return text

def data_reorginization(df_contr_share):
    iv_liquid_row = 0
    new_idx = "IV liquid"
    df_temp = dc(df_contr_share)
    
    for idx, row in df_temp.iterrows():
        try:
            if "market" in str(idx):
                iv_liquid_row += row
                df_temp.drop([idx], inplace=True)
        except IndexError:
            print(f"keyerror for {idx}")

    df_temp.loc[new_idx] = iv_liquid_row # adding a row
    
    return df_temp

def contribution_analysis_data_sorting(calc):
    pen_comp_cat = {
        "Prod.": ["penicil", "vial"],
        "Aux. mat.": ["wipe", "glove"],
        "IV": ["stopcock", "water", "sodium", " connector", "IV"],
        "Disposal": ["waste"]
        }

    pen_cat_sorted = {}
    leg_txt = []

    df_contr_share_raw = contribution_LCIA_calc(calc)
    df_contr_share = data_reorginization(df_contr_share_raw)

    for cat, id_lst in pen_comp_cat.items():
        pen_cat_sorted[cat] = []
        for id in id_lst:
            for idx in df_contr_share.index:
                if id in str(idx) and idx not in pen_cat_sorted[cat]:
                    pen_cat_sorted[cat].append(idx)
                    txt = act_to_string_simplification(idx)
                    if txt not in leg_txt:
                        leg_txt.append(f"{cat}: {txt}")

    return df_contr_share, pen_cat_sorted, leg_txt

def lcia_categories():
    ic_idx = [1, -3, -2, -1]
    ic_plt = []
    for ic in ic_idx:
        ic_plt.append(init.lcia_impact_method()[ic])

    return ic_plt

def contribution_results_to_dct(calc):
    dct = {}
    dct_tot = {}
    ic_plt = lcia_categories()
    df_contr_share, pen_cat_sorted, leg_txt = contribution_analysis_data_sorting(calc)
    for ic in ic_plt:
        dct[ic] = {}
        dct_tot[ic] = {}
        temp_dct = {}
        
        for cat, act_lst in pen_cat_sorted.items():
            temp_dct[cat] = {}
            tot = 0
            
            for act in act_lst:
                val = df_contr_share.at[act, ic]
                tot += val
                temp_dct[cat].update({act : val})
            
        dct[ic].update(temp_dct)
    
    return dct, leg_txt

def text_for_x_axis():
    return ("GWP", "Ecosystem\n damage", "Human health\n damage", "Natural \nresources damage")

def hatch_styles():
    return ["\\\\", "OO", "++", "**", "O."]

def sort_act_in_production(dct):
    prod_dct = {}
    dct_sorted = {}
    for ic, recipe in dct.items():
        dct_sorted[ic] = {}   
        for key, item in recipe.items():
            prod_dct[key] = {}
            if key == "Prod.":
                prod_keys = list(item.keys())
                prod_keys.sort()
                for pk in prod_keys:
                    prod_dct[key].update({pk : item[pk]})
            else:
                prod_dct[key] = (item)

            dct_sorted[ic].update(prod_dct)

    return dct_sorted

def penG_contribution_plot(calc):
    output_file_contr = r"C:\Users\ruw\Desktop\RA\penicilin\figures\penG_contribution.png"
    width = 0.5
    
    dct, leg_txt = contribution_results_to_dct(calc)
    dct_sorted = sort_act_in_production(dct)
    bottom = np.zeros(len(dct.keys()))

    colors = init.color_range(colorname="coolwarm", color_quantity=len(dct.keys()))
    width_in, height_in, dpi = init.plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    for idx, dct_ in enumerate(dct_sorted.values()):
        for col_idx, item_dct in enumerate(dct_.values()):
            for hatch, (act, item) in enumerate(item_dct.items()):
                ax.bar(
                    text_for_x_axis()[idx], 
                    item, 
                    width, 
                    label=str(act), 
                    bottom=bottom[idx],
                    color=colors[col_idx],
                    edgecolor="k", 
                    hatch=hatch_styles()[hatch],
                    alpha=.9,
                    zorder=10
                )

                bottom[idx] += item

    ax.set_title("Contribution analysis for 1 IV treatment")

    leg_color, _ = fig.gca().get_legend_handles_labels()
    
    # Reverse the order of handles and labels
    leg_txt_prod = leg_txt[:2]
    leg_txt[0] = leg_txt_prod[1]
    leg_txt[1] = leg_txt_prod[0]

    leg_txt = leg_txt[::-1]
    leg_color = leg_color[::-1]

    # ax.set_xticklabels(text_for_x_axis(), rotation=0)

    ax.legend(
            leg_color,
            leg_txt,
            loc='upper left',
            bbox_to_anchor=(0.995, 1.02),
            ncol= 1,  # Adactjust the number of columns based on legend size
            fontsize=10,
            frameon=False
        )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    ax.set_ylabel("Share of the impact")
    y_ticks = plt.gca().get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in y_ticks])
    ax.set_ylim(0,1.01)
    plt.tight_layout()
    plt.savefig(output_file_contr, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def create_results_figures(reload=False, calc=False):
    # Set the current Brightway project
    bw_project = init.bw_project
    bd.projects.set_current(bw_project)

    impact_categories = init.lcia_impact_method()
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]

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
    
    # midpoint_graph(data[0], plot_x_axis_mid)
    midpoint_normalized_graph(calc, plot_x_axis_mid)
    penG_contribution_plot(calc)

    return data
