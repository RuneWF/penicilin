import bw2data as bd
import bw2calc as bc
import pandas as pd
import numpy as np
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import seaborn as sns

import standards as s
import database_manipulation as dm

from lca import LCA

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

lca_init = LCA(path=path,matching_database=matching_database)

# Function to calculate treatment quantities based on scaling factors from an Excel file
def treatment_quantity():
    # Use a context manager to open the Excel file
    with pd.ExcelFile(lca_init.system_path) as excel_file:
        # Get the sheet names
        sheet_names = excel_file.sheet_names
    
    scaling_sheet = ""
    for sheet in sheet_names:
        if "scaling" in sheet:
            scaling_sheet = sheet

    # Read the treatment quantity data from the second sheet
    df_treatment_quantity = pd.read_excel(io=lca_init.system_path, sheet_name=scaling_sheet)
    scaling_dct = {}
    for act in lca_init.db:
        # Filter activities related to penicillin
        if "filling of glass vial" in act['name'] or "tablet" in act['name']:
            for exc in act.exchanges():
                # Match exchanges with penicillin types
                if "penicillium " in exc.input["name"]:
                    for pen_type in df_treatment_quantity.columns:
                        for _, row in df_treatment_quantity.iterrows():
                            if pen_type in exc["name"]:
                                # Calculate scaling factors
                                scaling_dct[exc["name"]] = exc["amount"] * row[pen_type]

    return scaling_dct

# Function to obtain activities and their exchanges from the database
def obtain_activities():
    pencillium_fu = {}
    proc_check = {}

    # Get scaling factors
    scaling_dct = treatment_quantity()

    for act in lca_init.db:
        # Filter activities related to penicillin
        if "penicillium" in act['name']:
            pencillium_fu[act["name"]] = []
            proc_check[act["name"]] = []
            for exc in act.exchanges():
                # Process only technological exchanges
                if "techno" in exc["type"]:
                    try:
                        pencillium_fu[act["name"]].append({exc.input: (exc["amount"] * scaling_dct[act["name"]]) * 1000})
                        proc_check[act["name"]].append(exc.input["name"])
                    except KeyError as e:
                        print(f"Keyerror for {e}")

    return pencillium_fu, proc_check

# Function to create and return the results folder path
def folder():
    sens_folder = rf"{lca_init.path_github}\results"
    save_dir = s.results_folder(sens_folder, "sensitivity")
    return save_dir

# Function to generate file paths for sensitivity results
def sensitivity_paths(pen_type, save_dir):
    if "V" in pen_type:
        sens_file_path = s.join_path(save_dir, r"penincillium_V.xlsx")
    else:
        sens_file_path = s.join_path(save_dir, r"penincillium_G.xlsx")
    
    return sens_file_path

# Function to calculate or import sensitivity results
def calculate_sensitivity_results(pencillium_fu, proc_check, calc=False):
    # Get the LCIA method and results folder
    method_GWP = lca_init.lcia_impact_method()[1]
    save_dir = folder()
    file_path = {pen_type: sensitivity_paths(pen_type, save_dir) for pen_type in pencillium_fu.keys()}
    pen_df = {}

    if calc:
        # Perform LCIA calculations if calc is True
        for pen_type, pen_fu in pencillium_fu.items():
            print(f"Performing LCIA for {pen_type}")

            # Initialize a DataFrame to store sensitivity results
            df_sens = pd.DataFrame(0, index=proc_check[pen_type], columns=[method_GWP[1]], dtype=object)

            # Set up the calculation environment for sensitivity analysis
            bd.calculation_setups['Sensitivity'] = {'inv': pen_fu, 'ia': [method_GWP]}

            # Perform the MultiLCA calculation
            sens_res = bc.MultiLCA('Sensitivity')
            sens_results_array = sens_res.results
            
            # Store results in the DataFrame
            for idx, res in enumerate(sens_results_array):
                df_sens.iat[idx, 0] = res

            # Save the results to the dictionary and export to an Excel file
            pen_df[pen_type] = df_sens
            s.save_LCIA_results(df_sens, file_path[pen_type], "results")
    else:
        # Import pre-calculated LCIA results if calc is False
        for pen_type, excel_path in file_path.items():
            temp = s.import_LCIA_results(excel_path, method_GWP)
            temp.columns = [method_GWP[1]]  # Rename columns for consistency
            pen_df[pen_type] = temp
    
    return pen_df

# Function to compact dataframes by grouping exchanges into categories
def compacting_penicillium_dataframes(pen_df):
    pen_compact_idx = {
        "Fermentation": ["pharmamedia", "phenyl", "phenoxy", "glucose", "oxygen", "tap water", "sulfate"],
        "Extracion": ["butyl", "sulfuric"],
        "Purification": ["sodium", "acetone"],
        "Energy": ["heat", "electricity"],
        "Waste": ["incineration", "penicillium"]
    }

    method_GWP = lca_init.lcia_impact_method()[1]

    pen_compact_df = {}
    tot_df = {}
    for pen_type, df in pen_df.items():
        df_temp = pd.DataFrame(0, index=list(pen_compact_idx.keys()), columns=[method_GWP[1]], dtype=object)
        tot = 0
        idx_lst = list(df.index)
        for idx, row in df.iterrows():
            for comp_idx, comp_lst in pen_compact_idx.items():
                for i in comp_lst:
                    if i.lower() in idx.lower():
                        df_temp.at[comp_idx, method_GWP[1]] += row[method_GWP[1]]
                        tot += row[method_GWP[1]]
                        try:
                            idx_lst.remove(idx)
                        except ValueError as e:
                            print(f"Value error for {idx} : {pen_type}")

        pen_compact_df[pen_type] = df_temp
        tot_df[pen_type] = tot

    return pen_compact_df, tot_df, pen_compact_idx

# Function to organize sensitivity scenarios
def organize_sensitivity_scenarios():
    col_sens = ["-25%", "-10%", "0%", "10%", "25%"]
    col_sens.reverse()
    sens_lst = [0.75, 0.9, 1, 1.1, 1.25]
    sens_lst.reverse()
    
    return col_sens, sens_lst

# Function to calculate sensitivity dataframes for penicillin
def calc_penicillium_sensitivity(pen_compact_df, pen_compact_idx):
    pen_sens = {}

    col_sens, sens_lst = organize_sensitivity_scenarios()

    for pen_type, df in pen_compact_df.items():
        df_sens_extr = pd.DataFrame(0, index=list(pen_compact_idx.keys()), columns=col_sens, dtype=object)
        for col in range(len(col_sens)):
            for idx in range(len(df_sens_extr.index)):
                df_sens_extr.iat[idx, col] = df.iat[idx, 0] * sens_lst[col]
        pen_sens[pen_type] = df_sens_extr

    return pen_sens

# Function to calculate sensitivity values and total impacts
def calc_senstivity_values(pen_compact_df, pen_compact_idx, tot_df):
    pen_stat_tot = {}
    col_sens, _ = organize_sensitivity_scenarios()
    pen_sens = calc_penicillium_sensitivity(pen_compact_df, pen_compact_idx)

    for pen_type, df in pen_sens.items():
        pen_stat_tot[pen_type] = None
        sens_idx = list(pen_compact_idx.keys())
        df_sens_res = pd.DataFrame(0, index=sens_idx, columns=col_sens, dtype=object)
        
        for idx in range(len(df.index)):
            for col in range(len(col_sens)):
                if col != 2: # Assigning the new total impact for the x% change column
                    new_impact_val = df.iat[idx, col] - df.iat[idx, 2]
                    new_total_impact_val = tot_df[pen_type] + new_impact_val
                    df_sens_res.iat[idx, col] = new_total_impact_val

                else: # Assigning the original total impact for the 0% change column
                    df_sens_res.iat[idx, col] = tot_df[pen_type]

        pen_stat_tot[pen_type] = (df_sens_res)
        
    stat_arr_dct = {}
    for pen, df in pen_stat_tot.items():
        arr_plc = 0
        arr_temp = np.zeros(df.size)
        for col in df.columns:
            for _, row in df.iterrows():
                arr_temp[arr_plc] = row[col]
                arr_plc += 1
        stat_arr_dct[pen] = arr_temp
    
    return stat_arr_dct, pen_stat_tot

# Function to calculate mean and standard deviation for sensitivity analysis
def calc_mean_std(stat_arr_dct):
    data_proccessing_idx = ['Mean', 'Standard Deviation']
    data_proccessing_dct = {} 

    for act, arr in stat_arr_dct.items():
        data_proccessing_dct[act] = {}
        for dp in data_proccessing_idx:
            if 'mean' in dp.lower():
                data_proccessing_dct[act].update({dp: np.mean(arr)})
            elif 'standard' in dp.lower():
                data_proccessing_dct[act].update({dp: np.std(arr)})

    return data_proccessing_dct

# Helper function to format titles
def title_text(txt):
    txt = txt.replace("manufacturing of raw ", "")
    txt = txt + " prod."
    return txt

# Function to sort dictionary keys
def sort_dict_keys(dct):
    keys_sorted = list(dct.keys())
    keys_sorted.sort()

    dct_sorted = {}
    
    for key in keys_sorted:
        dct_sorted[key] = dct[key]

    return dct_sorted

# Function to set font sizes for plots
def figure_font_sizes():
    plt.rcParams.update({
        'font.size': 12,      # General font size
        'axes.titlesize': 14, # Title font size
        'axes.labelsize': 12, # Axis labels font size
        'legend.fontsize': 10 # Legend font size
    }) 

# Helper function to create sensitivity legend
def sensitivity_legend(col_sens, idx, leg):
    if "-" in col_sens[idx]:
        leg.append(col_sens[idx].replace("-", "") + " decrease")
    elif "10" in col_sens[idx] or "25" in col_sens[idx]:
        leg.append(col_sens[idx] + " increase")
    else:
        leg.append("Baseline")

# Function to plot sensitivity analysis results
def sensitivity_plot(pen_stat_tot):
    figure_font_sizes()
    colors = s.color_range(colorname="coolwarm", color_quantity=5)
    colors.reverse()

    output_file_sens = r"C:\Users\ruw\Desktop\RA\penicilin\results\figures\senstivity.png"

    col_sens, _ = organize_sensitivity_scenarios()
    pen_stat_tot_sorted = sort_dict_keys(pen_stat_tot)

    _, axes = plt.subplots(1, len(pen_stat_tot_sorted), figsize=(15, 6))

    for fplc, (pen_type, df) in enumerate(pen_stat_tot_sorted.items()):
        ax = axes[fplc]
        leg = []
        for idx, (_, row) in enumerate(df.iterrows()):
            for c, val in enumerate(row):
                ax.scatter(idx, val, color=colors[c], edgecolor="k", s=75, zorder=10)
            sensitivity_legend(col_sens, idx, leg)
                
        x = np.arange(len(row.to_numpy()))
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=0)
        ax.set_ylabel('grams of CO$_2$-eq per FU')
        ax.set_title(f'Sensitivity analysis of manufacturing process - Penicillin {pen_type[-1]}')
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    # Retrieve the legend handles and labels


    ax.legend(
        leg,
        loc='upper left',
        bbox_to_anchor=(0.99, 1.01),
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(output_file_sens, dpi=300, format='png', bbox_inches='tight')
    plt.show()

# Function to perform Monte Carlo simulation and plot results
def monte_carlo_plot(stat_arr_dct, base=10, power=4):
    figure_font_sizes()
    
    output_file_MC = r"C:\Users\ruw\Desktop\RA\penicilin\results\figures\monte_carlo.png"
    data_proccessing_dct = calc_mean_std(stat_arr_dct)
    data_proccessing_dct_sorted = sort_dict_keys(data_proccessing_dct)

    num_samples = pow(base, power)  # Number of samples you want to generate
    plt.figure(figsize=(9, 5))
    hatch_style = ["oo", "////"]

    arr_lst = []
    colors = s.color_range(colorname="coolwarm", color_quantity=len(hatch_style))
    for c, (pen, dct) in enumerate(data_proccessing_dct_sorted.items()):
        mean = dct["Mean"]
        std_dev = dct["Standard Deviation"]

        samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        arr_lst.append(samples)
        # Plot histogram
        plt.hist(samples, bins=40, color=colors[c], hatch=hatch_style[c], edgecolor="k", alpha=0.65, label=title_text(pen))

    plt.xlabel('grams of CO$_2$-eq per FU')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo simulation of the penicillin manufacturing')
    plt.legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_file_MC, dpi=300, format='png', bbox_inches='tight')
    plt.show()

    # Perform t-test to compare distributions
    # print(f"length of arr_lst : {len(arr_lst)}")
    data1 = arr_lst[0]
    data2 = arr_lst[1]
    print(f"type of arr_lst : {type(arr_lst[0])}")
    t_stat, p_value = stats.ttest_ind(data1, data2)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

def extract_penG_actvitites():
    data = lca_init.LCA_initialization()
    fu_all = data["Penicillin G, defined system"]

    fu_sep = []
    for key, item in fu_all.items():
        fu_sep.append({key : item})
    fu_sep

    fu_all = data["Penicillin G, defined system"]

    fu_sep = []
    idx_lst = []

    scaling_dct = treatment_quantity()

    for key, item in fu_all.items():
        if "glass vials" in str(key):

            fu_sep.append({key : item})
            idx_lst.append(key)
            for act in lca_init.db:
                if "penicillium G" in str(act):
                    fu_sep.append({act : scaling_dct["manufacturing of raw penicillium G"]})
                    idx_lst.append(act)
        else:
            fu_sep.append({key : item})
            idx_lst.append(key)

    return fu_sep, idx_lst

def contribution(contr_excel_path):
    fu_sep, idx_lst = extract_penG_actvitites()
    calc_setup_name = str("PenG contrinbution")
    bd.calculation_setups[calc_setup_name] = {'inv': fu_sep, 'ia': lca_init.lcia_impact_method()}
    mylca = bc.MultiLCA(calc_setup_name)
    res = mylca.results
    df_contr = pd.DataFrame(0, index=idx_lst, columns=lca_init.lcia_impact_method(), dtype=object)

    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_contr.iat[col, row] = val
            
    df_contr_share = dc(df_contr)
    for col in df_contr_share.columns:
        tot = df_contr_share[col].to_numpy().sum()
        for _, row in df_contr_share.iterrows():
            row[col] /= tot

    s.save_LCIA_results(df_contr_share, contr_excel_path, sheet_name="contribution")

    return df_contr_share

def contribution_LCIA_calc(calc):
    contr_excel_path = s.join_path(folder(), "penG_contribution.xlsx")
    if os.path.isfile(contr_excel_path):
        # user_input = input("Select y for recalculate or n to import calculated results")
        if calc:
            df_contr_share = contribution(contr_excel_path)
        else:
            df_contr_share = s.import_LCIA_results(contr_excel_path, lca_init.lcia_impact_method())
 
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
        text = "packaging paper"
    if "set" in text.lower():
        text = "IV set"
    if "stopcock" in text.lower():
        text = "stopcock"
    if "water" in text.lower():
        text = "ultrapure water"
    if "medical connector" in text.lower():
        text = "medical connector"
    if "sodium chlorate" in text.lower():
        text = "sodium chlorate"
    if "penicillium g" in text.lower():
        text = "penicillium G"

    return text

def data_reorginization(df_contr_share):
    iv_liquid_row = 0
    new_idx = "IV liquid"

    for idx, row in df_contr_share.iterrows():
        if "market" in str(idx):
            iv_liquid_row += row
            df_contr_share.drop([idx], inplace=True)

    df_contr_share.loc[new_idx] = iv_liquid_row # adding a row
    # df_contr_share.index = df_contr_share.index + new_idx # shifting index

    # iv_liquid_row.to_dict()
    
    return df_contr_share

def contribution_analysis_data_sorting(calc):
    pen_comp_cat = {
        "Manufacturing": ["penicil", "vial"],
        "Auxilary product": ["wipe", "glove"],
        "IV": ["stopcock", "water", "sodium", " connector", "IV"],
        "Disposal": ["waste"]
        }

    pen_cat_sorted = {}
    leg_txt = []

    df_contr_share = contribution_LCIA_calc(calc)
    df_contr_share = data_reorginization(df_contr_share)

    for cat, id_lst in pen_comp_cat.items():
        pen_cat_sorted[cat] = []
        for id in id_lst:
            for idx in df_contr_share.index:
                if id in str(idx) and idx not in pen_cat_sorted[cat]:
                    pen_cat_sorted[cat].append(idx)
                    txt = act_to_string_simplification(idx)
                    if txt not in leg_txt:
                        leg_txt.append(f"{cat} : {txt}")
    return df_contr_share, pen_cat_sorted, leg_txt

def lcia_categories():
    ic_idx = [1, -3, -2, -1]
    ic_plt = []
    for ic in ic_idx:
        ic_plt.append(lca_init.lcia_impact_method()[ic])

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
    return ("GWP", "Ecosystem\n damage", "Human health\n damage", "Natural resources\n damage")

def hatch_styles():
    return ["\\\\", "OO", "++", "**", "O."]

def penG_contribution_plot(calc):
    figure_font_sizes()
    output_file_contr = r"C:\Users\ruw\Desktop\RA\penicilin\results\figures\penG_contribution.png"
    width = 0.5
    fig, ax = plt.subplots(figsize=(9, 5))
    dct, leg_txt = contribution_results_to_dct(calc)
    bottom = np.zeros(len(dct.keys()))

    colors = s.color_range(colorname="coolwarm", color_quantity=len(dct.keys()))

    for idx, dct_ in enumerate(dct.values()):
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

    ax.set_title("Contribution analysis for Penicillin G")

    leg_color, _ = fig.gca().get_legend_handles_labels()
    
    # Reverse the order of handles and labels
    leg_txt = leg_txt[::-1]
    leg_color = leg_color[::-1]


    ax.legend(
            leg_color,
            leg_txt,
            loc='upper left',
            bbox_to_anchor=(0.995, 1),
            ncol= 1,  # Adactjust the number of columns based on legend size
            fontsize=10,
            frameon=False
        )
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    ax.set_ylabel("Share of the impact")
    plt.tight_layout()
    plt.savefig(output_file_contr, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def func_unit_countries_sens():
    system_path = lca_init.system_path
    dm.import_databases(sensitivty=True)
    sheets_to_import = dm.extract_excel_sheets()
    func_unit = {}
    # Check if the database is case1
    for sheet in sheets_to_import:
        data = pd.read_excel(system_path, sheet_name=sheet)
        db_name = data.columns[1]
        db = bd.Database(db_name)
        func_unit[db_name] = []
        for act in db:
            temp = act['name']
            # Check if the flow is valid and add to the flow list
            if "defined" in temp:
                func_unit[db_name].append({act : 1})

    return func_unit

def sort_countries_func_unit():
    func_unit = func_unit_countries_sens()
    sorted_fu_keys = list(func_unit.keys())
    sorted_fu_keys.sort()
    sorted_fu_keys

    sorted_act = ["G", "V"]
    sorted_func_unit = {}

    for country in sorted_fu_keys:
        sorted_func_unit[country] = []
        for act in sorted_act:
            for fu in func_unit[country]:
                if act in str(fu.keys()):
                    sorted_func_unit[country].append(fu)

    return sorted_func_unit

def countries_calc(excel_path):
    impact_cat = lca_init.lcia_impact_method()
    impact_cat_GWP = impact_cat[1]
    sorted_func_unit = sort_countries_func_unit()
    res_arr = {}
    for country, fu in sorted_func_unit.items():
        print(f"Performing sensitivity for {country}")
        bd.calculation_setups[f'sensitivity_countries'] = {'inv': fu, 'ia': [impact_cat_GWP]}
        mylca = bc.MultiLCA(f'sensitivity_countries')
        res = mylca.results

        res_arr[country] = res
    
    df = pd.DataFrame(0, index=["Pen G", "Pen V"], columns=list(res_arr.keys()), dtype=object)

    for row, (country, arr) in enumerate(res_arr.items()):
        for idx, val in enumerate(arr):
            df.iloc[idx, row] = val[0]

    s.save_LCIA_results(df.T, excel_path, "sensitivity")

    return df.T, sorted_func_unit

def countries_LCIA_sens_calc(calc):
    # Set up and perform the LCA calculation
    excel_path = s.join_path(folder(), "countries_sensitvity.xlsx")

    if os.path.isfile(excel_path):
        if calc:
            df_sens, sorted_func_unit = countries_calc(excel_path)
        else:
            df_sens = s.import_LCIA_results(excel_path, lca_init.lcia_impact_method())
            sorted_func_unit = sort_countries_func_unit()
    else:
        df_sens, sorted_func_unit = countries_calc(excel_path)

    return df_sens, sorted_func_unit

def import_image_markers():
    img_dct = {
    "IN" : mpimg.imread(rf'{lca_init.path_github}\india.jpg'),
    "CN" : mpimg.imread(rf'{lca_init.path_github}\china.jpg'),
    "US" : mpimg.imread(rf'{lca_init.path_github}\usa.jpg'),
    "IT" : mpimg.imread(rf'{lca_init.path_github}\italy.jpg'),
    "CH" : mpimg.imread(rf'{lca_init.path_github}\switzerland.jpg')}

    return img_dct

def x_y_axis_text(sorted_func_unit):
    y_label = "grams of CO$_2$-eq per FU"

    x_tick_label = []
    
    for key in sorted_func_unit.keys():
        x_tick_label.append(key[-2:])

    return y_label, x_tick_label

def countries_penG_sens_plot(df, sorted_func_unit):
    figure_font_sizes()
    output_file_countries_sens = r"C:\Users\ruw\Desktop\RA\penicilin\results\figures\penG_countries_sens.png"
    
    penG = df["Pen G"].to_dict()

    y_label, x_tick_label = x_y_axis_text(sorted_func_unit)

    # Create some data
    data_G = {'': list(penG.keys()), f'kilo{y_label}': list(penG.values())}
    df_G = pd.DataFrame(data_G)

    # Create the scatter plot
    _, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_G, x='', y=f'kilo{y_label}', ax=ax, s=20)

    # Add custom markers
    for (xi, yi) in zip(df_G[''], df_G[f'kilo{y_label}']):
        for c, i in import_image_markers().items():
            if c in xi:
                imagebox = OffsetImage(i, zoom=0.025)
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
        ax.add_artist(ab)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove x-ticks
    ax.set_xticks(range(len(x_tick_label)))
    ax.set_xticklabels(x_tick_label)
    ax.set_ylim(0, 1.2)
    plt.title("Penicillin G - Differet location of production site")
    
    plt.tight_layout()
    plt.savefig(output_file_countries_sens, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def countries_penV_sens_plot(df, sorted_func_unit):
    figure_font_sizes()
    output_file_countries_sens = r"C:\Users\ruw\Desktop\RA\penicilin\results\figures\penV_countries_sens.png"
    penV = df["Pen V"].to_dict()

    for key, val in penV.items():
        penV[key] = val* 1000
    # Create some data

    y_label, x_tick_label = x_y_axis_text(sorted_func_unit)

    data_V = {'': list(penV.keys()), y_label: list(penV.values())}
    df_V = pd.DataFrame(data_V)

    # Create the scatter plot
    _, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_V, x='', y=y_label, ax=ax, s=20)

    # Add custom markers
    for (xi, yi) in zip(df_V[''], df_V[y_label]):
        for c, i in import_image_markers().items():
            if c in xi:
                imagebox = OffsetImage(i, zoom=0.025)
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
        ax.add_artist(ab)

    # Add grid lines for the y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove x-ticks
    ax.set_xticks(range(len(x_tick_label)))
    ax.set_xticklabels(x_tick_label)
    ax.set_ylim(0, 50)
    plt.title("Penicillin V - Differet location of production site")

    plt.tight_layout()
    plt.savefig(output_file_countries_sens, dpi=300, format='png', bbox_inches='tight')
    plt.show()

# Main function to perform sensitivity and uncertainty analysis
def perform_sens_uncert_analysis(mc_base=10, mc_power=4, calc=False):

    pencillium_fu, proc_check = obtain_activities()
    pen_df = calculate_sensitivity_results(pencillium_fu, proc_check, calc)
    pen_compact_df, tot_df, pen_compact_idx = compacting_penicillium_dataframes(pen_df)
    stat_arr_dct, pen_stat_tot = calc_senstivity_values(pen_compact_df, pen_compact_idx, tot_df)
    sensitivity_plot(pen_stat_tot)
    monte_carlo_plot(stat_arr_dct, mc_base, mc_power)
    penG_contribution_plot(calc)

    df_sens, sorted_func_unit = countries_LCIA_sens_calc(calc)

    countries_penG_sens_plot(df_sens, sorted_func_unit)
    countries_penV_sens_plot(df_sens, sorted_func_unit)

