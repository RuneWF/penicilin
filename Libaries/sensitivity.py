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

# import standards as s
# import database as dm
import sensitvity_countries as stc
import sensitivity_EoL as eol

import main as m

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

init = m.main(path=path,matching_database=matching_database)


# Function to calculate treatment quantities based on scaling factors from an Excel file
def treatment_quantity():
    # Use a context manager to open the Excel file
    with pd.ExcelFile(init.system_path) as excel_file:
        # Get the sheet names
        sheet_names = excel_file.sheet_names
    
    scaling_sheet = ""
    for sheet in sheet_names:
        if "scaling" in sheet:
            scaling_sheet = sheet

    # Read the treatment quantity data from the second sheet
    df_treatment_quantity = pd.read_excel(io=init.system_path, sheet_name=scaling_sheet)
    scaling_dct = {}
    for act in init.db:
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

    for act in init.db:
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
    sens_folder = rf"{init.results_path}\results"
    save_dir = init.results_folder(sens_folder, "sensitivity")
    return save_dir

# Function to generate file paths for sensitivity results
def sensitivity_paths(pen_type, save_dir):
    if "V" in pen_type:
        sens_file_path = init.join_path(save_dir, r"penincillium_V.xlsx")
    else:
        sens_file_path = init.join_path(save_dir, r"penincillium_G.xlsx")
    
    return sens_file_path

# Function to calculate or import sensitivity results
def calculate_sensitivity_results(pencillium_fu, proc_check, calc=False):
    # Get the LCIA method and results folder
    method_GWP = init.lcia_impact_method()[1]
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
            init.save_LCIA_results(df_sens, file_path[pen_type], "results")
    else:
        # Import pre-calculated LCIA results if calc is False
        for pen_type, excel_path in file_path.items():
            temp = init.import_LCIA_results(excel_path, method_GWP)
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

    method_GWP = init.lcia_impact_method()[1]

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
     
    colors = init.color_range(colorname="coolwarm", color_quantity=5)
    colors.reverse()

    output_file_sens = rf"{init.path_github}\figures\senstivity.png"

    col_sens, _ = organize_sensitivity_scenarios()
    pen_stat_tot_sorted = sort_dict_keys(pen_stat_tot)
    width_in, height_in, dpi = init.plot_dimensions(subfigure=True)
    _, axes = plt.subplots(1, len(pen_stat_tot_sorted), figsize=(width_in*1.9, height_in), dpi=dpi)

    title_identifier = [r"$\bf{Fig\ A:}$ ", r"$\bf{Fig\ B:}$ "]

    for marker, (pen_type, df) in enumerate(pen_stat_tot_sorted.items()):
        ax = axes[marker]
        leg = []
        for idx, (_, row) in enumerate(df.iterrows()):
            for c, val in enumerate(row):
                ax.scatter(idx, val, color=colors[c], edgecolor="k", s=75, zorder=10)
            sensitivity_legend(col_sens, idx, leg)
                
        x = np.arange(len(row.to_numpy()))
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=0)
        ax.set_ylabel('grams of CO$_2$-eq per treatment')
        ax.set_title(f'{title_identifier[marker]}GWP for manufacturing of penicillin {pen_type[-1]}', loc="left")
        if pen_type[-1] == "V":
            y = np.arange(30, 43, 2)
            ax.set_yticks(y)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    # Retrieve the legend handles and labels


    ax.legend(
        leg,
        loc='upper left',
        bbox_to_anchor=(0.96, 1.02),
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(output_file_sens, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

# Function to perform Monte Carlo simulation and plot results
def monte_carlo_plot(stat_arr_dct, base=10, power=4):
    
    output_file_MC = rf"{init.path_github}\figures\monte_carlo.png"
    data_proccessing_dct = calc_mean_std(stat_arr_dct)
    data_proccessing_dct_sorted = sort_dict_keys(data_proccessing_dct)

    num_samples = pow(base, power)  # Number of samples you want to generate
    width_in, height_in, dpi = init.plot_dimensions()
    plt.figure(figsize=(width_in, height_in), dpi=dpi)
    arr_lst = []
    pen_label = ["G", "V"]
    colors = init.color_range(colorname="coolwarm", color_quantity=len(pen_label))
    for c, dct in enumerate(data_proccessing_dct_sorted.values()):
        mean = dct["Mean"]
        std_dev = dct["Standard Deviation"]

        samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        arr_lst.append(samples)
        # Plot histogram
        plt.hist(samples, bins=40, density=True, alpha=0.6, color=colors[c], edgecolor='black', label=f"Penicillin {pen_label[c]}")

    plt.xlabel('grams of CO$_2$-eq per treatment')
    plt.ylabel('Probability')
    plt.title('Monte Carlo simulation of the GWP for penicillin materials & manufacturing*')
    plt.legend(loc='upper right', frameon=False)
    

    ax = plt.gca()
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in y_ticks])
    ax.set_ylim(0,0.26)
    
    plt.tight_layout()
    plt.savefig(output_file_MC, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

    # Perform t-test to compare distributions
    # print(f"length of arr_lst : {len(arr_lst)}")
    data1 = arr_lst[0]
    data2 = arr_lst[1]
    t_stat, p_value = stats.ttest_ind(data1, data2)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

def extract_fu_penG_contribution():
    db = init.db
    func_unit = []
    idx_lst = []
    for act in db:
        if "defined system" in act["name"] and "G" in act["name"]:

            for exc in act.exchanges():
                if "techno" in exc["type"]:
                    func_unit.append({exc.input : exc["amount"]})
                    idx_lst.append(exc.input)
    return func_unit, idx_lst

def extract_penG_actvitites():
    data = init.initialization()
    fu_all = data["Penicillin G, defined system"]

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

    init.save_LCIA_results(df_contr_share, contr_excel_path, sheet_name="contribution")

    return df_contr_share

def contribution_LCIA_calc(calc):
    contr_excel_path = init.join_path(folder(), "penG_contribution.xlsx")
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
        text = "penicillium G"

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

def penG_contribution_plot(calc):
    output_file_contr = r"C:\Users\ruw\Desktop\RA\penicilin\figures\penG_contribution.png"
    width = 0.5
    
    dct, leg_txt = contribution_results_to_dct(calc)
    bottom = np.zeros(len(dct.keys()))

    colors = init.color_range(colorname="coolwarm", color_quantity=len(dct.keys()))
    width_in, height_in, dpi = init.plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
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

    ax.set_title("Contribution analysis for 1 treatment with Penicillin G")

    leg_color, _ = fig.gca().get_legend_handles_labels()
    
    # Reverse the order of handles and labels
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


def obtain_func_unit():
    func_unit_recycling = {}
    for act in init.db:
        if "Penicillin G," in act["name"]:
            func_unit_recycling[act["name"]] = []
            for exc in act.exchanges():
                if "incineration" in exc["name"] or "recycling" in exc["name"] or "avoided" in exc["name"]:
                    func_unit_recycling[act["name"]].append({exc.input : exc["amount"]})
                elif "Penicillin G," in exc["name"]:
                    func_unit_recycling[act["name"]].append({exc.input : exc["amount"]})


    fu_keys_sorted = list(func_unit_recycling.keys())
    fu_keys_sorted.sort()
    fu_keys_sorted

    func_unit_recycling_sorted = {}

    for key in fu_keys_sorted:
        func_unit_recycling_sorted[key] = func_unit_recycling[key]

    return func_unit_recycling_sorted

def obtain_results(calc):
    excel_path = init.join_path(init.results_path, r"results\sensitivity\sens_eol_penG.xlsx")
    func_unit = obtain_func_unit() 
    if calc:
        ics = init.lcia_impact_method()     
        pen_arr = []         
        func_unit = obtain_func_unit()  
        # Set up and perform the LCA calculation
        for scenario, fu in func_unit.items():
            bd.calculation_setups[str(scenario)] = {'inv': fu, 'ia': [ics[1]]}
                
            mylca = bc.MultiLCA(str(scenario))
            res = mylca.results
            pen_arr.append(res)    

        res_countries_dct = {}
        for lst_idx, (scenario, fu) in enumerate(func_unit.items()):
            res_countries_dct[scenario] = {}
            for arr_idx, dct in enumerate(fu):
                temp = {}
                for act in dct.keys():  
                    val = pen_arr[lst_idx][arr_idx][0]
                    if arr_idx == 0:
                        act_0 = act
                        val_0 = val
                        temp[act_0] = val_0
                    else:
                        val_0 -= val
                        temp[act_0] = val_0
                        temp[act] =  val
                    res_countries_dct[scenario].update(temp)
        res_countries_dct

        idx_lst = [
        "Other",
        "Incineration",
        "Recycling",
        "Avoided"
        ]

        df = pd.DataFrame(0, index=idx_lst, columns=res_countries_dct.keys(), dtype=object)

        for scenario in df.columns:
            print(scenario)
            for idx, row in df.iterrows():
                for act, val in res_countries_dct[scenario].items():
                    if idx.lower() in str(act):
                        row[scenario] = val

                    elif "Penicillin G" in str(act) and idx_lst[0] in idx:
                        row[scenario] = val


        df.index = [
        "Cradle to Hospital",
        "Incineration",
        "Recycling",
        "Avoided",
        ]

        init.save_LCIA_results(df,file_name=excel_path, sheet_name="EoL")
    else:
        df = init.import_LCIA_results(excel_path, list(func_unit.keys()))
        df.index = [
            "Cradle to Hospital",
            "Incineration",
            "Recycling",
            "Avoided",
        ]
    
       

    return df


def legend_set_up(df, fig, ax, xtick_txt):
    tot_impact = {col : df[col].to_numpy().sum() for col in df.columns} 

    for scenario, total in enumerate(tot_impact.values()):
        ax.plot(xtick_txt[scenario], total, 'D', color="k", markersize=4, mec='k', label='Net impact', zorder=9)
        # Add the data value
        ax.text(
            xtick_txt[scenario], total-0.08, f"{total:.2f}", 
            ha='center', va='bottom', fontsize=11, 
            color="k", zorder=11)

    # Custom legend with 'Total' included
    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        plt.Line2D([0], [0], marker='D', color='k', markerfacecolor="k", mec='k', markersize=4, label='Net impact')
    )

    leg_color, _ = fig.gca().get_legend_handles_labels()
    leg_txt = list(df.index)

    # Reverse the order of handles and labels
    leg_txt.append("Net impact")
    leg_txt = leg_txt[::-1]


    leg_color.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor="k", mec='k', markersize=4, label='Net impact'))
    leg_color = leg_color[::-1]

    arr = np.array(list(tot_impact.values()))

    baseline = arr[0]
    scenarios = arr[1:]


    min_reduction = scenarios.max()/baseline
    max_reduction = scenarios.min()/baseline

    print(f"Min reduction : {round(min_reduction*100,2)}%")
    print(f"Max reduction : {round(max_reduction*100,2)}%")

    return leg_txt, leg_color

def sens_EoL_plot(calc=False):
    width = 0.5
    df = obtain_results(calc)
    colors = init.color_range(colorname="coolwarm", color_quantity=len(df.index))
    width_in, height_in, dpi = init.plot_dimensions()
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    df.T.plot(
        kind='bar',
        stacked=True,
        title="GWP for different MWT for auxillary product for peniciilin G",
        color=colors,
        ax=ax,
        width=width,
        edgecolor="k",
        zorder=5
    )

    xtick_txt = [
        "Baseline",
        "Recycling IV\n+ gloves",
        "Recycling IV",
        "Recycling \ngloves",
    ]

    leg_txt, leg_color = legend_set_up(df, fig, ax, xtick_txt)

    ax.legend(
            leg_color,
            leg_txt,
            loc='upper left',
            bbox_to_anchor=(0.995, 1),
            ncol= 1,  # Adactjust the number of columns based on legend size
            fontsize=10,
            frameon=False
        )

    ax.set_ylabel('kilograms of CO$_2$-eq per treatment')
    y_ticks = np.linspace(-0.3, 1, 14)
    ax.set_yticks(y_ticks)
    ax.set_ylim(-0.3, 1.01)
    ax.set_xticklabels(xtick_txt, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    plt.tight_layout()
    plot_save_path = init.join_path(init.path_github, r"figures")
    output_file = init.join_path(plot_save_path, f"penG_EoL_sens.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()


# Main function to perform sensitivity and uncertainty analysis
def perform_sens_uncert_analysis(mc_base=10, mc_power=4, reload=False, calc=False, sensitivty=False):
    init.database_setup(reload=reload, sensitivty=sensitivty)
    pencillium_fu, proc_check = obtain_activities()
    pen_df = calculate_sensitivity_results(pencillium_fu, proc_check, calc)
    pen_compact_df, tot_df, pen_compact_idx = compacting_penicillium_dataframes(pen_df)
    stat_arr_dct, pen_stat_tot = calc_senstivity_values(pen_compact_df, pen_compact_idx, tot_df)
    sensitivity_plot(pen_stat_tot)
    monte_carlo_plot(stat_arr_dct, mc_base, mc_power)
    penG_contribution_plot(calc)


    stc.countries_sens_plot(calc)
    sens_EoL_plot(calc)


