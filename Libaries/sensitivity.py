import bw2data as bd
import bw2calc as bc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import standards as s
import life_cycle_assessment as lc

# Function to calculate treatment quantities based on scaling factors from an Excel file
def treatment_quantity(system_path, db):
    # Use a context manager to open the Excel file
    with pd.ExcelFile(system_path) as excel_file:
        # Get the sheet names
        sheet_names = excel_file.sheet_names

    # Read the treatment quantity data from the second sheet
    df_treatment_quantity = pd.read_excel(io=system_path, sheet_name=sheet_names[1])
    scaling_dct = {}
    for act in db:
        # Filter activities related to penicillin
        if "vial for penicillin" in act['name'] or "tablet" in act['name']:
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
def obtain_activities(system_path, db):
    pencillium_fu = {}
    proc_check = {}

    # Get scaling factors
    scaling_dct = treatment_quantity(system_path, db)

    for act in db:
        # Filter activities related to penicillin
        if "penicillium" in act['name']:
            pencillium_fu[act["name"]] = []
            proc_check[act["name"]] = []
            for exc in act.exchanges():
                # Process only technological exchanges
                if "techno" in exc["type"]:
                    pencillium_fu[act["name"]].append({exc.input: (exc["amount"] * scaling_dct[act["name"]]) * 1000})
                    proc_check[act["name"]].append(exc.input["name"])

    return pencillium_fu, proc_check

# Function to create and return the results folder path
def folder():
    sens_folder = r"C:\Users\ruw\Desktop\RA\penicilin\results"
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
    method_GWP = lc.lcia_impact_method()[1]
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

    method_GWP = lc.lcia_impact_method()[1]

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
    colors = s.color_range(colorname="Greys", color_quantity=5)
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
                ax.scatter(idx, val, color=colors[c], edgecolor="k", s=75)
            sensitivity_legend(col_sens, idx, leg)
                
        x = np.arange(len(row.to_numpy()))
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=0)
        ax.set_ylabel('g CO$_2$e')
        ax.set_title(f'Sensitivity analysis for {pen_type}')


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
    plt.figure(figsize=(10, 5))
    hatch_style = ["oo", "////"]

    arr_lst = []
    for c, (pen, dct) in enumerate(data_proccessing_dct_sorted.items()):
        mean = dct["Mean"]
        std_dev = dct["Standard Deviation"]

        samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        arr_lst.append(samples)
        # Plot histogram
        plt.hist(samples, bins=40, color="w", hatch=hatch_style[c], edgecolor="k", alpha=0.7, label=title_text(pen))

    plt.xlabel('g CO$_2$e')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Simulation Results for the manufacturing')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file_MC, dpi=300, format='png', bbox_inches='tight')
    plt.show()

    # Perform t-test to compare distributions
    data1 = arr_lst[0]
    data2 = arr_lst[1]
    t_stat, p_value = stats.ttest_ind(data1, data2)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Main function to perform sensitivity and uncertainty analysis
def perform_sens_uncert_analysis(system_path, db, mc_base=10, mc_power=4, calc=False):
    pencillium_fu, proc_check = obtain_activities(system_path, db)
    pen_df = calculate_sensitivity_results(pencillium_fu, proc_check, calc)
    pen_compact_df, tot_df, pen_compact_idx = compacting_penicillium_dataframes(pen_df)
    stat_arr_dct, pen_stat_tot = calc_senstivity_values(pen_compact_df, pen_compact_idx, tot_df)
    sensitivity_plot(pen_stat_tot)
    monte_carlo_plot(stat_arr_dct, mc_base, mc_power)

