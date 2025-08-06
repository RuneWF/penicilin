import bw2data as bd
import bw2calc as bc
import pandas as pd
import numpy as np
from copy import deepcopy as dc
import os
import matplotlib.pyplot as plt
from scipy import stats

import sensitvity_countries as stc

import main as m

init = m.main()




# Function to obtain activities and their exchanges from the database
def obtain_activities():
    pencillium_fu = {}
    proc_check = {}

    # Get scaling factors
    scaling_dct = init.treatment_quantity()
    for act in init.db:
        # Filter activities related to penicillin
        if "penicillium" in str(act['name']):
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
    save_dir = init.results_folder(init.results_path, "sensitivity")
    return save_dir

# Function to generate file paths for sensitivity results
def sensitivity_paths(pen_type, save_dir):
    if "V" in pen_type:
        sens_file_path = init.join_path(save_dir, r"penincillium_V.xlsx")
    else:
        sens_file_path = init.join_path(save_dir, r"penincillium_G.xlsx")
    
    return sens_file_path

def LCIA(pen_type, proc_check, method_GWP, pen_fu, pen_df, file_path):
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
    init.save_LCIA_results(df_sens, file_path[pen_type])


# Function to calculate or import sensitivity results
def calculate_sensitivity_results(pencillium_fu, proc_check, calc=False):
    # Get the LCIA method and results folder
    method_GWP = init.lcia_impact_method()[1]
    save_dir = folder()
    file_path = {pen_type: sensitivity_paths(pen_type, save_dir) for pen_type in pencillium_fu.keys()}
    pen_df = {}
    
    
        # Perform LCIA calculations if calc is True
    for pen_type, pen_fu in pencillium_fu.items():
        if os.path.isfile(file_path[pen_type]) and calc is False:
            for pen_type, excel_path in file_path.items():
                temp = init.import_LCIA_results(excel_path)
                temp.columns = [method_GWP[1]]  # Rename columns for consistency
                pen_df[pen_type] = temp
        elif calc:
            LCIA(pen_type, proc_check, method_GWP, pen_fu, pen_df, file_path)
        else:
            LCIA(pen_type, proc_check, method_GWP, pen_fu, pen_df, file_path)

    
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

def penicillin_G_V_to_IV_oral(pen_type):
    if "G" in pen_type:
        # txt = idx.replace(f", defined system", "")
        return "IV"
    else:
        return "Oral"

def sens_data_initialization(reload, sensitivty, calc):
    init.database_setup(reload=reload, sensitivty=sensitivty)
    pencillium_fu, proc_check = obtain_activities()
    pen_df = calculate_sensitivity_results(pencillium_fu, proc_check, calc)
    pen_compact_df, tot_df, pen_compact_idx = compacting_penicillium_dataframes(pen_df)
    stat_arr_dct, pen_stat_tot = calc_senstivity_values(pen_compact_df, pen_compact_idx, tot_df)

    return [stat_arr_dct, pen_stat_tot]

# Function to plot sensitivity analysis results
def sensitivity_plot(reload, sensitivty, calc):
     
    colors = init.color_range(colorname="coolwarm", color_quantity=5)
    colors.reverse()

    output_file_sens = rf"{init.path_github}\figures\senstivity.png"

    col_sens, _ = organize_sensitivity_scenarios()

    pen_stat_tot = sens_data_initialization(reload, sensitivty, calc)[1]

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
        ax.set_title(f'{title_identifier[marker]}GWP for manufacturing of Penicillin {pen_type[-1]}', loc="left")
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
def monte_carlo_plot(reload, sensitivty, calc, base=10, power=4):
    
    output_file_MC = rf"{init.path_github}\figures\monte_carlo.png"

    stat_arr_dct = sens_data_initialization(reload, sensitivty, calc)[0]

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
    plt.title('Monte Carlo simulation of the GWP for penicillin materials & manufacturing')

    leg_color, _ = plt.gca().get_legend_handles_labels()

    plt.legend(
        leg_color,
        ["IV", "Oral"],
        loc='upper right', 
        frameon=False)
    

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

def LCIA_eol(excel_path):
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

    init.save_LCIA_results(df,file_name=excel_path)

    return df 

def obtain_results(calc):
    save_dir = init.results_folder(init.results_path,"sensitivity")
    excel_path = init.join_path(save_dir, r"sens_eol_penG.xlsx")
    func_unit = obtain_func_unit() 
    if os.path.isfile(excel_path):
        if calc:
            df = LCIA_eol(excel_path) 
        else:
            df = init.import_LCIA_results(excel_path)
            df.index = [
                "Cradle to Hospital",
                "Incineration",
                "Recycling",
                "Avoided",
            ]
    else:
        df = LCIA_eol(excel_path)

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
        title="GWP for different MWT for auxillary product for IV treatment",
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

def clinical_treatment_initialization():
    df = init.import_LCIA_results(init.LCIA_results)
    ic_gwp = init.lcia_impact_method()[1]
    df_gwp = df[ic_gwp]
    idx_gwp = list(df_gwp.index)

    idx_lst = [
    "IV",
    "Oral",
    "Combined"
    ]

    df_pneumonia = pd.DataFrame(0, index=idx_lst, columns=[ic_gwp], dtype=object)

    days = 5
    daily_adminstrations = 4
    adminstrations = days * daily_adminstrations

    days_com_dct = {
        idx_lst[0] : 2,
        idx_lst[1] : 3
    }


    for idx, row in df_pneumonia.iterrows():
        if "IV" in idx:
            row[ic_gwp] = df_gwp.at[idx_gwp[0]] * adminstrations
        elif "Oral" in idx:
            row[ic_gwp] = df_gwp.at[idx_gwp[1]] * adminstrations
        else:
            dct = {}
            for x, (key, item) in enumerate(days_com_dct.items()):
                dct[key] = df_gwp.at[idx_gwp[x]] * item * daily_adminstrations
            row[ic_gwp] = dct

    iv_impact = df_pneumonia.at[idx_lst[0], ic_gwp]
    oral_impact = df_pneumonia.at[idx_lst[1], ic_gwp]
    comb_impact_dct  = df_pneumonia.at[idx_lst[2], ic_gwp]
    comb_impact_lst = np.array(list(comb_impact_dct.values()))
    comb_impact = comb_impact_lst.sum()

    print(f"{idx_lst[1]} saves {round((1-oral_impact/iv_impact)*100,0)}%")
    print(f"{idx_lst[2]} saves {round((1-comb_impact/iv_impact)*100,0)}%")

    return idx_lst, df_pneumonia

def clinical_treatment_plot():
    idx_lst, df_pneumonia = clinical_treatment_initialization()

    # Create the plot
    colors = init.color_range(colorname="coolwarm", color_quantity=2)
    width_in, height_in, dpi = init.plot_dimensions()
    _, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    bar_width = 1 / (len(idx_lst) + 1)

    # Plot individual bars for the first two scenarios
    for i in range(2):
        ax.bar(i, df_pneumonia.iloc[i, 0], bar_width, color=colors[i], edgecolor='k', label=df_pneumonia.index[i], zorder=2)

    # Plot stacked bar for the third scenario
    stack_data = df_pneumonia.iloc[2, 0]
    bottom = 0
    for j, value in enumerate(stack_data.values()):
        ax.bar(2, value, bar_width, bottom=bottom, color=colors[j], edgecolor='k', label=idx_lst[j],  zorder=2)
        bottom += value

    # Customize the plot
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(df_pneumonia.index)
    ax.set_ylabel('kilograms of CO$_2$-eq per treatment')
    ax.set_title("GWP for clinical treatment scenarios of pneumonia")
    y_ticks = np.linspace(0, 20, 11)
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, 20.01)
    ax.legend(
        labels=idx_lst[:2], 
        frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    plt.tight_layout()
    plot_save_path = init.join_path(init.path_github, r"figures")
    output_file = init.join_path(plot_save_path, f"treatment_of_pneumonia.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

        

# Main function to perform sensitivity and uncertainty analysis
def perform_sens_uncert_analysis(mc_base=10, mc_power=4, reload=False, calc=False, sensitivty=False):

    sensitivity_plot(reload, sensitivty, calc)
    monte_carlo_plot(reload, sensitivty, calc, mc_base, mc_power)
    sens_EoL_plot(calc)
    stc.countries_sens_plot(calc, sensitivty)
    clinical_treatment_plot()
    


