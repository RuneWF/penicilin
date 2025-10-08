from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re 
from pathlib import Path


import main as m
init = m.main()

def penicillin_cf_factors_file_path():
    return Path.joinpath(init.path_github, "data", r"penicillin_impact_factors.xlsx")

def get_cf_excel_sheet_names():
    with pd.ExcelFile(penicillin_cf_factors_file_path()) as excel_file:
            # Get the sheet names
            sheet_names = excel_file.sheet_names

    return sheet_names

def get_baseline_environmental_impact():
    impact = pd.read_excel(init.LCIA_results, index_col=0)
    impact.columns = [i[2] for i in init.lcia_impact_method()]
    impact = impact[[col for col in impact.columns if any(c in col for c in get_cf_excel_sheet_names())]]

    htpc = impact.columns[-2]
    htpcn = impact.columns[-1]

    # # Add col1 and col2 into a new column
    impact["Total " + re.sub(" \(.*\)","",htpcn)] = impact[htpc] + impact[htpc]
    impact = impact.drop(columns=[htpc, htpcn])
    impact.columns = [re.sub(" \(.*\)","",col).title() for col in impact.columns]

    return impact

def max_penicillin_cf():
    max_dct = {}
    impact = get_baseline_environmental_impact()
    for ic, sheet in enumerate(get_cf_excel_sheet_names()):
        key = impact.columns[ic]
        max_dct[key] = {}
        df = pd.read_excel(penicillin_cf_factors_file_path(), skiprows=1, sheet_name=sheet)        

        idx = [pen for pen in df["API"]]

        cols = [col for col in df.columns if "water" in col or "soil" in col]

        df = df[cols]
        df.index = idx

        # Convert all values to numeric, set errors='coerce' to turn 'n.a' into NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        for idx, row in df.iterrows():
            if row.notna().any():
                max_col = row.idxmax()          # column name with the (first) max
                max_val = row.max()
                max_dct[key].update({idx : {max_col: max_val}})
            else:
                max_dct[key].update({idx : {}})

    return max_dct, impact

def get_penicillin_cf_impact():
    penicillin_amount = {
            "G" : 600*10e-6,
            "V" : 660*10e-6
        }
    
    emissions_dct = {}
    emissions_lst = []

    max_dct, impact = max_penicillin_cf()

    for pen, amount in penicillin_amount.items():
        penicillin = [idx for idx in impact.index if pen in idx][0]
        emissions_dct[penicillin] = {}
        
        for ic, pen_dct in max_dct.items():
            temp = {}
            for pen, dct in pen_dct.items():
                temp[pen] = {}
                for emission, cf in dct.items():
                    temp[pen].update({emission.replace("  ", " ") : amount*cf + impact.loc[penicillin, ic]})
                    if emission not in emissions_lst:
                        emissions_lst.append(emission)
                emissions_dct[penicillin].update({ic : temp})

    return emissions_dct, emissions_lst

def get_emission_type_marker(emissions_lst):
    marker_lst = ["o", "^", "s"]
    return {marker_name.replace("  ", " ") : marker_lst[n] for n, marker_name in enumerate(emissions_lst)}

def set_figure_title(fig):
    fig.suptitle(
        f'Sensitivity analysis of characterization factor for Penicillin',  # Your global title
        fontsize=14, y=1        # y adjusts vertical position
    )

def scaling_and_unit():
    scale = 1000
    unit = 'grams of 1,4-DCB-eq'
    
    return scale, unit

def base_car_barchart(data, ax):
    data.plot(
        kind="bar",
        ax=ax,
        colormap="coolwarm",
        # width=0.75,
        zorder=2
        )

def sensitivity_penicillin_cf_scatter_plot(data, emission_dct, marker_dct, col, ax):
    for x, idx in enumerate(data.index):
        df_cf = pd.DataFrame(emission_dct[idx])
        lst = df_cf[col]
        colors = init.color_range(colorname="coolwarm", color_quantity=len(lst)+1)
        colors = colors[1:]
        scale, _ = scaling_and_unit()
        for c, (s, val) in enumerate(lst.items()):
            
            try:
                # print(x)
                ax.scatter(
                    x=x,
                    y=list(val.values())[0]*scale,
                    color=colors[c],
                    marker=marker_dct[list(val.keys())[0]],
                    s=10,  # Adjusted marker size
                    zorder=3
                )
            except (IndexError, KeyError):
                pass
    return colors, df_cf

def get_penicillin_handles(colors, df_cf, marker_dct):
    pen_handles = []
    pen_labels = []
    
    for cl, lbl in enumerate(df_cf.index):
        for em, mk in marker_dct.items():
            pen_handles.append(
                Line2D(
                    [0], [0],
                    marker=mk,
                    linestyle='None',
                    markerfacecolor=colors[cl],   # fill color
                    markeredgecolor=colors[cl],   # outline color
                    markersize=4,
                    label=lbl
                )
            )
            pen_labels.append(f"{lbl[:4]}. {em.split()[-1].title()}")

    return pen_handles, pen_labels

def set_legend(axes_counter, ax, colors, df_cf, marker_dct):
    pen_handles, pen_labels = get_penicillin_handles(colors, df_cf, marker_dct)
    if axes_counter == 2:
        ax.legend(
            pen_handles,
            pen_labels,
            ncol=4,
            bbox_to_anchor=(2.35, -0.12),
            frameon=False
        )
        
    else:
        ax.get_legend().remove()

def set_y_axis_value(axes_counter, ax):
    if axes_counter == 0:
        ax.set_yticks(np.arange(0, 260, 50))
        ax.set_ylim(0, 260)
    elif axes_counter == 1:
        ax.set_yticks(np.arange(0, 105, 20))
        ax.set_ylim(0, 105)
    elif axes_counter == 2:
        ax.set_yticks(np.arange(0, 3100, 500))
        ax.set_ylim(0, 3050)
    elif axes_counter == 3:
        ax.set_yticks(np.arange(0, 155, 25))
        ax.set_ylim(0, 155)

def penicillin_cf_sensitivity_plot():
    width, height, dpi = init.plot_dimensions()

    fig, axes = plt.subplots(2,2, figsize=(width, height), dpi=dpi)

    set_figure_title(fig)

    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    axes = axes.flatten()
    penicillin_treatment_name = ["IV", "Oral"]

    _, impact =  max_penicillin_cf()
    emissions_dct, emissions_lst = get_penicillin_cf_impact()
    marker_dct = get_emission_type_marker(emissions_lst)

    for axes_counter, col in enumerate(impact.columns):

        scale, unit = scaling_and_unit()

        ax = axes[axes_counter]
        data = impact[col].to_frame()*scale

        base_car_barchart(data, ax)
        
        colors, df_cf = sensitivity_penicillin_cf_scatter_plot(data, emissions_dct, marker_dct, col, ax)

        set_legend(axes_counter, ax, colors, df_cf, marker_dct)

        ax.set_xticklabels(penicillin_treatment_name, rotation=0)
        ax.set_title(col)
        ax.set_ylabel(unit)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
        set_y_axis_value(axes_counter, ax)
    
    output_file = init.join_path(init.figure_folder, f"penicillin_cf_sensitivity.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')