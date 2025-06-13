import numpy as np

import bw2data as bd
import bw2calc as bc
import pandas as pd

import matplotlib.pyplot as plt

# import standards as s

import main as m

matching_database = "ev391cutoff"
path = r'C:/Users/ruw/Desktop'

init = m.main(path=path, matching_database=matching_database)

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

    print(f"Min reduction : {round((1-min_reduction)*100,2)}%")
    print(f"Max reduction : {round((1-max_reduction)*100,2)}%")

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

    ax.set_xticklabels(xtick_txt, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
    y_ticks = np.linspace(-0.3, 1, 14)
    ax.set_yticks(y_ticks)
    ax.set_ylim(-0.3, 1.01)

    plt.tight_layout()
    plot_save_path = init.join_path(init.path_github, r"figures")
    output_file = init.join_path(plot_save_path, f"penG_EoL_seninit.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()