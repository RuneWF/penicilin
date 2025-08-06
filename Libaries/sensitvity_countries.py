import bw2data as bd
import bw2calc as bc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import main as m

init = m.main()

pen_type =  ["G", "V"]

def extract_penG_electricity(sub, lst, val):
    val *= sub['amount']
    elec1 = val
    elec2 = val
    elec_val = {}
    for sub2x in sub.input.exchanges():
        if "electricity" in sub2x['name']:
            elec1 *= sub2x['amount']
            elec_val["prod 1"] = elec1
        elif "penicil" in sub2x['name']:
            lst.append({sub2x.input : sub2x['amount']})
            elec2 *= sub2x['amount']
            for sub3x in sub2x.input.exchanges():
                if "electricity" in sub3x['name']:
                    elec2 *=  sub3x['amount']
                    key = sub3x.input
                    elec_val["prod 2"] = elec2
    lst.append({key : elec1 + elec2})

    return elec_val

def extract_penG_activities(val, exc, lst):
    for sub in exc.input.exchanges():
        # print(sub)
        if "transport" in sub['name']:
            lst.append({sub.input : val*sub['amount']})
        elif  "filling" in sub['name']:
            elec_val = extract_penG_electricity(sub, lst, val)
    return elec_val

def extract_penV_activities(lst, val, exc):
    elec_val = {}
    for sub in exc.input.exchanges():
        if 'transport' in sub['name']:
            lst.append({sub.input : val*sub['amount']})
        elif "production" in sub['name']:
            val *= sub['amount']
            for sub2x in sub.input.exchanges():
                
                elec1 = val
                elec2 = val
                elec3 = val
                for sub2x in sub.input.exchanges():
                    if "electricity" in sub2x['name']:
                        elec1 *= sub2x['amount']
                        elec_val["prod 1"] = elec1
                    elif "tablet" in sub2x['name']:
                        elec2 *= sub2x['amount']
                        elec3 *= sub2x['amount']
                        for sub3x in sub2x.input.exchanges():
                            if "electricity" in sub3x['name']:
                                elec2 *=  sub3x['amount']
                                elec_val["prod 2"] = elec2
                            elif "penicil" in sub3x['name']:
                                if {sub3x.input : sub3x['amount']} not in lst:
                                    lst.append({sub3x.input : sub3x['amount']})
                                elec3 *= sub3x['amount']
                                for sub4x in sub3x.input.exchanges():
                                    if "electricity" in sub4x['name']:
                                        elec3 *=  sub4x['amount']
                                        key = sub4x.input
                                        elec_val["prod 3"] = elec3
            lst.append({key : elec1 + elec2 + elec3})
    return elec_val

def initialize_func_unit_keys(pen_type):
    
    func_unit = {}
    db = init.db
    for pt in pen_type:
        for act in db:
            if f"Penicillin {pt}" in str(act['name']) and "defined system" in str(act['name']):
                func_unit[act['name']] = {}
    
    return func_unit

def create_func_unit(sensitivty=False):
    system_path = init.system_path
    init.import_databases(sensitivty)
    sheets_to_import = init.extract_excel_sheets()
    
    func_unit = initialize_func_unit_keys(pen_type)

    for sheet in sheets_to_import:
        data = pd.read_excel(system_path, sheet_name=sheet)
        db_name = data.columns[1]
        db = bd.Database(db_name)
        
        for pt in pen_type:
            temp_lst = []
            for act in db:
                
                if f"Penicillin {pt}" in act['name'] and "defined system" in str(act['name']):
                    for exc in act.exchanges():
                        if exc['type'] == 'technosphere':
                            val = exc['amount']
                            temp_lst.append({exc.input : exc['amount']})
                            if "packaging of glass vials with penicillin G" in exc.input['name']:
                                elec_val_G = extract_penG_activities(val, exc, temp_lst)
                            elif "medicine strip" in exc.input['name']:
                                elec_val_V = extract_penV_activities(temp_lst, val, exc)
                        func_unit[act['name']].update({db_name : temp_lst})
                
    return func_unit, elec_val_G, elec_val_V

def perform_LCIA_countries_sens(func_unit):
    # Initialize DataFrame to store results
    df_dct = {}
    for pen, fu_dct in func_unit.items():
        pen_arr = []
        for country, fu in fu_dct.items():
            print(f"Performing LCIA for {pen} - {country}")
            idx_lst = []
            for dct in fu:
                idx_lst.append(list(dct.keys())[0])
            idx_lst.sort
            ics = init.lcia_impact_method()                
            # Set up and perform the LCA calculation
            bd.calculation_setups[str(country)] = {'inv': fu, 'ia': [ics[1]]}
                
            mylca = bc.MultiLCA(str(country))
            res = mylca.results
            pen_arr.append(res)


        df_dct[pen] = pen_arr
    return df_dct


def calc_penicillin_impact(elec_val_V, elec_val_G):

    tot_elec_V = 0
    tot_elec_G = 0

    for val in elec_val_V.values():
        tot_elec_V += val

    for val in elec_val_G.values():
        tot_elec_G += val


    penV_prod_share = elec_val_V["prod 3"]/tot_elec_V
    penG_prod_share = elec_val_G["prod 2"]/tot_elec_G
    

    return penG_prod_share, penV_prod_share
    
def results_correction(func_unit, df_dct, elec_val_V, elec_val_G):
    impact_dct = {pen: None for pen in func_unit.keys()}
    for pen, fu in func_unit.items():
        impact_dct[pen] = {}
        for lst_idx, (country, lst) in enumerate(fu.items()):
            for val_idx, act in enumerate(lst):
                val = df_dct[pen][lst_idx][val_idx]
                act_key = list(act.keys())[0]
                if country not in impact_dct[pen]:
                    impact_dct[pen][country] = {}
                impact_dct[pen][country][str(act_key)] = val[0]


    G = "'packaging of glass vials with penicillin G'"
    V = "'packaging of a medicine strip'"

    penG_prod_share, penV_prod_share = calc_penicillin_impact(elec_val_V, elec_val_G)
    for pen, res_dct in impact_dct.items():
        for country, val_dct in res_dct.items():
            for key, val in val_dct.items():
                if "electricity" in str(key) and "V" in pen:
                    proc = f"'manufacturing of raw penicillium V' (kilogram, {country[-2:]}, None)"
                    pen_impact = val_dct[proc] 
                    val_dct[proc] = pen_impact-(val*penV_prod_share)
                    pen_impact = val_dct[proc] 
                if "electricity" in str(key) and "G" in pen:
                    proc = f"'manufacturing of raw penicillium G' (kilogram, {country[-2:]}, None)"
                    pen_impact = val_dct[proc] 
                    val_dct[proc] = pen_impact-(val*penG_prod_share)
                    pen_impact = val_dct[proc] 

    for pen, res_dct in impact_dct.items():
        for country, val_dct in res_dct.items():
            for key, val in val_dct.items():
                if ("transport" in str(key) or "electricity" in str(key) or "raw penicillium" in str(key)) and "G" in pen:
                    val_dct[G + f" (unit, {country[-2:]}, None)"] -= val
                elif ("transport" in str(key) or "electricity" in str(key) or "raw penicillium" in str(key)) and "V" in pen:
                    val_dct[V + f" (unit, {country[-2:]}, None)"] -= val

    return impact_dct

def category_sorting():
    pen_contries_cat_dct = {
    "Penicillin manufacturing": ["raw penicillium"],
    "Penicillin packaging" : [ "strip", "vial"],
    "Electricity": ["electricity"],
    "Transport" : ["transport"],
    "Auxilary product": ["wipe", "glove", "stopcock", "water", "sodium", " connector", "IV", "medicin cup"],
    "Disposal": ["waste"]
    }

    return pen_contries_cat_dct

def LCIA_countries(func_unit, elec_val_V, elec_val_G, save_dir, df_res_dct):
    df_dct = perform_LCIA_countries_sens(func_unit)
    impact_dct = results_correction(func_unit, df_dct, elec_val_V, elec_val_G)
    pen_contries_cat_dct = category_sorting()
    for p, (pen, res_dct) in enumerate(impact_dct.items()):
        col = list(res_dct.keys())
        idx = list(pen_contries_cat_dct.keys())
        df = pd.DataFrame(0, index=idx, columns=col, dtype=object)

        
        for country, dct_ in res_dct.items():
            for act, val in dct_.items():
                
                for cat, keywords in pen_contries_cat_dct.items():
                    # Check each keyword in the category
                    for keyword in keywords:
                        if keyword in str(act):
                            df.loc[cat, country] += val
        
        excel_file = init.join_path(save_dir, f"countries_pen_{pen_type[p]}.xlsx")
        
        init.save_LCIA_results(df, excel_file)
        df_res_dct[pen] = df

def results_sorting(func_unit, elec_val_V, elec_val_G, calc):
    save_dir = init.results_folder(init.results_path, "sensitivity")
    df_res_dct = {}
    if os.path.exists(save_dir) and calc is False:
        for p, (pen, country_dct) in enumerate(func_unit.items()):
                excel_file = init.join_path(save_dir, f"countries_pen_{pen_type[p]}.xlsx")
                df_res_dct[pen] = init.import_LCIA_results(excel_file)

    elif os.path.exists(save_dir) is False or calc:
        LCIA_countries(func_unit, elec_val_V, elec_val_G, save_dir, df_res_dct)

    else:
        print("Error")
    
    return df_res_dct

def set_y_ticks(p, ax):
    if p == 0:
        y_ticks = np.linspace(0, 1, 11)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, 1.01)
    else:
        y_ticks = np.linspace(0, 0.06, 7)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, 0.0601)

def penicillin_G_V_to_IV_oral(pen_type):
    if "G" in pen_type:
        return "IV"
    else:
        return "Oral"

def countries_sens_plot(calc=False, sensitivity=False):
    width = 0.5
    
    func_unit, elec_val_G, elec_val_V = create_func_unit(sensitivity)
    
    df_res_dct = results_sorting(func_unit, elec_val_V, elec_val_G, calc)
    title_identifier = [r"$\bf{Fig\ A:}$ ", r"$\bf{Fig\ B:}$ "]
    
    plot_save_path = init.join_path(init.path_github, r"figures")
    
    width_in, height_in, dpi = init.plot_dimensions(subfigure=True)
    fig, axes = plt.subplots(1, len(df_res_dct.keys()), figsize=(width_in*1.9, height_in), dpi=dpi)

    for p, df in enumerate(df_res_dct.values()):
        colors = init.color_range(colorname="coolwarm", color_quantity=len(df.index))
        df.T.plot(
            kind='bar',
            stacked=True,
            title="",
            color=colors,
            ax=axes[p],
            width=width,
            edgecolor="k",
            zorder=10
        )

        axes[p].set_title(f"{title_identifier[p]}GWP for 1 {penicillin_G_V_to_IV_oral(pen_type[p])} treatment", loc="left")
        axes[p].set_ylabel('kilograms of CO$_2$-eq per treatment')

        xtick_txt = []
        for col in df.columns:
            xtick_txt.append(col[-2:])

        set_y_ticks(p, axes[p])

        axes[p].set_xticklabels(xtick_txt, rotation=0)
        axes[p].grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
        # Only show legend for the second figure (p == 1)
        if p == 1:
            leg_color, _ = fig.gca().get_legend_handles_labels()
            leg_txt = list(df.index)
            leg_txt = leg_txt[::-1]
            leg_color = leg_color[::-1]
            axes[p].legend(
                leg_color,
                leg_txt,
                loc='upper left',
                bbox_to_anchor=(0.995, 1.02),
                ncol=1,
                fontsize=10,
                frameon=False
            )
        else:
            axes[p].get_legend().remove()
    
    plt.tight_layout()
    output_file = init.join_path(plot_save_path, f"pen{pen_type[p]}_countries_sens.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()