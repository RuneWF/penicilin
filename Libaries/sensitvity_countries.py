import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bw2data as bd
import brightway2 as bw
import bw2calc as bc
from pathlib import Path
from uuid import uuid4

import main as m

init = m.main()

pen_type =  ["G", "V"]

databases_lst = [db for db in bw.databases if "cut_off" in db]
databases_lst.sort()

country_order = {
    "IN" : "India",
    "CN" : "China",
    "IT" : "Italy",
    "CH" : "Switzerland",
    "US" : "USA",
    "DK" : "Denmark"
    }

def countries_sensitivity_database():

    del bw.databases['countries sensitivity']
    init.database_setup(reload=True, sensitivty=True)
    
    countries_sensitivity_database = "countries sensitivity"

    if countries_sensitivity_database not in bw.databases:
        bw.Database(countries_sensitivity_database).write({})  # registers an empty database
        print(f"countries_sensitivity_database is created")
    countries_db = bw.Database(countries_sensitivity_database)

    db_to_extract = {}

    for db_str in databases_lst:
        src_db = bw.Database(db_str)

        # Map of names to codes for activities whose name contains "defined system"
        db_to_extract[src_db.name] = {
            a.get("name"): a.get("code")
            for a in src_db
            if "defined system" in (a.get("name") or "")
        }

        for src in src_db:
            name = src.get("name")
            location = src.get("database")[-2:]
            name_loc = fr"{name}_{location}"
            if "defined system" not in (name or ""):
                continue

            if name_loc not in [act['name'] for act in countries_db]:
                print(f"{name_loc} not in {countries_db.name}")
                # Create the new activity in the target DB
                new = countries_db.new_activity(
                    code=uuid4().hex,
                    name=name_loc,
                    unit=src.get("unit"),
                    location=src.get("location"),
                    type="process",  # <-- FIX: node type, not exchange type
                    **{"reference product": src.get("reference product")},
                )
                new.save()

                # Optional but recommended: add an explicit production exchange
                new.new_exchange(
                    input=new.key,
                    amount=src.get("production amount", 1),
                    type="production",
                ).save()

                # Optional: copy technosphere & biosphere exchanges
                # (skip production; keep original inputs)
                for exc in src.exchanges():
                    if exc["type"] == "production":
                        continue
                    new.new_exchange(
                        input=exc.input.key,
                        amount=exc["amount"],
                        type=exc["type"],
                    ).save()
            # else:
            #     print(f"{name_loc} is in {countries_db.name}")

    return countries_db

def append_func_unit(act, exc, act_check, func_unit_unq, func_unit_countries, upstream_quantity=1):
    if "defined system" not in str(exc.input):
        if str(exc.input) not in act_check:
            func_unit_unq.append({exc.input :1})
            act_check.append(str(exc.input))
        func_unit_countries[act].update({str(exc.input) : exc['amount']*upstream_quantity})

def is_raw_penicillin(db_act, pen):
    return "raw" in db_act["name"] and pen in db_act["name"]

def is_same_country(db_str, country_code, country_ISO, pen, exc):
    return db_str[-2:] == country_code and db_str[-2:] == country_ISO and pen in exc.input["name"]

def get_country_code(exc):
    return exc.input["name"][-2:]

def extract_activity_raw_penicillin(act, db_act, act_check, func_unit_unq, func_unit_countries):
    for up in db_act.upstream():
        if up.output != db_act:
            upstream_quantity = up["amount"]
    for db_act in db_act.exchanges():
        if "elec" in db_act["name"] or "production" in db_act["type"]:
            append_func_unit(act, db_act, act_check, func_unit_unq, func_unit_countries, upstream_quantity=upstream_quantity)

def is_penicillin_V(exc):
    return "V" in exc.input["name"]

def extract_activity_penicillin_V(act, db_act, act_check, func_unit_unq, func_unit_countries):
    if "production of a medicine strip" in db_act["name"]:
        for vexc in db_act.exchanges():
            if "strip" in vexc["name"] or "tablet" in vexc["name"] or "secondary" in vexc["name"]:
                for vup in db_act.upstream():
                    upstream_quantity = vup["amount"]
                append_func_unit(act, vexc, act_check, func_unit_unq, func_unit_countries, upstream_quantity= upstream_quantity/vexc["amount"])

def is_glass_vial_production(db_act):
    return "production of glass vial" in db_act["name"]

def extract_activity_glass_vial_production(act, db_act, act_check, func_unit_unq, func_unit_countries):
    for gv_exc in db_act.exchanges():
        if gv_exc["type"] == "production":
            append_func_unit(act, gv_exc, act_check, func_unit_unq, func_unit_countries)

def is_transport(db_act, pen):
    
    if "V" in pen:
        return "production of a medicine strip" in db_act["name"] 
    else:
        return "packaging of glass vials with penicillin G" in db_act["name"]

def is_IV_or_oral(pen):
    if "V" in pen:
        return 20
    else:
        return 10

def extract_activity_transport(act, db_act, act_check, func_unit_unq, func_unit_countries, pen):
    for trans_exc in db_act.exchanges():
        if "transport" in trans_exc.input["name"]:
            for up in db_act.upstream():
                upstream_quantity = up["amount"]
            upstream_quantity_scaling = upstream_quantity / is_IV_or_oral(pen)
            append_func_unit(act, trans_exc, act_check, func_unit_unq, func_unit_countries, 
                             upstream_quantity=upstream_quantity_scaling)
            
def is_box_of_vials(db_act, pen):
    return "G" in pen and "packaging of glass vials with penicillin G" in db_act["name"]

def extract_production_amounts(act, pen, country_ISO, act_check, func_unit_unq, func_unit_countries):
    for exc in act.exchanges():
        country_code = get_country_code(exc)
        for db_str in databases_lst:
            if is_same_country(db_str, country_code, country_ISO, pen, exc):
                db = bw.Database(db_str)
                for db_act in db:
                    if is_raw_penicillin(db_act, pen):
                        extract_activity_raw_penicillin(act, db_act, act_check, func_unit_unq, func_unit_countries)
                    
                    elif is_glass_vial_production(db_act):
                        extract_activity_glass_vial_production(act, db_act, act_check, func_unit_unq, func_unit_countries)

                    elif is_transport(db_act, pen):
                        extract_activity_transport(act, db_act, act_check, func_unit_unq, func_unit_countries, pen)
                    
                    elif is_penicillin_V(exc):
                        extract_activity_penicillin_V(act, db_act, act_check, func_unit_unq, func_unit_countries)

                    if is_box_of_vials(db_act, pen):
                        for exc in db_act.exchanges():
                            if exc['type'] == 'production':
                                exc_act = exc
                        for up in db_act.upstream():
                            amount = up["amount"]
                        append_func_unit(act, exc_act, act_check, func_unit_unq, func_unit_countries, upstream_quantity=amount)
                    
def fill_penicillin_dct_keys(pen_type, countries_db):
    func_unit_countries =  {}
    for pt in pen_type:
        for act in countries_db:
            if f"Penicillin {pt}" in str(act['name']) and "defined system" in str(act['name']):
                func_unit_countries[act] = {}
    return func_unit_countries

def is_pen_and_country_ISO(pen, act, country_ISO):
    return pen in act["name"] and country_ISO in act["name"]

def is_technosphere(exc):
    return exc['type'] == 'technosphere' and "packaging of glass vials" not in str(exc.input)

def is_act_type_process(act):
    return act["type"] == "process"

def func_unit_to_dataframe(func_unit_countries):
    df = pd.DataFrame(func_unit_countries)

    new_cols = []
    for pen in pen_type: 
        for iso in country_order.keys():
            matching_cols = [col for col in df.columns if (iso in col["name"] and pen in col["name"])]
            if matching_cols:
                new_cols.append(matching_cols[0])

    df_new = pd.DataFrame(0, index=df.index, columns=new_cols, dtype=object)

    for col in df_new.columns:
        df_new[col] = df[col]
    
    return df_new

def get_functional_unit_and_unique_activities():
    act_check = []
    func_unit_unq = []
    countries_db = countries_sensitivity_database()
    func_unit_countries = fill_penicillin_dct_keys(pen_type, countries_db)
    
    for pen in pen_type:
        for country_ISO in country_order.keys():
            for act in countries_db:
                if is_pen_and_country_ISO(pen, act, country_ISO):
                    for exc in act.exchanges():
                        if is_technosphere(exc):
                            append_func_unit(act, exc, act_check, func_unit_unq, func_unit_countries)
                if is_act_type_process(act):
                    extract_production_amounts(act, pen, country_ISO, act_check, func_unit_unq, func_unit_countries)
    
    print(f"Found {len(func_unit_unq)} unique activites")

    df = func_unit_to_dataframe(func_unit_countries)

    # Precompute a lookup dict: {key: fu_dict}
    lookup = {str(list(fu.keys())[0]): fu for fu in func_unit_unq}
    
    # Build the sorted list in one pass
    func_unit_unq_sorted = [lookup[idx] for idx in df.index if idx in lookup]

    return df, func_unit_unq_sorted

def value_correction_dct():
    return {
        "strip" : ["tablet", "secondary", "transport"],
        "glass vials with" :["raw", "production of glass vial", "transport"],
        "tablet" : ["raw"],
        "raw pen" : ["elec"],
    }

def pen_contries_cat_dct():
    return {
    "P. chemicals": ["raw penicillium"],
    "P. packaging" : ["strip", "vial", "secondary"],
    "P. Electricity": ["electricity"],
    "Transport" : ["transport"],
    "Auxilary prod.": ["wipe", "glove", "stopcock", "water", "sodium", " connector", "IV", "medicin cup"],
    "Disposal": ["waste"]
    }

def update_df_gwp_columns(df_gwp):
    try:
        df_gwp.columns = [str(col) for col in df_gwp.columns]
    except TypeError:
        pass 

def categorize_gwp_results(df_gwp):

    update_df_gwp_columns(df_gwp)

    for col in df_gwp.columns:
        df_temp = df_gwp[col].dropna().to_frame()
        for key, item in value_correction_dct().items():
            idx_key = [idx for idx in df_temp.index if key in str(idx)]
            idx_item = [idx for idx in df_temp.index if any(i in str(idx) for i in item)]
            if idx_key and idx_item:

                val = df_temp.loc[idx_item].sum()

                df_gwp.loc[idx_key, col] -= val.sum()
        
    df_cat = pd.DataFrame(0, index=list(pen_contries_cat_dct().keys()), columns=df_gwp.columns, dtype=object)

    for col in df_gwp.columns:
        df_col = df_gwp[col].dropna().to_frame()
        for idx, row in df_col.iterrows():
            for cat, st in pen_contries_cat_dct().items():
                if any(s in str(idx) for s in st):
                    df_cat.loc[cat, col] += row[col]
    
    dct_sens_coutries = {p : df_cat[[col for col in df_cat.columns if p in col]] for p in pen_type}

    for pen, df in dct_sens_coutries.items():
        new_cols = []
        for iso in country_order.keys():
            new_cols.append([col for col in df.columns if f"_{iso}" in col][0])
        df_new = pd.DataFrame(0, index=df.index, columns=new_cols, dtype=object)

        for col in df_new.columns:
            df_new[col] = df[col]

        dct_sens_coutries[pen] = df_new

    return dct_sens_coutries

def calculate_gwp(calc):
    countries_sens_path = Path.joinpath(init.results_path, "sensitivity", "countries_sens.xlsx")
    ics = init.lcia_impact_method()  
    gwp =  [ics[1]]
    if countries_sens_path.exists() and calc is False:
        df_gwp = init.import_LCIA_results(countries_sens_path)

    elif not countries_sens_path.exists() or calc:
        df_fu, func_unit_unq = get_functional_unit_and_unique_activities()
        # Set up and perform the LCA calculation
        bd.calculation_setups["countries"] = {'inv': func_unit_unq, 'ia': gwp}
            
        mylca = bc.MultiLCA("countries")
        res = mylca.results
        df_gwp = df_fu * res * 1000
        
        init.save_LCIA_results(df_gwp, countries_sens_path)

    dct_sens_coutries = categorize_gwp_results(df_gwp)

    return dct_sens_coutries

def set_y_ticks(p, ax):
    if p == 0:
        y_ticks = np.linspace(0, 900, 10)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, 905)

def penicillin_G_V_to_IV_oral(pen_type):
    if "G" in pen_type:
        return "IV"
    else:
        return "Oral"

def countries_sens_plot(calc):
    dct_sens_coutries = calculate_gwp(calc)
    colors = init.color_range(colorname="coolwarm", color_quantity=len(pen_contries_cat_dct().keys()))
    width_in, height_in, dpi = init.plot_dimensions()
    fig, axes = plt.subplots(1, 2, figsize=(width_in*1.6, height_in), dpi=dpi)
    fig.suptitle(
        f'Sensitivity analysis of origin of manufacturing - Global Warming Potential',  # Your global title
        fontsize=16, 
        y=0.95       # y adjusts vertical position
    )
    title_identifier = [r"$\bf{Fig\ A:}$", r"$\bf{Fig\ B:}$"]

    for p, df in enumerate(dct_sens_coutries.values()):
        df.T.plot(
                    kind='bar',
                    stacked=True,
                    title="",
                    color=colors,
                    ax=axes[p],
                    width=0.6,
                    edgecolor="k",
                    zorder=10
                )
        axes[p].set_title(f"{title_identifier[p]} A SSD {penicillin_G_V_to_IV_oral(pen_type[p])} treatment", loc="left")
        axes[p].set_xticklabels(list(country_order.values()), rotation=0)
        axes[p].grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
        set_y_ticks(p, axes[p])
        axes[p].set_ylabel('grams of CO$_2$-eq per SSD treatment')
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
                frameon=False
                )
        else:
            axes[p].get_legend().remove()
    
    plt.tight_layout()
    output_file = init.join_path(init.figure_folder, f"countries_sens.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

    return dct_sens_coutries