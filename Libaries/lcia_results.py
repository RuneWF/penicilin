import bw2data as bd
import bw2calc as bc
import pandas as pd
import copy

# Importing self-made libraries
from standards import *
from life_cycle_assessment import *
from results_figures import *
import sensitivity as st

def quick_LCIA_calculator(unique_process_index, func_unit, impact_categories, file_name_unique, sheet_name, case):
    # Ensure impact categories is a list
    if isinstance(impact_categories, tuple):
        impact_categories = [ic for ic in impact_categories]
    
    # Initialize DataFrame to store results
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)

    print(f'Total amount of calculations {len(impact_categories) * len(func_unit)}')
    
    # Set up and perform the LCA calculation
    bd.calculation_setups[f'calc_setup_{case}'] = {'inv': func_unit, 'ia': impact_categories}
    mylca = bc.MultiLCA(f'calc_setup_{case}')
    res = mylca.results
    
    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_unique.iat[col, row] = val

    # Save results to file
    save_LCIA_results(df_unique, file_name_unique, sheet_name)

    return df_unique

def redo_LCIA_unique_process(df_unique, initialization, file_name_unique, sheet_name):
    database_project, database_name, flows, lcia_method, db_type = initialization
    functional_unit, impact_category, plot_x_axis_all = LCA_initialization(database_project, database_name, flows, lcia_method, db_type)

    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category

    unique_process = []
    unique_process_index = []
    
    # Identify unique processes
    for func_dict in functional_unit:
        for FU_item in func_dict.values():
            for proc in FU_item.keys():
                if proc not in unique_process:
                    unique_process.append(proc)
                    unique_process_index.append(f'{proc}')
    
    unique_process_index.sort()

    unique_process_ordered = [0] * len(unique_process)

    # Order unique processes
    for i, upi in enumerate(unique_process_index):
        for proc in unique_process:
            if upi == f'{proc}':
                unique_process_ordered[i] = proc
                
    redo_func_unit = []
    process_index = []
    
    # Ask user which activities to recalculate
    for idx in unique_process_ordered:
        user_input = input(f'Do you want to redo the calculation for {idx}? [y/n]')
        if 'y' in user_input.lower():
            redo_func_unit.append({idx: 1})
            process_index.append(f'{idx}')

    # Initialize DataFrame for recalculated results
    df_unique_redone = pd.DataFrame(0, index=process_index, columns=impact_categories, dtype=object)

    print(f'Calculating for {len(impact_categories)} methods and {len(redo_func_unit)} activities: Total calculations {len(impact_categories) * len(redo_func_unit)}')
    
    # Perform the LCA calculation and store results
    bd.calculation_setups['calc_setup'] = {'inv': redo_func_unit, 'ia': impact_categories}
    mylca = bc.MultiLCA('calc_setup')
    res = mylca.results
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_unique_redone.iat[col, row] = val

    # Insert recalculated results into original DataFrame
    df_unique_copy = copy.deepcopy(df_unique)
    for col in impact_categories:
        for row in df_unique_redone.itertuples():
            df_unique_copy.at[row.Index, col] = getattr(row, col)

    # Save updated results to file
    save_LCIA_results(df_unique_copy, file_name_unique, sheet_name)
    
    return df_unique_copy

def lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, FU, functional_unit, flows):
    user_input = ''
    
    # Check if file exists
    if os.path.isfile(file_name_unique):
        try:
            # Import LCIA results
            df_unique = import_LCIA_results(file_name_unique, impact_categories)
            user_input = input(f"Do you want to redo the calculations for some process in {initialization[1]}?\n"
                               "Options:\n"
                               "  'y' - Yes, redo calculations for some processes\n"
                               "  'n' - No, do not redo any calculations\n"
                               "  'a' - Redo everything\n"
                               "  'r' - Recalculate based only on the functional unit (FU)\n"
                               "Please enter your choice: ")
            
            if 'y' in user_input.lower():
                df_unique = redo_LCIA_unique_process(df_unique, initialization, file_name_unique, sheet_name)
            elif 'a' in user_input.lower():
                df_unique = quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name, case=initialization[1])
        except (ValueError, KeyError, TypeError) as e:
            print(e)
            df_unique = quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name, case=initialization[1])
    else:
        print(f"{file_name_unique} do not exist, but will be created now")
        df_unique = quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name, case=initialization[1])

    if 'n' in user_input.lower():
        df = import_LCIA_results(file_name, impact_categories)
    else:
        df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)

        for col in impact_categories:
            for _, row in df.iterrows():
                row[col] = []

        for col, impact in enumerate(impact_categories):
            for proc, fu in functional_unit.items():
                for key, item in fu.items():
                    exc = str(key)
                    val = float(item)
                    factor = df_unique.at[exc, impact]
                    if factor is not None:
                        impact_value = val * factor
                    else:
                        impact_value = None
       
                    try:
                        df.at[proc, impact].append([exc, impact_value])
                    except ValueError:
                        try:
                            df.at[proc, impact].append([exc, impact_value])
                            exc = exc.replace(")",")'")
                        except ValueError:
                            print(f'value error for {proc}')
        
        save_LCIA_results(df, file_name, sheet_name)   

    return df

def quick_LCIA(initialization, file_name, file_name_unique, sheet_name):
    """
    Perform a quick Life Cycle Impact Assessment (LCIA) calculation.

    Parameters:
    initialization (tuple): Contains database project, database name, flows, LCIA method, and database type.
    file_name (str): The name of the file to save the results.
    file_name_unique (str): The name of the file to save unique process results.
    sheet_name (str): The name of the sheet to save the results.

    Returns:
    tuple: DataFrame with LCIA results, list of impact categories for plotting, list of impact categories, DataFrame with unique process results.
    """
    _, database_name, flows, lcia_method, db_type = initialization
    functional_unit, impact_category = LCA_initialization(database_name, flows, lcia_method)

    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category
    
    unique_process_index = []
    unique_process = []

    # Identify unique processes
    for exc in functional_unit.values():
        for proc in exc.keys():
            if str(proc) not in unique_process_index:
                unique_process_index.append(str(proc))
                unique_process.append(proc)
    
    unique_process_index.sort()

    FU = []
    for upi in unique_process_index:
        for proc in unique_process:
            if upi == f'{proc}':
                FU.append({proc: 1})

    # Obtain a shortened version of the impact categories for the plots
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]

    df = lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, FU, functional_unit, flows)

    return df, plot_x_axis_all, impact_categories

def df_index(key):
    if '1' in key:
        flow_leg = [
                    'H2S',
                    'H2R',
                    'ASC',
                    'ASW',
                    'H4S',
                    'H4R',
                    'ALC',
                    'ALW'
                    ]
        return flow_leg
    else:
        return ['SUD', 'MUD']

def break_even_dataframe(be_path, case, lcia_method='recipe'):
    file = s.join_path(be_path,'results')
    file_path = s.join_path(file, f'{case}\\break_even_data_{case}.xlsx')

    variables = st.break_even_initialization(be_path, lcia_method)

    with pd.ExcelWriter(file_path) as writer:
        for key in variables.keys():
            df_GWP = variables[key][1]
            database_name = variables[key][0]
            flow_legend = variables[key][4]
            columns = variables[key][-1]

            columns = unique_elements_list(database_name)

            df_be, ignore = process_categorizing(df_GWP, case, flow_legend, columns)
            df_be.index = df_index(key)
            df_be_copy = break_even_orginization(df_be, database_name)
            
            # Write each DataFrame to a different sheet
            sheet_name = f"{key}"
            df_be_copy.to_excel(writer, sheet_name=sheet_name, index=True, header=True)
