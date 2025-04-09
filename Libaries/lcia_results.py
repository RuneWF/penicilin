import bw2data as bd
import bw2calc as bc
import pandas as pd
import copy

# Importing self-made libraries
from standards import *
from life_cycle_assessment import *
from results_figures import *

def perform_LCIA(unique_process_index, func_unit, impact_categories, file_name_unique, sheet_name, case):
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

    # Ensure impact categories is a list
    if isinstance(impact_categories, tuple):
        impact_categories = list(impact_categories)
    
    # Initialize DataFrame to store results
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)

    print(f'Total amount of calculations: {len(impact_categories) * len(func_unit)}')
    
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

def lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, unique_func_unit, functional_unit, flows):
    user_input = ''
    
    # Check if the file exists
    if os.path.isfile(file_name_unique):
        try:
            # Import LCIA results
            # df_unique = import_LCIA_results(file_name_unique, impact_categories)
            user_input = input(f"Do you want to redo the calculations for some process in {initialization[1]}?\n"
                               "Options:\n"
                               "  'y' - Yes, redo the calculations\n"
                               "  'n' - No, do not redo any calculations\n"
                               "  'r' - Recalculate based only on the functional unit (FU)\n"
                               "Please enter your choice: ")
            
            # Redo calculations if user chooses 'y'
            if 'y' in user_input.lower():
                df_unique = perform_LCIA(unique_process_index, unique_func_unit, impact_categories, file_name_unique, sheet_name, case=initialization[1])
        
        except (ValueError, KeyError, TypeError) as e:
            print(e)
            # Perform LCIA if there is an error in importing results
            df_unique = perform_LCIA(unique_process_index, unique_func_unit, impact_categories, file_name_unique, sheet_name, case=initialization[1])
    
    else:
        print(f"{file_name_unique} do not exist, but will be created now")
        # Perform LCIA if file does not exist
        df_unique = perform_LCIA(unique_process_index, unique_func_unit, impact_categories, file_name_unique, sheet_name, case=initialization[1])

    # Import LCIA results if user chooses 'n'
    if 'n' in user_input.lower():
        df = import_LCIA_results(file_name, impact_categories)
        
    else:
        # Initialize DataFrame for results
        df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)
        df_unique =  import_LCIA_results(file_name_unique, impact_categories)
        for col in impact_categories:
            for idx, row in df.iterrows():
                row[col] = {}

        # Calculate impact values and store in DataFrame
        for col, impact in enumerate(impact_categories):
            for proc, fu in functional_unit.items():
                for row, (key, item) in enumerate(fu.items()):
                    exc = str(key)
                    
                    val = float(item)
                    factor = df_unique.iat[row, col]
                    if factor is not None:
                        impact_value = val * factor
                    else:
                        impact_value = None

                    try:
                        df.at[proc, impact].update({exc : impact_value})
                    except ValueError:
                        try:
                            exc = exc.replace(")",")'")
                            df.at[proc, impact].update({exc : impact_value})
                            
                        except ValueError:
                            print(f'value error for {proc}')
            
        # Save LCIA results to file
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
    _, database_name, flows, lcia_method = initialization
    functional_unit, impact_category = LCA_initialization(database_name, flows)

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

    unique_func_unit = []
    for upi in unique_process_index:
        for proc in unique_process:
            if upi == f'{proc}':
                unique_func_unit.append({proc: 1})

    # Obtain a shortened version of the impact categories for the plots
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]

    df = lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, unique_func_unit, functional_unit, flows)

    return df, plot_x_axis_all, impact_categories



