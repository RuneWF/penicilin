import bw2data as bd
import bw2calc as bc
import pandas as pd

# Importing self-made libraries
from standards import *
# from life_cycle_assessment import *
from results_figures import *
from lca import LCA

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

lca_init = LCA(path=path,matching_database=matching_database)

def perform_LCIA(unique_process_index, unique_func_unit, sheet_name, case):
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

    impact_categories = lca_init.lcia_impact_method()

    # Ensure impact categories is a list
    if isinstance(impact_categories, tuple):
        impact_categories = list(impact_categories)
    
    # Initialize DataFrame to store results
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)

    print(f'Total amount of calculations: {len(impact_categories) * len(unique_func_unit)}')
    # Set up and perform the LCA calculation
    bd.calculation_setups[f'calc_setup_{case}'] = {'inv': unique_func_unit, 'ia': impact_categories}
     
    mylca = bc.MultiLCA(f'calc_setup_{case}')
    res = mylca.results
    
    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_unique.iat[col, row] = val

    # Save results to file
    save_LCIA_results(df_unique, lca_init.file_name_unique_process, sheet_name)

    return df_unique

def func_unit_LCIA(df_unique):
    # Initialize DataFrame for results
    df = pd.DataFrame(0, index=lca_init.flow, columns=lca_init.lcia_impact_method(), dtype=object)

    for col in lca_init.lcia_impact_method():
        for _, row in df.iterrows():
            row[col] = {}

    # Calculate impact values and store in DataFrame
    for col, impact in enumerate(lca_init.lcia_impact_method()):
        for proc, fu in lca_init.func_unit.items():
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
    
    return df

def lcia_dataframe_handling(sheet_name, unique_process_index, unique_func_unit):
    user_input = ''
    
    database = lca_init.database_name
    file_name = lca_init.file_name
    file_name_unique = lca_init.file_name_unique_process

    # Check if the file exists
    if os.path.isfile(file_name_unique):
        try:
            # Import LCIA results
            # df_unique = import_LCIA_results(file_name_unique, impact_categories)
            user_input = input(f"Do you want to redo the calculations for some process in {database}?\n"
                               "Options:\n"
                               "  'y' - Yes, redo the calculations\n"
                               "  'n' - No, do not redo any calculations\n"
                               "Please enter your choice: ")
            
            # Redo calculations if user chooses 'y'
            if 'y' in user_input.lower():
                df_unique = perform_LCIA(unique_process_index, unique_func_unit, sheet_name, case=database)
        
        except (ValueError, KeyError, TypeError) as e:
            print(e)
            # Perform LCIA if there is an error in importing results
            df_unique = perform_LCIA(unique_process_index, unique_func_unit, sheet_name, case=database)
    
    else:
        print(f"{file_name_unique} do not exist, but will be created now")
        # Perform LCIA if file does not exist
        df_unique = perform_LCIA(unique_process_index, unique_func_unit, sheet_name, case=database)

    # Import LCIA results if user chooses 'n'
    if 'n' in user_input.lower():
        df = import_LCIA_results(file_name, lca_init.lcia_impact_method())
        
    elif 'y' in user_input.lower():
        df = func_unit_LCIA(df_unique)
        # Save LCIA results to file
        save_LCIA_results(df, file_name, sheet_name)   
    else:
        print("select either n or y, try again")
        lcia_dataframe_handling(sheet_name, unique_process_index, unique_func_unit)

    return df

def quick_LCIA(sheet_name):
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
    functional_unit = lca_init.LCA_initialization()

    
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

    impact_categories = lca_init.lcia_impact_method()

    # Obtain a shortened version of the impact categories for the plots
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]

    df = lcia_dataframe_handling(sheet_name, unique_process_index, unique_func_unit)

    return df



