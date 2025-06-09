import bw2data as bd
import bw2calc as bc
import pandas as pd

# Importing self-made libraries
from standards import *
from results_figures import *
import main as m

path = r'C:/Users/ruw/Desktop'
matching_database = "ev391cutoff"

lca_init = m.main(path=path,matching_database=matching_database)

def extract_func_unit():
    db = lca_init.db
    func_unit = []
    idx_lst = []
    func_unit_keys = []
    for act in db:
        if "defined system" in act["name"]:
            idx_lst.append(act["name"])
            func_unit_keys.append(act)
    func_unit_keys.sort()
    func_unit_keys.reverse()
    idx_lst.sort()

    for key in func_unit_keys:
        func_unit.append({key : 1})
        
    return func_unit, idx_lst

def perform_LCIA():
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
    func_unit, idx_lst = extract_func_unit()
    # Initialize DataFrame to store results
    df = pd.DataFrame(0, index=idx_lst, columns=impact_categories, dtype=object)

    print(f'Total amount of calculations: {len(impact_categories) * len(idx_lst)}')
    # Set up and perform the LCA calculation
    bd.calculation_setups[f'1 treatment'] = {'inv': func_unit, 'ia': impact_categories}
     
    mylca = bc.MultiLCA(f'1 treatment')
    res = mylca.results
    
    # Store results in DataFrame
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df.iat[col, row] = val

    # Save results to file
    save_LCIA_results(df, lca_init.file_name, "results")

    return df

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

def obtain_LCIA_results(calc):
    
    file_name = lca_init.file_name

    # Check if the file exists
    if os.path.isfile(file_name):
        try:
            if calc:
                df = perform_LCIA()
        
        except (ValueError, KeyError, TypeError) as e:
            print(e)
            # Perform LCIA if there is an error in importing results
            df = perform_LCIA()
    
    else:
        print(f"{file_name} do not exist, but will be created now")
        # Perform LCIA if file does not exist
        df = perform_LCIA()

    # Import LCIA results if user chooses 'n'
    if calc is False:
        df = import_LCIA_results(file_name, lca_init.lcia_impact_method())
        
 
    return df





