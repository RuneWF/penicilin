# Import libaries
import pandas as pd
from copy import deepcopy as dc
import os
from openpyxl import load_workbook
import bw2data as bd


# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import sensitivity_case1 as c1
import sensitivity_case2 as c2
import results_figures as rfig
 

def join_path(path1, path2):
    return os.path.join(path1, path2)

def get_all_flows(path):
    # Set the current Brightway2 project
    bd.projects.set_current("Single Use vs Multi Use")
    
    # Define the types of databases
    db_type = ['apos', 'consq', 'cut_off']

    # Initialize dictionaries to store flows, save directories, database names, and file names
    flows = {}
    save_dir = {}
    database_name_lst = []
    file_name = {}
    db_type_dct = {}

    # Loop through case numbers and database types
    for nr in range(1, 3):
        for tp in db_type:
            # Construct the database name
            database_name = f'case{nr}' + '_' + tp
            database_name_lst.append(database_name)
            db = bd.Database(database_name)
            
            # Initialize a list to store flow names
            flow = []
            
            # Check if the database is case1
            if 'case1' in str(db):
                for act in db:
                    temp = act['name']
                    # Add specific flows based on conditions
                    if ('H2' in temp or 'H4' in temp) and ('SU' in temp or 'REC' in temp) and temp not in flow:
                        flow.append(temp)
                    elif 'alubox' in temp and '+' in temp and 'eol' not in temp.lower():
                        flow.append(temp)
                flow.sort()
            # Check if the database is case2
            elif 'case2' in str(db):
                for act in db:
                    temp = act['name']
                    if temp == 'SUD' or temp == 'MUD':
                        flow.append(temp)
                flow.sort()
                flow.reverse()
            
            # Store the flows in the dictionary
            flows[database_name] = flow
            
            # Create the results folder and store the directory path
            dir_temp = s.results_folder(join_path(path, 'results'), f"case{nr}")
            save_dir[database_name] = dir_temp
            
            # Store the file name for the results
            file_name[database_name] = join_path(dir_temp,rf'data_case{nr}_{tp}_recipe.xlsx')
            
            # Store the database type in the dictionary
            db_type_dct[database_name] = tp
    
    # Return the collected data
    return flows, database_name_lst, db_type_dct, save_dir, file_name

def break_even_initialization(path, lcia_method):
    # Reloading the self-made libraries to ensure they are up to date
    flows, database_name, db_type, save_dir, file_name = get_all_flows(path)
    
    # Get the impact category based on the LCIA method
    impact_category = lc.lcia_impact_method(lcia_method)
    
    # Check if flows is a list
    if isinstance(flows, list):
        # Import the results data frame
        df = lc.import_LCIA_results(file_name, flows, impact_category)
        
        # Rearrange the data frame index
        df_rearranged = lc.rearrange_dataframe_index(df, database=database_name)
        
        # Split the data frame if the LCIA method is 'recipe'
        if 'recipe' in lcia_method:
            df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
        else:
            df_res = df_rearranged
        
        # Separate the GWP from the rest
        df_col = [df_res.columns[1]]
        df_GWP = df_res[df_col]
        
        # Store the variables in a list
        variables = [database_name, df_GWP, db_type, save_dir, flows, impact_category]

    else:
        # Initialize an empty dictionary for variables
        variables = {}
        
        # Loop through the flows dictionary
        for db, key in enumerate(flows.keys()):
            # Import the results data frame
            df = lc.import_LCIA_results(file_name[key], impact_category)
            
            # Rearrange the data frame index
            df_rearranged = lc.rearrange_dataframe_index(df, database_name[db])
            
            # Split the data frame if the LCIA method is 'recipe'
            if 'recipe' in lcia_method:
                df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
            else:
                df_res = df_rearranged
            
            # Separate the GWP potential from the rest
            df_col = [df_res.columns[1]]
            df_GWP = df_res[df_col]
            
            # Store the variables in the dictionary
            variables[key] = [database_name[db], df_GWP, db_type[key], save_dir[key], flows[key], impact_category]
    
    # Return the variables
    return variables

def column_sum_dataframe(df_sensitivity_v):
    """
    Sums the values of each column in the given DataFrame, excluding the 'total' row.

    Parameters:
    df_sensitivity_v (pd.DataFrame): DataFrame containing sensitivity values.

    Returns:
    dict: Dictionary with column names as keys and their respective sums as values.
    """
    tot_dct = {}
    for col in df_sensitivity_v.columns:
        tot_dct[col] = 0
        for idx, row in df_sensitivity_v.iterrows():
            if idx != 'total':
                tot_dct[col] += row[col]
    return tot_dct

def sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity_v):
    """
    Calculates the sensitivity table results based on the given dataframes.

    Parameters:
    totals_df (pd.DataFrame): DataFrame containing total values.
    idx_sens (list): List of sensitivity indices.
    col_to_df (list): List of columns for the sensitivity DataFrame.
    df_sensitivity_v (pd.DataFrame): DataFrame containing sensitivity values.

    Returns:
    pd.DataFrame: DataFrame with sensitivity percentage results.
    """
    tot_lst = {}
    # Create a dictionary with total values from totals_df
    for tidx in totals_df.index:
        tot_lst[tidx] = totals_df.at[tidx, 'Value']

    # Initialize a DataFrame to store sensitivity percentages
    df_sensitivity_p = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    # Calculate the sum of each column in df_sensitivity_v
    dct = column_sum_dataframe(df_sensitivity_v)

    # Iterate over each column and row in df_sensitivity_v
    for col in df_sensitivity_v.columns:
        for idx, row in df_sensitivity_v.iterrows():
            if 'lower' in col:
                tidx = col.replace(" - lower%", "")
                tot = tot_lst[tidx]
            else:
                tidx = col.replace(" - upper%", "")
                tot = tot_lst[tidx]

            if row[col] != 0 and 'total' not in idx:
                val = row[col]
                if 'lower' in col:
                    sens = tot - val
                else:
                    sens = tot + val
                df_sensitivity_p.at[idx, col] = (val - tot) / tot * 100
            elif 'total' in idx:
                if 'lower' in col:
                    tidx = col.replace(" - lower%", "")
                    tot = tot_lst[tidx]
                    df_sensitivity_p.at[idx, col] = ((tot - dct[col]) - tot) / tot * 100
                else:
                    tidx = col.replace(" - upper%", "")
                    tot = tot_lst[tidx]
                    df_sensitivity_p.at[idx, col] = ((tot + dct[col]) - tot) / tot * 100

            # Format the sensitivity percentage values
            if round(df_sensitivity_p.at[idx, col], 2) != 0:
                df_sensitivity_p.at[idx, col] = f"{df_sensitivity_p.at[idx, col]:.2f}%"
            else:
                df_sensitivity_p.at[idx, col] = "-"

    return df_sensitivity_p

def autoclave_gwp_impact_case1(variables, path):
    """
    Calculates the GWP impact of the autoclave for case1 based on the given variables and path.

    Parameters:
    variables (tuple): A tuple containing database name, _, db type, _, flows, and impact category.
    path (str): The path to the results folder.

    Returns:
    float: The GWP impact of the autoclave.
    """
    database_name, _, db_type, _, flows, impact_category = variables
    db = bd.Database(database_name)
    unique_process_index = []

    # Iterate through the database to find unique process indices for the given flows
    for act in db:
        for f in flows:
            for exc in act.exchanges():
                if act['name'] == f and str(exc.input) not in unique_process_index and exc['type'] != 'production':
                    unique_process_index.append(str(exc.input))

    unique_process_index.sort()
    save_dir_case1 = s.results_folder(join_path(path, "results"), 'case1')
    results_path = join_path(save_dir_case1, f"data_uniquie_case1_{db_type}_recipe.xlsx")
    df_unique = lc.import_LCIA_results(results_path, impact_category)
    autoclave_gwp = None

    # Try to find the autoclave GWP impact for different locations
    for location in ["DK", "GLO", "RER"]:
        try:
            autoclave_gwp = df_unique.at[f"'autoclave' (unit, {location}, None)", impact_category[1]]
            if autoclave_gwp is not None:
                break
        except KeyError:
            continue

    if autoclave_gwp is None:
        raise KeyError("Autoclave GWP impact not found for DK, GLO, or RER.")

    return autoclave_gwp

def results_dataframe(df_sensitivity, df_dct, df_be):
    """
    Generates a results DataFrame with sensitivity percentages.

    Parameters:
    df_sensitivity (pd.DataFrame): DataFrame containing initial sensitivity values.
    df_dct (dict): Dictionary containing sensitivity data for different cases.
    df_be (pd.DataFrame): DataFrame containing break-even data.

    Returns:
    pd.DataFrame: DataFrame with sensitivity percentage results.
    """
    df_perc = dc(df_sensitivity)
    
    # Iterate over each sensitivity case
    for sc, case in df_dct.items():
        for desc, data in case.items():
            if desc != 'total':
                idx = [i for i in data.index][0]
                lst1 = data.loc[idx].to_list()
                lst2 = df_be.loc[idx].to_list()
                tot_sens = 0
                tot = 0
                
                # Calculate the total sensitivity and total values
                for k, n in enumerate(lst1):
                    tot_sens += n
                    tot += lst2[k]
                
                # Calculate the percentage change
                p = (tot_sens - tot) / tot
                df_perc.at[desc, sc] = p

    results_df = dc(df_perc)
    
    # Update the results DataFrame with total values and format
    for col in results_df.columns:
        for idx, row in results_df.iterrows():
            if 'total' not in idx:
                results_df.at['total', col] += row[col]
            if row[col] == 0:
                row[col] = '-'

    return results_df

def calculate_sensitivity_values(variables, autoclave_gwp, case):
    """
    Calculates the sensitivity values based on the given variables and autoclave GWP.

    Parameters:
    variables (tuple): A tuple containing database name, df_GWP, db type, save directory, impact category, and flows.
    autoclave_gwp (float): The GWP impact of the autoclave.

    Returns:
    pd.DataFrame: DataFrame with sensitivity percentage results.
    """
    database_name, df_GWP, db_type, save_dir, impact_category, flows = variables

    # Define flow legend based on the database name
    if 'case1' in database_name:
        flow_legend = [
            'H2I',
            'H2R',
            'ASC',
            'ASW',
            'H4I',
            'H4R',
            'ALC',
            'ALW'
        ]
    else:
        flow_legend = ['SUD', 'MUD']

    # Create the dataframe for min and max values
    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = rfig.process_categorizing(df_GWP, case, flow_legend, columns)

    # Organize the break-even data
    df_be = rfig.break_even_orginization(df_stack_updated, database_name)

    # Find the minimum and maximum value of the sensitivity analysis
    if 'case1' in database_name:
        df, val_dct, idx_sens, col_to_df = c1.case1_initilazation(df_be)
        df_dct = c1.uncertainty_case1(df, val_dct, df_be, totals_df, idx_sens, col_to_df)
        return results_dataframe(df, df_dct, df_be)
    elif 'case2' in database_name:
        df, val_dct, idx_sens, col_to_df = c2.case2_initilazation(df_be, db_type, autoclave_gwp)
        df_dct = c2.uncertainty_case2(val_dct, df_be, df)
        return results_dataframe(df, df_dct, df_be)

def save_sensitivity_to_excel(variables, case, autoclave_gwp_dct):
    """
    Saves the sensitivity analysis results to an Excel file.

    Parameters:
    variables (tuple): A tuple containing database name, df_GWP, db type, save directory, impact category, and flows.
    path (str): The path to the results folder.
    autoclave_gwp_dct (dict): Dictionary containing autoclave GWP impacts.

    Returns:
    pd.DataFrame: DataFrame with sensitivity percentage results.
    """
    identifier = variables[0]
    save_dir = variables[3]
    df_sens = calculate_sensitivity_values(variables, autoclave_gwp_dct, case)

    results_path = join_path(save_dir, f"sensitivity_{identifier}.xlsx")
    
    if os.path.exists(results_path):
        try:
            # Try to load the existing workbook
            book = load_workbook(results_path)
            with pd.ExcelWriter(results_path, engine='openpyxl', mode='a') as writer:
                writer.book = book
                df_sens.to_excel(writer, sheet_name=identifier, index=True)
        except Exception as e:
            print(f"Error loading existing workbook: {e}")
            # If there's an error loading the workbook, create a new one
            with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
                df_sens.to_excel(writer, sheet_name=identifier, index=True)
    else:
        # If the file does not exist, create a new one
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_sens.to_excel(writer, sheet_name=identifier, index=True)
    
    print(f"Saved successfully to {results_path} in sheet {identifier}")

    return df_sens

def obtain_case1_autoclave_gwp(variables, path):
    """
    Obtains the GWP impact of the autoclave for case1 based on the given variables and path.

    Parameters:
    variables (dict): Dictionary containing variable tuples for different cases.
    path (str): The path to the results folder.

    Returns:
    dict: Dictionary with case identifiers as keys and their respective autoclave GWP impacts as values.
    """
    autoclave_gwp_dct = {}
    for key, item in variables.items():
        try:
            if '1' in key:
                # Calculate the autoclave GWP impact for case1 and store it in the dictionary
                autoclave_gwp_dct[f'case2_{item[2]}'] = autoclave_gwp_impact_case1(item, path)
                autoclave_gwp_dct[item[0]] = ''
        except KeyError as e:
            print(e)
            
    return autoclave_gwp_dct

def iterative_save_sensitivity_results_to_excel(path, case):
    """
    Iteratively saves the sensitivity analysis results to an Excel file for each case.

    Parameters:
    variables (dict): Dictionary containing variable tuples for different cases.
    path (str): The path to the results folder.

    Returns:
    tuple: A tuple containing two dictionaries:
        - df_dct: Dictionary with case identifiers as keys and their respective sensitivity DataFrames as values.
        - df_dct_be: Dictionary with case identifiers as keys and their respective break-even DataFrames as values.
    """
    variables = break_even_initialization(path, 'recipe')
    # Obtain the autoclave GWP impacts for case1
    autoclave_gwp_dct = obtain_case1_autoclave_gwp(variables, path)
    df_dct = {}

    # Iterate over each case in the variables dictionary
    for key, item in variables.items():
        if '1' in key:
            # Save sensitivity results for case1
            df_sens = save_sensitivity_to_excel(item, case, autoclave_gwp_dct[key])
        elif '2' in key:
            # Save sensitivity results for case2
            df_sens = save_sensitivity_to_excel(item, case, autoclave_gwp_dct[key])
        
        # Store the sensitivity DataFrame in the dictionary
        df_dct[key] = df_sens

    return df_dct
