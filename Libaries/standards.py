import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to create a results folder with a specified path and name
def results_folder(path, name, db=None):
    # Determine the save directory and folder name based on the presence of a database name
    if db is not None:
        save_dir = f'{path}/{name}_{db}'
        temp = f'{name}_{db}'
    else:
        save_dir = f'{path}/{name}'
        temp = f'{name}'

    try:
        # Check if the directory already exists
        if os.path.exists(save_dir):
            pass
        else:
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            print(f'The folder {save_dir} is created')
    
    except (OSError, FileExistsError) as e:
        # Handle potential UnboundLocalError
        print('Error occurred')
    return save_dir

def join_path(path1, path2):
    return os.path.join(path1, path2)

def data_paths(path):
    # Path to where the code is stored
    path_github = join_path(path, r'RA\penicilin')

    ecoinevnt_paths = {'ev391apos' : join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_apos_ecoSpold02\datasets"),
                    'ev391consq' :   join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_consequential_ecoSpold02\datasets"),
                    'ev391cutoff' :  join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets")}
    
    system_path = join_path(path_github, r'data\database.xlsx')

    
    return path_github, ecoinevnt_paths, system_path

# saving the LCIA results to excel
def save_LCIA_results(df, file_name, sheet_name):
    with pd.ExcelWriter(file_name) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)

# Function to import the LCIA results from excel
def import_LCIA_results(file_name, impact_category):
    if type(impact_category) == tuple:
        impact_category = [impact_category]
    
    # Reading from Excel
    df = pd.read_excel(io=file_name, index_col=0)

    # Convert the cell values to original data type
    for col in df.columns:
        for idx, row in df.iterrows():
            cell_value = row[col]
            try:
                row[col] = ast.literal_eval(cell_value)
            except ValueError:
                row[col] = float(cell_value)

    # Updating column names
    df.columns = impact_category

    # Return the imported dataframe
    return df

def color_range(colorname="Accent", color_quantity=9):
    cmap = plt.get_cmap(colorname)
    return [cmap(i) for i in np.linspace(0, 1, color_quantity)]