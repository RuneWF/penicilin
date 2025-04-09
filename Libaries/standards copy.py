import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FileHandling():
    def __init__(self, file_path, folder_name):
        self.file_path = file_path
        self.folder_name = folder_name
        self.save_dir = ""

    # Function to create a results folder with a specified path and name
    def results_folder(self, db=None):
        # Determine the save directory and folder name based on the presence of a database name
        if db:
            self.save_dir = f'{self.file_path}/{self.folder_name}_{db}'
            temp = f'{self.folder_name}_{db}'
        else:
            self.save_dir = f'{self.file_path}/{self.folder_name}'
            temp = f'{self.folder_name}'

        try:
            # Check if the directory already exists
            if os.path.exists(self.save_dir):
                pass
            else:
                # Create the directory if it doesn't exist
                os.makedirs(self.save_dir, exist_ok=True)
                print(f'The folder {temp} is created')
        
        except (OSError, FileExistsError) as e:
            # Handle potential UnboundLocalError
            print('Error occurred')

        return self.save_dir

    def join_path(path1, path2):
        return os.path.join(path1, path2)

    def data_paths(self, path):
        # Path to where the code is stored
        path_github = self.join_path(path, r'RA\penicilin')
        # Specifying the LCIA method

        ecoinevnt_paths = {'ev391apos' : self.join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_apos_ecoSpold02\datasets"),
                        'ev391consq' :   self.join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_consequential_ecoSpold02\datasets"),
                        'ev391cutoff' :  self.join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets")}
        
        system_path = self.join_path(path_github, r'data\database.xlsx')

        return path_github, ecoinevnt_paths, system_path

    # saving the LCIA results to excel
    def save_LCIA_results(df, file_name, sheet_name):
        # Convert each cell to a JSON string for all columns
        df_save = df.map(lambda x: json.dumps(x) if isinstance(x, list) else x)

        # Save to Excel
        with pd.ExcelWriter(file_name) as writer:
            df_save.to_excel(writer, sheet_name=sheet_name, index=True, header=True)

        print(f'DataFrame saved successfully to {file_name}')


    # Function to import the LCIA results from excel
    def import_LCIA_results(file_name, impact_category):
        
        if type(impact_category) == tuple:
            impact_category = [impact_category]
        
        # Reading from Excel
        df = pd.read_excel(io=file_name, index_col=0)

        # Convert JSON strings back to lists for all columns
        df = df.map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)

        # Updating column names
        df.columns = impact_category

        # Return the imported dataframe
        return df

    def color_range():
        cmap = plt.get_cmap('Accent')
        return [cmap(i) for i in np.linspace(0, 1, 9)]
