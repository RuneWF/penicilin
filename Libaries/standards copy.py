import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FileHandling():
    def __init__(self, file_path, folder_name):
        self.file_path = file_path
        self.folder_name = folder_name
        self.save_dir = ""

        self.path_github = ""
        self.ecoinevnt_paths = ""
        self.system_path = ""

    # Function to create a results folder with a specified path and name
    def results_folder(self, db=None):
        # Determine the save directory and folder name based on the presence of a database name
        if db:
            self.save_dir = f'{self.file_path}/{self.folder_name}_{db}'
        else:
            self.save_dir = f'{self.file_path}/{self.folder_name}'

        try:
            # Check if the directory already exists
            if os.path.exists(self.save_dir):
                pass
            else:
                # Create the directory if it doesn't exist
                os.makedirs(self.save_dir, exist_ok=True)
        
        except (OSError, FileExistsError) as e:
            # Handle potential UnboundLocalError
            print('Error occurred')

        return self.save_dir

    def join_path(path1, path2):
        return os.path.join(path1, path2)

    def data_paths(self):
        # Path to where the code is stored
        self.path_github = self.join_path(self.file_path, r'RA\penicilin')
        # Specifying the LCIA method

        self.ecoinevnt_paths = {'ev391apos' : self.join_path(self.file_path, r"4. semester\EcoInvent\ecoinvent 3.9.1_apos_ecoSpold02\datasets"),
                        'ev391consq' :   self.join_path(self.file_path, r"4. semester\EcoInvent\ecoinvent 3.9.1_consequential_ecoSpold02\datasets"),
                        'ev391cutoff' :  self.join_path(self.file_path, r"4. semester\EcoInvent\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets")}
        
        self.system_path = self.join_path(self.path_github, r'data\database.xlsx')

        return self.path_github, self.ecoinevnt_paths, self.system_path

    # saving the LCIA results to excel
    def save_LCIA_results(df, dataframe_path, sheet_name):
        with pd.ExcelWriter(dataframe_path) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)


    # Function to import the LCIA results from excel
    def import_LCIA_results(dataframe_path, impact_category):
        if type(impact_category) == tuple:
            impact_category = [impact_category]
        
        # Reading from Excel
        df = pd.read_excel(io=dataframe_path, index_col=0)

        # Convert the cell values to original data type
        for col in df.columns:
            for _, row in df.iterrows():
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

    # def all_functions(self):
    #     self.(results_folder)