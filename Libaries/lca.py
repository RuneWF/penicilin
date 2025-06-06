import pandas as pd
from copy import deepcopy as dc
import re

# Import BW25 packages
import bw2data as bd
import brightway2 as bw 

# Importing self-made libraries
from standards import *
import database_manipulation as dm

class LCA():
    # initialization of all the required parameters
    def __init__(self, path, matching_database, lcia_meth='recipe', bw_project="Penicillin"):
        """
        Initialize the LCA class with the given parameters.
        
        :param path: Path to the data directory
        :param matching_database: Name of the matching database
        :param database_name: Name of the database
        :param lcia_meth: LCIA method (default is 'recipe')
        :param bw_project: Brightway project name (default is 'Penicillin')
        """
        self.path = path
        self.path_github, self.ecoinevnt_paths, self.system_path, self.results_path = data_paths(self.path)
        self.matching_database = matching_database
        self.bw_project = bw_project
        bd.projects.set_current(self.bw_project)
        
        self.db_excel = pd.read_excel(self.system_path)
        self.database_name = self.db_excel.columns[1]
        
        self.db = bd.Database(self.database_name)
        self.lcia_meth = lcia_meth
       

        self.data_info = []
        self.flow = []

        # Store the flow and other information in the respective dictionaries
        self.dir_temp = results_folder(self.results_path, "LCIA")
        self.file_name = join_path(self.dir_temp, "LCIA_results.xlsx")
        self.file_name_unique_process = join_path(self.dir_temp, "LCIA_results_unique.xlsx")

        # LCIA method variable
        self.all_methods = []
        self.impact_categories_mid = []
        self.impact_categories_end = []

        # initialization of the functional unit parameter
        self.func_unit = {}

        # initialization of the dataframes for the LCIA parameter
        self.df_midpoint = pd.DataFrame()
        self.df_endpoint = pd.DataFrame()

        self.df_rearranged = pd.DataFrame()

    def initialization(self, reload=False, sensitivity=False):
        """
        Initialize the Brightway project and set up the database.
        
        :return: Tuple containing file information and initialization details
        """
        dm.database_setup(self.path, self.matching_database, bw_project=self.bw_project, reload=reload, sensitivty=sensitivity)
        # dm.remove_bio_co2_recipe()
        # dm.add_new_biosphere_activities(self.bw_project, self.path)

        
        # Check if the database is case1
        for act in self.db:
            temp = act['name']
            # Check if the flow is valid and add to the flow list
            if "defined system" in temp:
                self.flow.append(temp)
        
        self.flow.sort()

        self.data_info = [self.bw_project, self.database_name, self.flow, self.lcia_meth]

        # Create a list of the collected information
        self.file_info = [self.dir_temp, self.file_name, self.file_name_unique_process]
        
        return self.file_info, self.data_info


    # Function to obtain the LCIA category to calculate the LCIA results
    def lcia_impact_method(self):
        
        dm.remove_bio_co2_recipe()
        midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
        self.all_methods = midpoint_method + endpoint_method

        # Returning the selected LCIA methods
        return self.all_methods

    # Function to initialize parameters for the LCIA calculations
    def LCA_initialization(self, reload=False, sensitivity=False):
        # Setting up an empty dictionary with the flows as the key
        self.initialization(reload=reload, sensitivity=sensitivity)
        procces_keys = {key: None for key in self.flow}

        size = len(self.flow)
        
        # Iterate over the database to find matching processes
        for act in self.db:
            for proc in range(size):
                if act['name'] == self.flow[proc]:
                    procces_keys[act['name']] = act['code']

        process = []

        # Obtaining all the subprocesses in a list 
        for key, item in procces_keys.items():
            try:
                process.append(self.db.get(item))
            except KeyError:
                print(f"Process with key '{item}' not found in the database '{self.db}'")
                process = None
        
        # Obtaining the impact categories for the LCIA calculations
        
        product_details = {}
        product_details_code = {}

        # Obtaining the subprocesses
        if process:
            for proc in process:
                product_details[proc['name']] = []
                product_details_code[proc['name']] = []

                for exc in proc.exchanges():
                    if exc['type'] == 'technosphere' or ('Use' in exc.output['name'] and exc['type'] == 'biosphere'):
                        product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
            
        # Creating the Functional Unit (FU) to calculate for
        self.func_unit = {key: {} for key in product_details.keys()}
        for key, item in product_details.items():
            for idx in item:
                for m in idx.values():
                    self.func_unit[key].update({m[1]: m[0]})
        
        print(f'Initialization is completed for {self.database_name}')
        return self.func_unit

    # Function to seperate the midpoint and endpoint results for ReCiPe
    def recipe_dataframe_split(self, df):
        # Obtaining the coluns from the dataframe
        col_df = list(df.columns)

        # Seperating the dataframe into one for midpoint and another for endpoint
        self.df_midpoint = df[col_df[:-3]]
        self.df_endpoint = df[col_df[-3:]]
        
        return self.df_midpoint, self.df_endpoint

    # Function to create two dataframes, one where each subprocess' in the process are summed 
    # and the second is scaling the totals in each column to the max value
    def dataframe_element_scaling(self, df):
        # Creating a deep copy of the dataframe to avoid changing the original dataframe
        # df_tot = dc(df)
        # # Obating the sum of each process for each given LCIA category
        # for col in range(df.shape[1]):  # Iterate over columns
        #     for row in range(df.shape[0]):  # Iterate over rows
        #         tot = 0
        #         for val in df.iloc[row,col].values():
        #             tot += val
        #         df_tot.iloc[row,col] = tot

        df_cols = df.columns
        df_cols = df_cols.to_list()

        df_scaled = dc(df)

        # Obtaing the scaled value of each LCIA results in each column to the max
        for i in df_cols:
            scaling_factor = max(abs(df_scaled[i]))
            for _, row in df_scaled.iterrows():
                row[i] /= scaling_factor

        return df_scaled

    def x_label_text(self):
        impact_categories = self.lcia_impact_method()
         # Extract the endpoint categories from the plot x-axis
        self.impact_categories_end = impact_categories[-3:]
        
        # Extract the midpoint categories from the plot x-axis
        ic_mid = impact_categories[:-3]
        if 'recipe' in self.lcia_meth.lower():
            # Process each midpoint category to create a shortened version for the plot x-axis
            for ic in ic_mid:
                string = re.findall(r'\((.*?)\)', ic)
                if 'ODPinfinite' in string[0]:
                    string[0] = 'ODP'
                elif '1000' in string[0]:
                    string[0] = 'GWP'
                self.impact_categories_mid.append(string[0])

        return self.impact_categories_mid, self.impact_categories_end

    def dataframe_results_handling(self, df):
        # Rearrange the dataframe index based on the database name

        # Check if the LCIA method is ReCiPe
        if 'recipe' in self.lcia_meth.lower():
            # Split the dataframe into midpoint and endpoint results
            self.df_midpoint, self.df_endpoint = self.recipe_dataframe_split(df)

            # Return the processed dataframes and plot x-axis labels
            return [self.df_midpoint, self.df_endpoint]

        else:
            # Return the processed dataframe
            return df
