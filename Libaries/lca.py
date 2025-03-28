import pandas as pd
import copy
import re

# Import BW25 packages
import bw2data as bd
import brightway2 as bw 

# Importing self-made libraries
from standards import *
import results_figures as rfig
import database_manipulation as dm

class lca():
    # initialization of all the required parameters
    def __init__(self, path, matching_database, database_name, lcia_meth='recipe', bw_project="Penicillin"):
        """
        Initialize the LCA class with the given parameters.
        
        :param path: Path to the data directory
        :param matching_database: Name of the matching database
        :param database_name: Name of the database
        :param lcia_meth: LCIA method (default is 'recipe')
        :param bw_project: Brightway project name (default is 'Penicillin')
        """
        self.path = path
        self.path_github, self.ecoinevnt_paths, self.system_path = data_paths(self.path)
        self.matching_database = matching_database
        self.database_name = database_name
        self.lcia_meth = lcia_meth
        self.bw_project = bw_project
        self.file_name = []
        self.file_name_unique_process = []
        self.data_info = []
        self.flow = []

        # Store the flow and other information in the respective dictionaries
        self.dir_temp = results_folder(join_path(self.path_github, 'results'), "LCIA")
        self.file_name = join_path(self.dir_temp, "LCIA_results.xlsx")
        self.file_name_unique_process = join_path(self.dir_temp, "LCIA_results_unique.xlsx")

        # LCIA method variable
        self.all_methods = []

        # initialization of the functional unit parameter
        self.func_unit = {}

        # initialization of the dataframes for the LCIA parameter
        self.df_mid_end = pd.DataFrame()
        self.df_midpoint = pd.DataFrame()
        self.df_endpoint = pd.DataFrame()

    def initialization(self):
        """
        Initialize the Brightway project and set up the database.
        
        :return: Tuple containing file information and initialization details
        """
        # Set the current Brightway project
        bd.projects.set_current(self.bw_project)
        dm.database_setup(self.path, self.matching_database)
        # dm.remove_bio_co2_recipe()
        # dm.add_new_biosphere_activities(self.bw_project, self.path)
        
        # Get the database
        db = bd.Database(self.database_name)
        
        # Check if the database is case1
        for act in db:
            temp = act['name']
            # Check if the flow is valid and add to the flow list
            if "pill" in temp or ("vial" in temp and "sc" in temp) or 'combined' in temp:
                self.flow.append(temp)
        
        self.flow.sort()

        self.data_info = [self.bw_project, self.database_name, self.flow, self.lcia_meth]

        # Create a list of the collected information
        self.file_info = [self.dir_temp, self.file_name, self.file_name_unique_process]
        
        return self.file_info, self.data_info


    # Function to obtain the LCIA category to calculate the LCIA results
    def lcia_impact_method(self):
        # Using H (hierachly) due to it has a 100 year span
        # Obtaining the midpoint categpries and ignoring land transformation (Land use still included)
        dm.remove_bio_co2_recipe()
        midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
        self.all_methods = midpoint_method + endpoint_method

        # Returning the selected LCIA methods
        return self.all_methods

    # Function to initialize parameters for the LCIA calculations
    def LCA_initialization(self):
        # Setting up an empty dictionary with the flows as the key
        procces_keys = {key: None for key in self.flows}

        size = len(self.flows)
        db = bd.Database(self.database_name)
        
        # Iterate over the database to find matching processes
        for act in db:
            for proc in range(size):
                if act['name'] == self.flows[proc]:
                    procces_keys[act['name']] = act['code']

        process = []

        # Obtaining all the subprocesses in a list 
        for key, item in procces_keys.items():
            try:
                process.append(db.get(item))
            except KeyError:
                print(f"Process with key '{item}' not found in the database '{db}'")
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
        return self.func_unit, lca.lcia_impact_method()

    # Function to seperate the midpoint and endpoint results for ReCiPe
    def recipe_dataframe_split(self):
        # Obtaining the coluns from the dataframe
        col_df = list(self.df_mid_end.columns)

        # Seperating the dataframe into one for midpoint and another for endpoint
        self.df_midpoint = self.df_mid_end[col_df[:-3]]
        self.df_endpoint = self.df_mid_end[col_df[-3:]]
        
        return self.df_midpoint, self.df_endpoint

# Function to create two dataframes, one where each subprocess' in the process are summed 
# and the second is scaling the totals in each column to the max value
def dataframe_element_scaling(df):
    # Creating a deep copy of the dataframe to avoid changing the original dataframe
    df_tot = copy.deepcopy(df)

    # Obating the sum of each process for each given LCIA category
    for col in range(df.shape[1]):  # Iterate over columns
        for row in range(df.shape[0]):  # Iterate over rows
            tot = 0
            for i in range(len(df.iloc[row,col])):
                tot += df.iloc[row,col][i][1]
            df_tot.iloc[row,col] = tot

    df_cols = df_tot.columns
    df_cols = df_cols.to_list()

    df_scaled = copy.deepcopy(df_tot)

    # Obtaing the scaled value of each LCIA results in each column to the max
    for i in df_cols:
        scaling_factor = max(abs(df_scaled[i]))
        for _, row in df_scaled.iterrows():
            row[i] /= scaling_factor

    return df_tot, df_scaled

# Obtaining the uniquie elements to determine the amount of colors needed for the plots
def unique_elements_list(database_name):
    category_mapping = rfig.category_organization(database_name)
    unique_elements = []
    for item in category_mapping.values():
        for ilst in item:
            unique_elements.append(ilst)

    return unique_elements

def rearrange_dataframe_index(df, database):
    # Initialize a dictionary to store the new index positions
    idx_dct = {}
    idx_lst = list(df.index)
    
    if len(idx_lst) == 3:
        print(idx_lst)
        # Define the new order of the index
        plc_lst = [2,   # new placement of the first index
                   0,   # new placement of the second index
                   1]   # new placement of the third index
                   

        # Assign the new order to the index dictionary
        for plc, idx in enumerate(df.index):
            idx_dct[idx] = plc_lst[plc]
            
        # Create the new index list
        idx_lst = [''] * len(idx_dct)
        for key, item in idx_dct.items():
            idx_lst[item] = key

        # Get the impact categories from the dataframe columns
        impact_category = df.columns
        
        # Create a new dataframe with the rearranged index
        df_rearranged = pd.DataFrame(0, index=idx_lst, columns=impact_category, dtype=object)

        # Rearrange the dataframe according to the new index
        for icol, col in enumerate(impact_category):
            for row_counter, idx in enumerate(df_rearranged.index):
                rearranged_val = df.at[idx, col] # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at
                df_rearranged.iloc[row_counter, icol] = rearranged_val

        return df_rearranged
    else:
        # If the database is not 'case1', return the original dataframe
        return df

def dataframe_results_handling(df, database_name, plot_x_axis_all, lcia_meth):
    # Rearrange the dataframe index based on the database name
    df_rearranged = rearrange_dataframe_index(df, database_name)

    # Check if the LCIA method is ReCiPe
    if 'recipe' in lcia_meth.lower():
        # Split the dataframe into midpoint and endpoint results
        df_midpoint, df_endpoint = lca.recipe_dataframe_split(df_rearranged)
        
        # Extract the endpoint categories from the plot x-axis
        plot_x_axis_end = plot_x_axis_all[-3:]
        
        # Extract the midpoint categories from the plot x-axis
        ic_mid = plot_x_axis_all[:-3]
        plot_x_axis_mid = []
        
        # Process each midpoint category to create a shortened version for the plot x-axis
        for ic in ic_mid:
            string = re.findall(r'\((.*?)\)', ic)
            if 'ODPinfinite' in string[0]:
                string[0] = 'ODP'
            elif '1000' in string[0]:
                string[0] = 'GWP'
            plot_x_axis_mid.append(string[0])

        # Return the processed dataframes and plot x-axis labels
        return [df_midpoint, df_endpoint], [plot_x_axis_mid, plot_x_axis_end]

    else:
        # If the LCIA method is not ReCiPe, use the rearranged dataframe as is
        df_res = df_rearranged
        plot_x_axis = plot_x_axis_all

        # Return the processed dataframe and plot x-axis labels
        return df_res, plot_x_axis
