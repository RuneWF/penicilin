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

def initilization(path, matching_database, database_name, lcia_meth='recipe', bw_project="Penicillin"):
    # Set the current Brightway project
    bd.projects.set_current(bw_project)
    path_github, ecoinevnt_paths, system_path = data_paths(path)
    dm.database_setup(path, matching_database)
    dm.remove_bio_co2_recipe()
    dm.add_new_biosphere_activities(bw_project, path)
    # Initialize dictionaries to store various information
    file_name = []
    file_name_unique_process = []
    initialization = []
    
    # Get the database
    db = bd.Database(database_name)
    
    flow = []
    # Check if the database is case1
    for act in db:
        temp = act['name']
        # Check if the flow is valid and add to the flow list
        if "Defined daily dose" in temp:
            flow.append(temp)
    
    flow.sort()

    # Store the flow and other information in the respective dictionaries
    dir_temp = results_folder(join_path(path_github, 'results'), "LCIA")
    file_name = join_path(dir_temp, "LCIA_results.xlsx")
    file_name_unique_process = join_path(dir_temp, "LCIA_results_uniquie.xlsx")
    initialization = [bw_project, database_name, flow, lcia_meth]

    # Create a list of the collected information
    file_info = [dir_temp, file_name, file_name_unique_process]
    
    return file_info, initialization

# Function to obtain the LCIA category to calculate the LCIA results
def lcia_impact_method():
    # Using H (hierachly) due to it has a 100 year span
    # Obtaining the midpoint categpries and ignoring land transformation (Land use still included)
    dm.remove_bio_co2_recipe()
    midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)] # Midpoint

    # Obtaining the endpoint categories and ignoring land transformation
    endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

    # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
    all_methods = midpoint_method + endpoint_method

    # Returning the selected LCIA methods
    return all_methods

# Function to initialize parameters for the LCIA calculations
def LCA_initialization(database_name: str, flows: list) -> tuple:
    # Setting up an empty dictionary with the flows as the key
    procces_keys = {key: None for key in flows}

    size = len(flows)
    db = bd.Database(database_name)
    
    # Iterate over the database to find matching processes
    for act in db:
        for proc in range(size):
            if act['name'] == flows[proc]:
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
    func_unit = {key: {} for key in product_details.keys()}
    for key, item in product_details.items():
        for idx in item:
            for m in idx.values():
                func_unit[key].update({m[1]: m[0]})
    
    print(f'Initialization is completed for {database_name}')
    return func_unit, lcia_impact_method()

# Function to seperate the midpoint and endpoint results for ReCiPe
def recipe_dataframe_split(df):
    # Obtaining the coluns from the dataframe
    col_df = df.columns
    col_df = col_df.to_list()

    # Seperating the dataframe into one for midpoint and another for endpoint
    df_midpoint = df[col_df[:-3]]
    df_endpoint = df[col_df[-3:]]
    
    return df_midpoint, df_endpoint

# Function to create two dataframes, one where each subprocess' in the process are summed 
# and the second is scaling the totals in each column to the max value
def dataframe_element_scaling(df):
    # Creating a deep copy of the dataframe to avoid changing the original dataframe
    # df_tot = copy.deepcopy(df)

    df_tot = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=object)

    for col in df.columns:
        for idx, dct in df.iterrows():
            # print(f"{dct[col]} type is {type(dct[col])}")
            for val in dct[col].values():
                df_tot.at[idx, col] += val


    df_scaled = copy.deepcopy(df_tot)

    # Obtaing the scaled value of each LCIA results in each column to the max
    for col in df_scaled.columns:
        scaling_factor = max(abs(df_scaled[col]))
        for _, row in df_scaled.iterrows():
            row[col] /= scaling_factor

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
    
    # Check if the database is 'case1'
    if len(idx_lst) == 3:
        print(idx_lst)
        # Define the new order of the index
        plc_lst = [2,   # new placement of the first index
                   0,   # new placement of the second index
                   1    # new placement of the third index
                   ]

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
    # df_rearranged = rearrange_dataframe_index(df, database_name)

    # Check if the LCIA method is ReCiPe
    if 'recipe' in lcia_meth.lower():
        # Split the dataframe into midpoint and endpoint results
        df_midpoint, df_endpoint = recipe_dataframe_split(df)
        
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
        df_res = df
        plot_x_axis = plot_x_axis_all

        # Return the processed dataframe and plot x-axis labels
        return df_res, plot_x_axis
