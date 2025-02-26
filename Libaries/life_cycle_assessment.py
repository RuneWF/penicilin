import pandas as pd
import json
import copy
import re
import os

# Import BW25 packages
import bw2data as bd
import brightway2 as bw 

# Importing self-made libraries
from standards import *
import results_figures as rfig
import non_bio_co2 as nbc
import import_ecoinvent_and_databases as ied


# Function to join two paths
def join_path(path1, path2):
    return os.path.join(path1, path2)

# Function to check if a flow is valid
def is_valid_flow(temp, flow):
    return (('H2' in temp or 'H4' in temp) and ('SU' in temp or 'REC' in temp) and temp not in flow)

def get_all_flows(path, lcia_meth='recipe', bw_project="Single Use vs Multi Use", case_range=range(1, 3)):
    # Set the current Brightway project
    bd.projects.set_current(bw_project)
    
    # Define the database types
    db_type = ['apos', 'consq', 'cut_off']

    # Initialize dictionaries to store various information
    flows = {}
    save_dir = {}
    database_name_dct = {}
    file_name = {}
    db_type_dct = {}
    flow_legend = {}
    file_name_unique_process = {}
    sheet_name = {}
    initialization = {}

    # Iterate over the specified case range
    for nr in case_range:
        # Iterate over each database type
        for tp in db_type:
            # Construct the database name
            database_name = f'case{nr}' + '_' + tp
            database_name_dct[database_name] = database_name
            
            # Get the database
            db = bd.Database(database_name)
            
            flow = []
            # Check if the database is case1
            if 'case1' in str(db):
                for act in db:
                    temp = act['name']
                    # Check if the flow is valid and add to the flow list
                    if is_valid_flow(temp, flow):
                        flow.append(temp)
                    elif 'alubox' in temp and '+' in temp and 'eol' not in temp.lower():
                        flow.append(temp)
                flow.sort()
                flow_leg = [
                    'H2S',
                    'H2R',
                    'ASC',
                    'ASW',
                    'H4S',
                    'H4R',
                    'ALC',
                    'ALW'
                ]
                sheet_name[database_name] = 'case1'
            # Check if the database is case2
            elif 'case2' in str(db):
                for act in db:
                    temp = act['name']
                    if temp == 'SUD' or temp == 'MUD':
                        flow.append(temp)
                    flow_leg = ['SUD', 'MUD']
                    sheet_name[database_name] = 'case2'
                flow.sort()
                flow.reverse()

            # Store the flow and other information in the respective dictionaries
            flows[database_name] = flow
            dir_temp = results_folder(join_path(path, 'results'), f"case{nr}")
            save_dir[database_name] = dir_temp
            file_name[database_name] = join_path(dir_temp, f"data_case{nr}_{tp}_recipe.xlsx")
            db_type_dct[database_name] = tp
            flow_legend[database_name] = flow_leg
            file_name_unique_process[database_name] = join_path(dir_temp, f"data_uniquie_case{nr}_{tp}_{lcia_meth}.xlsx")
            initialization[database_name] = [bw_project, database_name, flow, lcia_meth, tp]

    # Create a list of the collected information
    lst = [save_dir, file_name, flow_legend, file_name_unique_process, sheet_name]
    
    return lst, initialization

# Function to initialize the database and get all flows
def initilization(path, lcia_method, ecoinevnt_paths, system_path, bw_project="Single Use vs Multi Use", case_range=range(1, 3)):
    # Setup the database with the provided paths
    ied.database_setup(ecoinevnt_paths, system_path)

    # Get all flows and initialization parameters
    lst, initialization = get_all_flows(path, lcia_method, bw_project, case_range)
    save_dir, file_name, flow_legend, file_name_unique_process, sheet_name = lst

    # Return the collected information
    return flow_legend, file_name, sheet_name, save_dir, initialization, file_name_unique_process

# Function to obtain the LCIA category to calculate the LCIA results
def lcia_impact_method(method='recipe'):
    # Checking if the LCIA method is ReCiPe, and ignores difference between lower and upper case 
    if 'recipe' in method.lower():

        # Using H (hierachly) due to it has a 100 year span
        # Obtaining the midpoint categpries and ignoring land transformation (Land use still included)
        nbc.remove_bio_co2_recipe()
        all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
        for meth in endpoint:
            all_methods.append(meth)
            
        print('Recipe is selected')

    # Checking if EF is choses for the LCIA method
    elif 'ef' in method.lower():
        all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m) and "climate change:" not in str(m)]
        print('EF is selected')

    else:
        print('Select either EF or ReCiPe as the LCIA methos')
        all_methods = []

    # Returning the selected LCIA methods
    return all_methods

# Function to initialize parameters for the LCIA calculations
def LCA_initialization(database_name: str, flows: list, method: str) -> tuple:
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
    key_counter = 0

    # Obtaining all the subprocesses in a list 
    for key, item in procces_keys.items():
        try:
            process.append(db.get(item))
        except KeyError:
            print(f"Process with key '{item}' not found in the database '{db}'")
            process = None
        key_counter += 1
    
    # Obtaining the impact categories for the LCIA calculations
    impact_category = lcia_impact_method(method)
    
    # Obtaining a shortened version of the impact categories for the plots
    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        plot_x_axis[i] = impact_category[i][2]

    product_details = {}
    product_details_code = {}

    # Obtaining the subprocesses
    if process:
        for proc in process:
            product_details[proc['name']] = []
            product_details_code[proc['name']] = []

            for exc in proc.exchanges():
                if 'Use' in exc.output['name'] and exc['type'] == 'biosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})

                elif exc['type'] == 'technosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
        
    # Creating the Functional Unit (FU) to calculate for
    FU = {key: {} for key in product_details.keys()}
    for key, item in product_details.items():
        for idx in item:
            for m in idx.values():
                FU[key].update({m[1]: m[0]})
    
    print(f'Initialization is completed for {database_name}')
    return FU, impact_category

# saving the LCIA results to excel
def save_LCIA_results(df, file_name, sheet_name):
    # if type(impact_category) == tuple:
    #     impact_category = [impact_category]

    # Convert each cell to a JSON string for all columns
    df_save = df.map(lambda x: json.dumps(x) if isinstance(x, list) else x)

    # Save to Excel
    with pd.ExcelWriter(file_name) as writer:
        df_save.to_excel(writer, sheet_name=sheet_name, index=True, header=True)

    print('DataFrame with nested lists written to Excel successfully.')


# Function to import the LCIA results from excel
def import_LCIA_results(file_name, impact_category):
    
    if type(impact_category) == tuple:
        impact_category = [impact_category]
    
    # Reading from Excel
    df = pd.read_excel(io=file_name, index_col=0)

    # Convert JSON strings back to lists for all columns
    df = df.map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)
    # Setting the index to the flow

    # Updating column names
    df.columns = impact_category

    # Return the imported dataframe
    return df

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
    idx_lst = df.index
    
    # Check if the database is 'case1'
    if 'case1' in database:
        # Define the new order of the index
        plc_lst = [1, 0, 5, 4, 6, 7, 2, 3]

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
        df_res, df_endpoint = recipe_dataframe_split(df_rearranged)
        
        # Extract the endpoint categories from the plot x-axis
        plot_x_axis_end = plot_x_axis_all[-3:]
        
        # Extract the midpoint categories from the plot x-axis
        ic_mid = plot_x_axis_all[:-3]
        plot_x_axis = []
        
        # Process each midpoint category to create a shortened version for the plot x-axis
        for ic in ic_mid:
            string = re.findall(r'\((.*?)\)', ic)
            if 'ODPinfinite' in string[0]:
                string[0] = 'ODP'
            elif '1000' in string[0]:
                string[0] = 'GWP'
            plot_x_axis.append(string[0])

        # Return the processed dataframes and plot x-axis labels
        return [df_res, df_endpoint], [plot_x_axis, plot_x_axis_end]
    else:
        # If the LCIA method is not ReCiPe, use the rearranged dataframe as is
        df_res = df_rearranged
        plot_x_axis = plot_x_axis_all

        # Return the processed dataframe and plot x-axis labels
        return df_res, plot_x_axis
