# Importing libaries to perform the code
import pandas as pd
from copy import deepcopy as dc
import re
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import BW25 packages
import bw2data as bd
import brightway2 as bw 
import bw2io as bi


class main():
    # initialization of all the required parameters
    def __init__(self, desktop_name="Desktop", matching_database="ev391cutoff", bw_project="Penicillin"):
        self.desktop_path = Path.home() / desktop_name
        self.path_github, self.system_path, self.results_path = self.data_paths()
        self.matching_database = matching_database
        self.bw_project = bw_project
        bd.projects.set_current(self.bw_project)
        
        self.db_excel = pd.read_excel(self.system_path)
        self.database_name = self.db_excel.columns[1]
        
        self.db = bd.Database(self.database_name)
       

        self.data_info = []
        self.flow = []

        # Store the flow and other information in the respective dictionaries
        self.LCIA_folder = self.results_folder(self.results_path, "LCIA")
        self.LCIA_results = self.join_path(self.LCIA_folder, "LCIA_results.xlsx")
        self.LCIA_results_unique_process = self.join_path(self.LCIA_folder, "LCIA_results_unique.xlsx")

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

        self.reload_database_has_been_called = False
        self.import_excel_database_to_brightway_has_been_called = False


    # Function to obtain the LCIA category to calculate the LCIA results
    def lcia_impact_method(self):
        
        self.remove_bio_co2_recipe()
        # Obtaining the midpoint categories and ignoring land transformation
        midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
        self.all_methods = midpoint_method + endpoint_method

        # Returning the selected LCIA methods
        return self.all_methods

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
    # def dataframe_cell_scaling(self, df):

    #     df_cols = df.columns
    #     df_cols = df_cols.to_list()

    #     df_scaled = dc(df)

    #     # Obtaing the scaled value of each LCIA results in each column to the max
    #     for i in df_cols:
    #         scaling_factor = max(abs(df_scaled[i]))
    #         for _, row in df_scaled.iterrows():
    #             row[i] /= scaling_factor

    #     return df_scaled

    # def dataframe_results_handling(self, df):
    #     # Rearrange the dataframe index based on the database name

    #     # Split the dataframe into midpoint and endpoint results
    #     self.df_midpoint, self.df_endpoint = self.recipe_dataframe_split(df)

    #     # Return the processed dataframes and plot x-axis labels
    #     return [self.df_midpoint, self.df_endpoint]


    # Function to create a results folder with a specified path and name
    def results_folder(self, path, name, database_name=None):
        # Determine the save directory and folder name based on the presence of a database name
        if database_name:
            save_dir = Path(f'{path}/{name}_{database_name}')
        else:
            save_dir = Path(f'{path}/{name}')

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
            print(f'Error occurred : {e}')
        return save_dir

    def join_path(self, path1, *argv):
        path = Path(path1)
        for arg in argv:
            path = path / arg
        return path

    def data_paths(self):
        # Path to where the code is stored
        current_directory = str(Path.cwd())
        main_folder_path = Path(current_directory.replace(f"\{str(Path.cwd().name)}",""))

        database_path = self.join_path(main_folder_path, r'data\database.xlsx')

        results_path = self.join_path(self.desktop_path, r'RA\penicillin_results')
        
        return main_folder_path, database_path, results_path

    # saving the LCIA results to excel
    def save_LCIA_results(self, df, file_name):
        with pd.ExcelWriter(file_name) as writer:
            df.to_excel(writer, index=True, header=True)

    # Function to import the LCIA results from excel
    def import_LCIA_results(self, file_name, columns):
        if type(columns) == tuple:
            columns = [columns]
        
        # Reading from Excel
        df = pd.read_excel(io=file_name, index_col=0)

        # Convert the cell values to original data type
        for col in df.columns:
            for _, row in df.iterrows():
                cell_value = row[col]
                try:
                    row[col] = ast.literal_eval(cell_value)
                except ValueError:
                    row[col] = float(cell_value)
        try:
            # Updating column names
            df.columns = columns
        except ValueError:
            pass

        # Return the imported dataframe
        return df

    def color_range(self, colorname="Accent", color_quantity=9):
        cmap = plt.get_cmap(colorname)
        return [cmap(i) for i in np.linspace(0, 1, color_quantity)]

    def plot_dimensions(self, subfigure=False):
        if subfigure:
            plt.rcParams.update({
                'font.size': 14,      # General font size
                'axes.titlesize': 16, # Title font size
                'axes.labelsize': 14, # Axis labels font size
                'legend.fontsize': 13 # Legend font size
            }) 
        else:
            plt.rcParams.update({
                'font.size': 11,      # General font size
                'axes.titlesize': 13, # Title font size
                'axes.labelsize': 11, # Axis labels font size
                'legend.fontsize': 9 # Legend font size
            }) 

        dpi = 300
        width_in = 2244 / dpi
        height_in = width_in * 0.65

        return width_in, height_in, dpi
    
    def import_excel_database_to_brightway(self, data):
        # Save the data to a temporary file that can be used by ExcelImporter
        temp_path = self.join_path(self.path_github, r"data\LCI_temp.xlsx")

        data.to_excel(temp_path, index=False)
        try:
            print("database name = ", data.columns[1])
            imp = bi.ExcelImporter(temp_path)  # the path to your inventory excel file
            imp.apply_strategies()
        
            # Loop through each database and match
            print(f"Matching database: {self.matching_database}")
            imp.match_database(self.matching_database, fields=('name', 'unit', 'location', 'reference product'))
            if list(imp.unlinked):
                print(f"Unlinked items after matching {self.matching_database}: {list(imp.unlinked)}")


            # Generate statistics and write results
            imp.statistics()
            imp.write_excel(only_unlinked=True)
            unlinked_items = list(imp.unlinked)
            imp.write_database()

            # Print unlinked items if needed
            if unlinked_items:
                print(unlinked_items)
            print("database name = ",data.columns[1])
            self.import_excel_database_to_brightway_has_been_called = True
            
        except ValueError:
            print(data.columns[1])
    
    def reload_database(self, proj_database, reload, sheet):
        self.reload_database_has_been_called = True
        if reload:
            
            data = pd.read_excel(self.system_path, sheet_name=sheet)

            # Removing the old database
            db_old = bd.Database(proj_database)
            db_old.deregister()

            self.import_excel_database_to_brightway(data)

    def extract_excel_sheets(self):
        # Use a context manager to open the Excel file
        with pd.ExcelFile(self.system_path) as excel_file:
            # Get the sheet names
            sheet_names = excel_file.sheet_names
        
        sheets_to_import = []
        for sheet in sheet_names:
            if self.matching_database in sheet:
                sheets_to_import.append(sheet)
        return sheets_to_import

    def extract_database(self, data, reload, sheet):
        proj_database_str = data.columns[1]
        if proj_database_str not in bd.databases:
            self.import_excel_database_to_brightway(data)
        # Reload databases if needed
        self.reload_database(proj_database_str, reload, sheet)

    def is_db_in_project(self, sheets_to_import):
        import_db = False

        if isinstance(sheets_to_import,str):
            sheets_to_import = [sheets_to_import]

        for sheet in sheets_to_import:
            data = pd.read_excel(self.system_path, sheet_name=sheet)
            if data.columns[1] not in bd.databases:
                print(f"{data.columns[1]} not in {bd.projects.current}")
                import_db = True
                break

        return import_db

    def import_ecoinvent_database(self):
        ecoinevnt_term = {
            'ev391apos' : r"ecoinvent 3.9.1_apos_ecoSpold02\datasets",
            'ev391consq' :  r"ecoinvent 3.9.1_consequential_ecoSpold02\datasets",
            'ev391cutoff' :  r"ecoinvent 3.9.1_cutoff_ecoSpold02\datasets"
            }
        
        
        base_path = Path.home()

        return list(base_path.rglob(ecoinevnt_term[self.matching_database]))[0]
    
    def import_databases(self, reload=False, sensitivty=False):
        # Check if Ecoinvent databases are already present
        if self.matching_database not in bd.databases:
            # Import Ecoinvent database
            ei = bi.SingleOutputEcospold2Importer(dirpath=self.import_ecoinvent_database(), db_name=self.matching_database)
            ei.apply_strategies()
            ei.statistics()
            ei.write_database()

        sheets_to_import = self.extract_excel_sheets()

        if sensitivty is False:
            # Extracting the first sheet only
            sheets_to_import = sheets_to_import[0]

        data = pd.read_excel(self.system_path, sheet_name=sheets_to_import)

        if reload:
            if isinstance(data, pd.DataFrame):
                self.extract_database(data, reload, sheets_to_import)
            elif isinstance(data, dict):
                for sheet, df in data.items():
                    # print(df.columns[1])
                    self.import_excel_database_to_brightway_has_been_called = False
                    self.extract_database(df, reload, sheet)
            else:
                print("Wrong data format")
        
    def remove_bio_co2_recipe(self):
        all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H)' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H)' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        method_name_new_mid = all_methods[0][0] + ' - no biogenic'
        method_name_new_end = endpoint[0][0] + ' - no biogenic'

        # Checking if the method exist
        if method_name_new_mid not in [m[0] for m in list(bd.methods)] or method_name_new_end not in [m[0] for m in list(bd.methods)]:
            # Combining mid- and endpoint method into one method
            for method in endpoint:
                all_methods.append(method)

            # Dictionary to store new method names
            new_methods = {}
            check = {}

            # For loop for setting bio-genic CO2 factor to 0
            for metod in all_methods:
                recipe_no_bio_CO2 = []  # Temporary storage for filtered CFs
                # Ensuring that the method only will be handled once
                if metod[1] not in check.keys():
                    method_unit = bw.Method(metod)
                    unit = method_unit.metadata.get('unit', 'No unit found')
                    
                    check[metod[1]] = None
                    method = bw.Method(metod)
                    cf_data = method.load()
                    # filtering the CFs
                    for cf_name, cf_value in cf_data:
                        flow_object = bw.get_activity(cf_name)
                        flow_name = flow_object['name'] if flow_object else "Unknown Flow"
                        if 'non-fossil' not in flow_name:
                            recipe_no_bio_CO2.append((cf_name, cf_value))
                        else:
                            recipe_no_bio_CO2.append((cf_name, 0))

                    # registering the new method for later use
                    new_metod = (metod[0] + ' - no biogenic', metod[1], metod[2])
                    new_method_key = new_metod
                    new_method = bw.Method(new_method_key)
                    new_method.metadata['unit'] = unit
                    new_method.register()
                    new_method.write(recipe_no_bio_CO2)

                    # Storing the new method
                    new_methods[metod] = new_method_key
                    print(f"New method created: {new_method_key} with {len(recipe_no_bio_CO2)} CFs")

    def create_new_bs3_activities(self, df):

        biosphere3 = bw.Database('biosphere3')

        # Get the list of columns from the dataframe
        new_biosphere3_flow = {}
        biosphere3_codes = {}
        
        # Iterate over each column in the dataframe
        for col in df.columns:
            # Define the new emission flow
            new_biosphere3_flow[col] = {
                'name': col,
                'categories': ('water',),
                'unit': 'kilogram',
                'type': 'emission',
                'location': ""
            }
            # Create a unique code for the new flow
            biosphere3_codes[col] = f"self-made-{col}-1"
            # Check if the code already exists in biosphere3
            if biosphere3_codes[col] not in [act['code'] for act in biosphere3]:
                # Create and save the new activity in biosphere3
                new_flow_entry = biosphere3.new_activity(code=biosphere3_codes[col], **new_biosphere3_flow[col])
                new_flow_entry.save()
                print(f"{col} is added to biosphere3")
            # else:
            #     print(f"{col} is present in biosphere3")
        
        return biosphere3_codes

    def ecotoxicity_impact_category(self):
        # Get all methods related to ReCiPe 2016 midpoint/endpoint (H) without biogenic
        midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)]
        endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)]

        all_methods = midpoint_method + endpoint_method

        # Filter methods related to ecotoxicity
        meth_ecotoxicity = [m for m in all_methods if "ecotoxicity" in m[1] or "ecosystem quality" in m[1]]

        return meth_ecotoxicity

    def add_activity_to_biosphere3(self, df, act_dct):
        # Filter methods related to ecotoxicity
        ecotoxicity_methods = self.ecotoxicity_impact_category()
        
        # Iterate over each method in ecotoxicity_methods
        for method in ecotoxicity_methods:
            new_cfs = []
            method_key = (method[0], method[1], method[2])
            method_obj = bw.Method(method_key)
            
            try:
                # Load existing characterization factors (CFs) for the method
                existing_cfs = method_obj.load()
                
                # Iterate over each bioflow in the dataframe columns
                for bioflow in df.columns:
                    # Iterate over each row in the dataframe
                    for impact_category, row in df.iterrows():
                        # Check if the impact category matches the method
                        if impact_category in method[1]:
                            new_cf = (((act_dct[bioflow][0], act_dct[bioflow][1])), row[bioflow])
                            
                            # Check if the new CF is already present in the existing CFs
                            if new_cf not in existing_cfs:
                                # Append the new CF to the list
                                new_cfs.append(new_cf)
                
                # Combine existing CFs with new CFs
                updated_cfs = existing_cfs + new_cfs
                
                if new_cfs:
                    # Save the updated method
                    method_obj.write(updated_cfs)
                    print(f"{method[1]} update complete.")

            except Exception as e:
                print(f"An error occurred while processing {method[1]}: {e}")

    def add_new_biosphere_activities(self, bw_project):
        path_github = self.path_github

        # Set the current Brightway2 project
        bd.projects.set_current(bw_project)
        
        # Load the new impacts data from the specified path
        data_path = self.join_path(path_github, r"data\new_impacts.xlsx")
        df = pd.read_excel(data_path, index_col=0)
        
        # Create new biosphere activities and get their codes
        activity_codes = self.create_new_bs3_activities(df)

        # Load the biosphere3 database
        biosphere3 = bw.Database('biosphere3')

        act_dct = {}
        # Map the new activities to their names
        for code in activity_codes.values():
            for bs3 in biosphere3:
                if code in bs3[1]:
                    act_dct[bs3['name']] = bs3

        # Add the new activities to the biosphere3 database
        self.add_activity_to_biosphere3(df, act_dct)

    def database_setup(self, reload, sensitivty=False):
        self.import_excel_database_to_brightway_has_been_called = False

        # Set the current Brightway project
        bd.projects.set_current(self.bw_project)
        

        # Check if biosphere database is already present
        if "biosphere3" not in bd.databases:
            bi.bw2setup()

        if isinstance(self.matching_database, str):
            self.import_databases(reload, sensitivty)
        elif isinstance(self.matching_database, list):
            for _ in self.matching_database:
                self.import_databases(reload, sensitivty)

        if self.reload_database_has_been_called is False:
            self.add_new_biosphere_activities(self.bw_project)

    def remove_empty_rows(self, df):
        temp = df.isna()

        bio_idx = []
        for idx, val in temp.iterrows():
            if val["Reference Flow"]:
                bio_idx.append(idx)

        df = df.drop(bio_idx)
        return df

    def create_LCI_tables(self):
        # Path to the LCI table template Excel file
        lci_table_template_path = self.join_path(self.results_path, r"LCI_tables.xlsx")

        # Read the template into a DataFrame
        lci_table_template_df = pd.read_excel(lci_table_template_path)

        dct = {}
        # Iterate over each activity in the database
        for act in self.db:
            # Reload the template for each activity to avoid overwriting
            lci_table_template_df = pd.read_excel(lci_table_template_path)

            # Create a new index to fit all exchanges plus one for the production exchange
            new_idx = range(0, len(act.exchanges()) + 1)
            lci_table_template_df = lci_table_template_df.reindex(new_idx)
            bio_idx_start = 0

            # Fill in the template with technosphere exchanges
            for i, exc in enumerate(act.exchanges()):
                i += 1  # Offset for production exchange at index 0
                for col in lci_table_template_df.columns:
                    lci_table_template_df[col] = lci_table_template_df[col].astype('object')
                    search_term = col.lower()
                    if "Provider" in col:
                        search_term = "name"
                    if "Reference Flow" in col:
                        search_term = "reference product"
                    # Fill in the production exchange at the top
                    if i == 1 and "production" in exc["type"]:
                        lci_table_template_df.loc[0, col] = exc[search_term]
                    # Fill in technosphere exchanges
                    if "techno" in exc["type"]:
                        lci_table_template_df.loc[i, col] = exc[search_term]
                        if "Provider" in col:
                            location = exc["location"]
                            lci_table_template_df.loc[i, col] = fr"{exc[search_term]} | {location}"
                        bio_idx_start = i
            try:
                # Start filling biosphere flows after technosphere exchanges
                bio_idx_start += 1
                lci_table_template_df.at[bio_idx_start, "Amount"]
                for col in lci_table_template_df.columns:
                    lci_table_template_df[col] = lci_table_template_df[col].astype('object')
                    for i, exc in enumerate(act.exchanges()):
                        idx = i + bio_idx_start
                        search_term = col.lower()
                        if "Provider" in col:
                            continue
                        if "Reference Flow" in col:
                            search_term = "name"
                        # Add a label for biosphere flows
                        if idx == bio_idx_start and col == lci_table_template_df.columns[0]:
                            lci_table_template_df.loc[idx, col] = "Biosphere flow"
                        # Fill in biosphere exchanges
                        elif "bio" in exc["type"] and i > 0:
                            lci_table_template_df.loc[idx, col] = exc[search_term]
                # Remove empty rows from the DataFrame
                lci_table_template_df = self.remove_empty_rows(lci_table_template_df)
            except KeyError:
                pass
            # Store the filled DataFrame for this activity
            dct[act["name"]] = lci_table_template_df

        # Create the results folder if it doesn't exist
        lci_table_folder = self.results_folder(self.path_github, r"data")
        # Path to save the LCI tables Excel file
        lci_table_path = self.join_path(lci_table_folder, r"LCI_tables.xlsx")

        # Write each activity's LCI table to a separate sheet in the Excel file
        with pd.ExcelWriter(lci_table_path, engine='xlsxwriter') as writer:
            for act, df in dct.items():
                sheet_name = act
                # Excel sheet names have a 31 character limit
                if len(act) > 30:
                    sheet_name = act[:29] + " " + act[-1]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

# Function to calculate treatment quantities based on scaling factors from an Excel file
def treatment_quantity(self):
    # Use a context manager to open the Excel file
    with pd.ExcelFile(self.system_path) as excel_file:
        # Get the sheet names
        sheet_names = excel_file.sheet_names
    
    scaling_sheet = ""
    for sheet in sheet_names:
        if "scaling" in sheet:
            scaling_sheet = sheet

    # Read the treatment quantity data from the second sheet
    df_treatment_quantity = pd.read_excel(io=self.system_path, sheet_name=scaling_sheet)
    scaling_dct = {}
    for act in self.db:
        # Filter activities related to penicillin
        if "filling of glass vial" in act['name'] or "tablet" in act['name']:
            for exc in act.exchanges():
                # Match exchanges with penicillin types
                if "penicillium " in exc.input["name"]:
                    for pen_type in df_treatment_quantity.columns:
                        for _, row in df_treatment_quantity.iterrows():
                            if pen_type in exc["name"]:
                                # Calculate scaling factors
                                scaling_dct[exc["name"]] = exc["amount"] * row[pen_type]

    return scaling_dct