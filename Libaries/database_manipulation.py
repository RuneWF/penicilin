# Importing libraries
import bw2io as bi
import bw2data as bd
import brightway2 as bw 
import pandas as pd

import standards as s


def import_excel_database_to_brightway(data, matching_database, path_github):
    # Save the data to a temporary file that can be used by ExcelImporter
    temp_path = s.join_path(path_github, r"data\temp.xlsx")

    data.to_excel(temp_path, index=False)
    
    # Use the temporary file with ExcelImporter
    imp = bi.ExcelImporter(temp_path)  # the path to your inventory excel file
    imp.apply_strategies()

    # Loop through each database and match
    print(f"Matching database: {matching_database}")
    imp.match_database(matching_database, fields=('name', 'unit', 'location', 'reference product'))
    if list(imp.unlinked):
        print(f"Unlinked items after matching {matching_database}: {list(imp.unlinked)}")

    # Match without specifying a database
    # imp.match_database(fields=('name', 'unit', 'location'))

    # Generate statistics and write results
    imp.statistics()
    imp.write_excel(only_unlinked=True)
    unlinked_items = list(imp.unlinked)
    imp.write_database()

    # Print unlinked items if needed
    if unlinked_items:
        print(unlinked_items)

    import_excel_database_to_brightway.has_been_called = True

import_excel_database_to_brightway.has_been_called = False

def reload_database(matching_database, system_path, path_github, proj_database):
    reload_database.has_been_called = True
    user_input = input(f"Do you want to reload the {proj_database}? Enter 'y' for yes and 'n' for no")
    
    if user_input.lower() == 'y':

        data = pd.read_excel(system_path, sheet_name=matching_database)

        # Removing the old database
        db_old = bd.Database(proj_database)
        db_old.deregister()

        import_excel_database_to_brightway(data, matching_database, path_github)

    elif user_input.lower() == 'n':
        print('You selected to not reload')

    else:
        print('Invalid argument, try again')
        reload_database(matching_database, matching_database, system_path)

reload_database.has_been_called = False

def import_databases(matching_database, path):

    path_github, ecoinevnt_paths, system_path = s.data_paths(path)

    # Check if Ecoinvent databases are already present
    if matching_database in bd.databases:# and 'ev391apos' in bd.databases and 'ev391cutoff' in bd.databases:
        pass
        # print('Ecoinvent 3.9.1 is already present in the project.')
    else:

        # Import Consequential database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths[matching_database], db_name=matching_database)
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

    data = pd.read_excel(system_path, sheet_name=matching_database)
    proj_database_str = data.columns[1]
    if proj_database_str not in bd.databases:
        import_excel_database_to_brightway(data, matching_database, path_github)

    # Reload databases if needed
    if import_excel_database_to_brightway.has_been_called is False:
        reload_database(matching_database, system_path, path_github, proj_database_str)


def remove_bio_co2_recipe():
    all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H)' in str(m) and 'no LT' not in str(m)] # Midpoint

    # Obtaining the endpoint categories and ignoring land transformation
    endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H)' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

    method_name_new_mid= all_methods[0][0] + ' - no biogenic'
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
                new_method.register()
                new_method.write(recipe_no_bio_CO2)

                # Step 6: Store the new method
                new_methods[metod] = new_method_key
                print(f"New method created: {new_method_key} with {len(recipe_no_bio_CO2)} CFs")

def create_new_bs3_activities(df):
    biosphere3 = bw.Database('biosphere3')
    # Get the list of columns from the dataframe
    new_flow = {}
    codes = {}
    
    # Iterate over each column in the dataframe
    for col in df.columns:
        # Define the new emission flow
        new_flow[col] = {
            'name': col,
            'categories': ('water',),
            'unit': 'kilogram',
            'type': 'emission',
            'location': ""
        }
        # Create a unique code for the new flow
        codes[col] = f"self-made-{col}-1"
        # Check if the code already exists in biosphere3
        if codes[col] not in [act['code'] for act in biosphere3]:
            # Create and save the new activity in biosphere3
            new_flow_entry = biosphere3.new_activity(code=codes[col], **new_flow[col])
            new_flow_entry.save()
            print(f"{col} is added to biosphere3")
        # else:
        #     print(f"{col} is present in biosphere3")
    
    return codes

def filtered_lcia_methods():
    # Get all methods related to ReCiPe 2016 midpoint/endpoint (H) without biogenic
    midpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)]
    endpoint_method = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) - no biogenic' in str(m) and 'no LT' not in str(m)]

    all_methods = midpoint_method + endpoint_method

    # Filter methods related to ecotoxicity
    meth_ecotoxicity = [m for m in all_methods if "ecotoxicity" in m[1] or "ecosystem quality" in m[1]]

    return meth_ecotoxicity

def add_activity_to_biosphere3(df, act_dct):
    # Filter methods related to ecotoxicity
    ecotoxicity_methods = filtered_lcia_methods()
    
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

def add_new_biosphere_activities(bw_project, path):
    path_github, ecoinevnt_paths, system_path = s.data_paths(path)

    # Set the current Brightway2 project
    bd.projects.set_current(bw_project)
    
    # Load the new impacts data from the specified path
    data_path = s.join_path(path_github, r"data\new_impacts.xlsx")
    df = pd.read_excel(data_path, index_col=0)
    
    # Create new biosphere activities and get their codes
    codes = create_new_bs3_activities(df)

    # Load the biosphere3 database
    biosphere3 = bw.Database('biosphere3')

    act_dct = {}
    # Map the new activities to their names
    for c in codes.values():
        for bs3 in biosphere3:
            if c in bs3[1]:
                act_dct[bs3['name']] = bs3

    # Add the new activities to the biosphere3 database
    add_activity_to_biosphere3(df, act_dct)



def database_setup(path, matching_database, bw_project="Penicillin"):
    import_excel_database_to_brightway.has_been_called = False

    # Set the current Brightway project
    bd.projects.set_current(bw_project)
    

    # Check if biosphere database is already present
    if any("biosphere" in db for db in bd.databases):
        pass
        # print('Biosphere is already present in the project.')
    else:
        bi.bw2setup()

    if isinstance(matching_database, str):
        import_databases(matching_database, path)
    elif isinstance(matching_database, list):
        for db in matching_database:
            import_databases(db, path)

    if reload_database.has_been_called is False:
        add_new_biosphere_activities(bw_project, path)