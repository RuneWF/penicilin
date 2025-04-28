import bw2data as bd
import bw2calc as bc
import pandas as pd
import copy

# Importing self-made libraries
# from standards import *
# from life_cycle_assessment import *
from results_figures import *
from lca import LCA

class LCIA_calculation(LCA):
    def __init__(self, path, matching_database, database_name):
        super().__init__(path, matching_database, database_name)

        self.functional_unit, self.impact_category = self.LCA_initialization()
        self.bw_project, self.database_name, self.flow, self.lcia_meth = self.data_info
        self.dir_temp, self.file_name, self.file_name_unique_process = self.file_info
         # Ensure impact categories is a list
        self.impact_category = list(self.impact_category) if isinstance(self.impact_category, tuple) else self.impact_category
        
        self.unique_process_index = []
        self.unique_process = []

        # Identify unique processes
        for exc in self.functional_unit.values():
            for proc in exc.keys():
                if str(proc) not in self.unique_process_index:
                    self.unique_process_index.append(str(proc))
                    self.unique_process.append(proc)
        
        self.unique_process_index.sort()

        self.unique_func_unit = []
        for upi in self.unique_process_index:
            for proc in self.unique_process:
                if upi == f'{proc}':
                    self.unique_func_unit.append({proc : 1})


        self.df_unique = pd.DataFrame(0, index=self.unique_process_index, columns=self.impact_category, dtype=object)

        self.df_func_unit = pd.DataFrame(0, index=self.flow, columns=self.impact_category, dtype=object)

    def perform_LCIA(self, sheet_name, case):


        print(f'Total amount of calculations: {len(self.impact_category) * len(self.unique_func_unit)}')
        
        # Set up and perform the LCA calculation
        bd.calculation_setups[f'calc_setup_{case}'] = {'inv': self.unique_func_unit, 'ia': self.impact_category}
        mylca = bc.MultiLCA(f'calc_setup_{case}')
        res = mylca.results
        
        # Store results in DataFrame
        for col, arr in enumerate(res):
            for row, val in enumerate(arr):
                self.df_unique.iat[col, row] = val

        # Save results to file
        save_LCIA_results(self.df_unique, self.file_name_unique_process, sheet_name)

        return self.df_unique
    
    def lcia_dataframe_handling(self, case="penicillin"):
        user_input = ''
        
        # Check if the file exists
        if os.path.isfile(self.file_name_unique_process):
            try:

                user_input = input(f"Do you want to redo the calculations for some process in {self.database_name}?\n"
                                "Options:\n"
                                "  'y' - Yes, redo the calculations\n"
                                "  'n' - No, do not redo any calculations\n"
                                "  'r' - Recalculate based only on the functional unit (FU)\n"
                                "Please enter your choice: ")
                
                # Redo calculations if user chooses 'y'
                if 'y' in user_input.lower():
                    df_unique = self.perform_LCIA(self.file_name_unique_process, self.database_name, case)
            
            except (ValueError, KeyError, TypeError) as e:
                print(e)
                # Perform LCIA if there is an error in importing results
                df_unique = self.perform_LCIA(self.file_name_unique_process, self.database_name, case)        
        else:
            print(f"{self.file_name_unique_process} do not exist, but will be created now")
            # Perform LCIA if file does not exist
            df_unique = self.perform_LCIA(self.database_name, case)
        # Import LCIA results if user chooses 'n'
        if 'n' in user_input.lower():

            self.df_func_unit = import_LCIA_results(self.file_name, self.impact_category)
            
        else:
            # Initialize DataFrame for results
            
            df_unique =  import_LCIA_results(self.file_name_unique_process, self.impact_category)
            for col in self.impact_category:
                for idx, row in self.df_func_unit.iterrows():
                    row[col] = {}

            # Calculate impact values and store in DataFrame
            for col, impact in enumerate(self.impact_category):
                for proc, fu in self.functional_unit.items():
                    for row, (key, item) in enumerate(fu.items()):
                        exc = str(key)
                        
                        val = float(item)
                        factor = df_unique.iat[row, col]
                        if factor is not None:
                            impact_value = val * factor
                        else:
                            impact_value = None

                        try:
                            self.df_func_unit.at[proc, impact].update({exc : impact_value})
                        except ValueError:
                            try:
                                exc = exc.replace(")",")'")
                                self.df_func_unit.at[proc, impact].update({exc : impact_value})
                                
                            except ValueError:
                                print(f'value error for {proc}')
                
            # Save LCIA results to file
            save_LCIA_results(self.df_func_unit, self.file_name, self.database_name)   

        return self.df_func_unit






