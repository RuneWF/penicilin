import bw2data as bd
import bw2calc as bc

import numpy as np
import pandas as pd


from time import time

from life_cycle_assessment import lcia_impact_method

# Importing the parent class
from lca import LCA

class MonteCarlo(LCA):
    # initialization of all the required parameters
    def __init__(self, path, matching_database, database_name, itterations, project="Penicillin", database="penicillin_cut_off"):
        super().__init__(path, matching_database, database_name)
        self.project = project
        self.database = database
        bd.projects.set_current(self.project)
        self.db = bd.Database(self.database)

        self.itterations = itterations

        self.actvities = []
        self.scenarios = ["sc1", "sc2", "sc3"]
        self.dct = {}
        self.uncert = {}
        self.method_GWP = [lcia_impact_method()[1]]
        self.sc_uncrt = {}
        self.df_mc = pd.DataFrame()
        self.dct_MC = {}
        self.data_proccessing_dct = {}

    def obtain_activities(self):
        actvities = []
        for sc in self.scenarios:
            for act in self.db:
                temp = act['name']
                # Check if the flow is valid and add to the flow list
                if ("pill" in temp or( "vial" in temp and "sc" in temp) or 'combined' in temp) and sc in temp:
                    actvities.append(act)
        return actvities

    def find_exchanges(exc, dct, act, uncert):
        stack = [exc]
        while stack:
            current_exc = stack.pop()
            try:
                for lvl in current_exc.input.exchanges():
                    # print(lvl["name"], lvl)
                    if "ev391cutoff" == lvl['database'] and lvl["type"] == "technosphere" and "biosphere3" not in exc['database']:
                        dct[act].update({lvl.input: current_exc['amount'] * lvl['amount']})
                        uncert[act].update({lvl.input: lvl.uncertainty})
                    elif "biosphere3" in lvl['database']:
                        dct[act].update({lvl.input: lvl['amount']})
                    else:
                        if lvl.input != lvl.output:
                            stack.append(lvl)
            except KeyError as e:
                print(f"{e} for {lvl['name']}")
    
    def extract_exchanges_for_MC(self):
        self.actvities = self.obtain_activities()
        self.dct = {a: {} for a in self.actvities}
        self.uncert = {a: {} for a in self.actvities}
        for act in self.actvities:
            for exc in act.exchanges():
                if "technosphere" in exc["type"] and "biosphere3" not in exc['database']:
                    if "ev391cutoff" == exc['database']:
                        self.dct[act].update({exc.input: exc['amount']})
                    else:
                        self.find_exchanges(exc, self.dct, act, self.uncert)
                elif "biosphere3" in exc['database']:
                    self.dct[act].update({exc.input: exc['amount']})
                    # print(exc.input)

        return self.dct, self.uncert
    
    def extract_background_uncertainty(self):
        self.dct, self.uncert = self.extract_exchanges_for_MC()
        for key, dct_act in self.dct.items():
            self.uncert[key] = {}
            for act in dct_act.keys():
                for exc in act.exchanges():
                    unc = exc.uncertainty
                    if exc["type"] == "technosphere"  and unc["uncertainty type"] == 2:
                        self.uncert[key].update({act : {exc.input : exc.uncertainty}})

        return self.uncert
    
    
    def setting_foreground_uncertainty(self):

        self.uncert = self.extract_background_uncertainty()
        for sc in self.actvities:
            loc = 0
            scale = 0
            self.sc_uncrt[sc] = {}
            try:
                for act, uncrt_dct in self.uncert[sc].items():
                    for uncrt in uncrt_dct.values():
                        loc += uncrt['loc']
                        scale += uncrt['scale']
                self.sc_uncrt[sc].update({'loc' : loc/len(self.uncert[sc])})
                self.sc_uncrt[sc].update({'scale' : scale/len(self.uncert[sc])})

            except KeyError:
                print(f"{KeyError} for {sc}")
        
        return self.sc_uncrt

    def monte_carlo_simulation(self):
        bd.projects.set_current(self.project)
        self.actvities = self.obtain_activities()
        self.df_mc = pd.DataFrame(0, index=self.actvities, columns=list(range(1,self.itterations+1)), dtype=object)
        self.sc_uncrt = self.setting_foreground_uncertainty()

        for itt in range(self.itterations):
            start = time()
            func_unit_MC = []
            print(f"Iterration {itt+1} of {self.itterations}")

            for act, uncert in self.sc_uncrt.items():
                mean = np.exp(uncert['loc'])
                sigma = uncert['scale']
                new_val = np.random.lognormal(mean, sigma)
                func_unit_MC.append({act : new_val})
            
                

            bd.calculation_setups[f'MC_{itt+1}'] = {'inv': func_unit_MC, 'ia': self.method_GWP}
            MC_res = bc.MultiLCA(f'MC_{itt+1}')
            MC_results_array = MC_res.results

            # Store results in DataFrame
            for idx, res in enumerate(MC_results_array):
                self.df_mc.iat[idx, itt] = res
            print(f"Iterration {itt+1} took {round(time() - start,2)} seconds")

        return self.df_mc
    
    def monte_carlo_dct(self):
        self.df_mc = self.monte_carlo_simulation()        

        for idx,row in self.df_mc.iterrows():
            arr_temp = np.zeros(self.itterations)
            for col in self.df_mc.columns:
                arr_temp[col-1] = row[col]
            self.dct_MC[idx] = arr_temp
        return self.dct_MC

    def obtain_uncertainty_values(self):
        data_proccessing = ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum']
        
        self.dct_MC = self.monte_carlo_dct()

        for act, arr in self.dct_MC.items():
            self.data_proccessing_dct[act] = {}
            for dp in data_proccessing:
                if 'mean' in dp.lower():
                    self.data_proccessing_dct[act].update({dp : np.mean(arr)})
                elif 'median' in dp.lower():
                    self.data_proccessing_dct[act].update({dp : np.median(arr)})
                elif 'standard' in dp.lower():
                    self.data_proccessing_dct[act].update({dp : np.std(arr)})
                elif 'minimum' in dp.lower():
                    self.data_proccessing_dct[act].update({dp : np.min(arr)})
                elif 'maximum' in dp.lower():
                    self.data_proccessing_dct[act].update({dp : np.max(arr)})

        return self.data_proccessing_dct
    

