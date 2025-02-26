# Import libraries
import pandas as pd
from copy import deepcopy as dc

# Importing self-made libraries
import life_cycle_assessment as lc

# Finding the minimum and maximum values for sterilization equipment
def sterilization_min_max(database_type, autoclave_gwp):
    # Obtaining the impact categories from ReCiPe 2016 v1.03, midpoint (H) and ReCiPe 2016 v1.03, endpoint (H)
    impact_category = lc.lcia_impact_method('recipe')

    # Importing the data from Excel
    file_name = f'C:/Users/ruw/Desktop/RA/Single-use-vs-multi-use-in-health-care/results/case1/data_case1_{database_type}_recipe.xlsx'
    df = lc.import_LCIA_results(file_name, impact_category)

    # Calculating the total values of each scenario
    df_tot, df_scaled = lc.dataframe_element_scaling(df)

    # Extracting GWP (Global Warming Potential)
    gwp_col = df_tot.columns[1]
    df_gwp = df_tot[gwp_col].to_frame()
    df_sens = dc(df_gwp)

    # Initialize min and max values and their corresponding autoclave values
    min = 0
    max = 0
    min_auto = 0
    max_auto = 0

    min_idx = ''
    max_idx = ''

    # Finding the minimum and maximum value
    for col in df_sens:
        for idx, row in df_sens.iterrows():
            val = row[col]
            # Adjust value based on the scenario
            if '200' in idx or 'small' in idx:
                val /= 4 
            else:
                val /= 6

            # Update max value and corresponding autoclave value
            if val > max:
                max = val
                max_idx = idx
                if '2' in idx:
                    max_auto = 14 * 4
                elif '4' in idx:
                    max_auto = 7 * 6
                elif 'S' in idx:
                    max_auto = 9 * 4
                else:
                    max_auto = 5 * 6
        
            # Update min value and corresponding autoclave value
            if val < min or min == 0:
                min = val
                min_idx = idx
                if '2' in idx:
                    min_auto = 28 * 4
                elif '4' in idx:
                    min_auto = 13 * 6
                elif 'S' in idx:
                    min_auto = 14 * 4
                else:
                    min_auto = 7 * 6
                
    # Calculate autoclave GWP for min and max values
    autoclave_gwp_min = autoclave_gwp / min_auto
    autoclave_gwp_max = autoclave_gwp / max_auto

    # Create list of min and max values adjusted by autoclave GWP
    min_max_lst = [max - autoclave_gwp_max, min - autoclave_gwp_min]
    min_max_auto = [min_auto, max_auto]

    # Print the scenarios with the lowest and highest impact
    print(f"lowest impact : {min_idx}, highest impact : {max_idx}")

    return min_max_lst, min_max_auto

def uncertainty_case2(val_dct, df_be, df):
    # Calculate the use electricity
    use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
    df_dct = {}

    # Iterate over the columns of the dataframe
    for cr in df.columns:
        df_dct[cr] = {}
        # Iterate over the rows of the dataframe
        for ir, rr in df.iterrows():
            # Deep copy the base dataframe
            df_sens = dc(df_be)
            # Iterate over the columns of the copied dataframe
            for col in df_sens.columns:
                # Iterate over the rows of the copied dataframe
                for idx, row in df_sens.iterrows():
                    # Skip the total row
                    if ir != 'total':
                        # Get the dictionary for the current sensitivity index
                        dct = val_dct[ir]
                        # Determine if the current column is lower or upper bound
                        val = 0 if 'lower' in cr else 1
                        # Adjust the values based on the sensitivity index
                        if ir == 'Life time' and idx in cr and 'SUD' not in cr and 'Disinfection' not in col and 'autoclave' not in col:
                            row[col] *= 250 / dct[idx][val]
                        elif ir == 'autoclave' and 'autoclave' in col.lower() and 'SUD' not in idx:
                            row[col] *= (14*4) / dct[idx][val]
                        elif ir == 'sterilization' and 'consumables' in col and 'SUD' not in idx:
                            row[col] = dct[idx][val]
                        elif ir == 'cabinet washer' and 'Disinfection' in col and 'SUD' not in idx:
                            row[col] *= 32 / dct[idx][val]
                        elif ir == 'surgery time':
                            row[col] *= use_elec / dct[idx][val]
            # Create a temporary dataframe for the current column
            df_temp = df_sens.loc[cr[:3]].to_frame().T
            # Update the dictionary with the temporary dataframe
            df_dct[cr].update({ir: df_temp})

    return df_dct

def case2_initilazation(df_be, database_type, autoclave_gwp):
    # Define the sensitivity indices
    idx_sens = [
        'autoclave',
        'cabinet washer',
        'Life time',
        'sterilization',
        'surgery time',
        'total'
    ]

    # Initialize the dictionary to store sensitivity values
    val_dct = {
        'autoclave': {},
        'surgery time': {},
        'sterilization': {},
        'Life time': {},
        'cabinet washer': {}
    }
    col_to_df = []

    # Get the minimum and maximum values for sterilization equipment
    min_max_lst, min_max_auto = sterilization_min_max(database_type, autoclave_gwp)

    # Calculate the lower and upper bound for the use electricity
    use_elec_var = [((60-2)/60*40 + 500 * 2/60)/1000, ((60-10)/60*40 + 500 * 10/60)/1000]

    # Iterate over the base dataframe index
    for idx in df_be.index:
        if 'SUD' in idx:
            # Update the dictionary for surgery time if index contains 'SUD'
            val_dct['surgery time'].update({idx: [use_elec_var[0], use_elec_var[1]]})
        else:
            # Update the dictionary for other sensitivity indices
            val_dct['Life time'].update({idx: [50, 500]})
            val_dct['autoclave'].update({idx: [30, 56]})
            val_dct['cabinet washer'].update({idx: [32, 48]})
            val_dct['sterilization'].update({idx: min_max_lst})
            val_dct['surgery time'].update({idx: [use_elec_var[0], use_elec_var[1]]})

        # Append the lower and upper bound columns to the list
        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    # Create a dataframe with zeros, indexed by sensitivity indices and columns by lower and upper bounds
    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df
