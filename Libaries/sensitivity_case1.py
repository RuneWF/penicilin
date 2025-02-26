# Import libaries
import pandas as pd
from copy import deepcopy as dc

def uncertainty_case1(df_sensitivity, val_dct, df_be, totals_df, idx_sens, col_to_df):
    """
    Perform sensitivity analysis for case 1.

    Parameters:
    df_sensitivity (pd.DataFrame): DataFrame to store sensitivity results.
    val_dct (dict): Dictionary containing sensitivity values.
    df_be (pd.DataFrame): DataFrame containing break-even analysis data.

    Returns:
    pd.DataFrame: Updated DataFrame with sensitivity analysis results.
    """
    
    df_dct = {}
    for cr in df_sensitivity.columns:
        df_dct[cr] = {}
        for ir, rr in df_sensitivity.iterrows():
            df_sens = dc(df_be)

            
            for col in df_sens.columns:
                for idx, row in df_sens.iterrows():
                    if ir != 'total':
                        dct = val_dct[ir]
                        val = 0 if 'lower' in cr else 1

                        if ir == 'Life time' and idx in cr  and 'H' not in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                            row[col] *=  513 / dct[idx][val] 

                        elif ir == 'autoclave' and 'Autoclave' in col and idx in cr:
                            if '2' in cr:
                                    row[col] *= 14 / dct[idx][val]
                            elif '4' in cr:
                                    row[col] *= 7 /dct[idx][val]
                            elif 'AS' in cr:
                                    row[col] *= 9/ dct[idx][val]
                            elif 'AL' in cr:
                                    row[col] *=  5 / dct[idx][val]
                        elif ir == 'protection cover' and idx in cr and 'A' not in idx and 'Disinfection' not in col and 'Autoclave' not in col and 'Recycling' not in col :
                            row[col] *= dct[idx][val] / dct[idx][0]
            df_temp = df_sens.loc[cr[:3]].to_frame().T
            df_dct[cr].update({ir : df_temp})

    return df_dct



def case1_initilazation(df_be):
    """
    Initialize the sensitivity analysis for case 1.

    Parameters:
    df_be (pd.DataFrame): DataFrame containing break-even analysis data.

    Returns:
    tuple: A tuple containing the initialized DataFrame, value dictionary, index list, and column list.
    """
    # Define the indices for sensitivity analysis
    idx_sens = [
        'Life time',
        'autoclave',
        'protection cover',
        'total'
    ]

    # Initialize the dictionary to store sensitivity values
    val_dct = {
        'Life time': {},
        'autoclave': {},
        'protection cover': {}
    }

    # Initialize the list to store column names
    col_to_df = []

    # Populate the value dictionary and column list based on the break-even DataFrame index
    for idx in df_be.index:
        if '2' in idx:
            val_dct['autoclave'].update({idx: [14, 28]})
            val_dct['protection cover'].update({idx: [71/1000, 63/1000]})
        elif '4' in idx:
            val_dct['autoclave'].update({idx: [7, 13]})
            val_dct['protection cover'].update({idx: [202/1000, 190/1000]})
        elif 'S' in idx:
            val_dct['Life time'].update({idx: [314, 827]})
            val_dct['autoclave'].update({idx: [9, 14]})
        else:
            val_dct['Life time'].update({idx: [314, 827]})
            val_dct['autoclave'].update({idx: [5, 7]})

        # Add lower and upper percentage columns for each index
        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    # Create an empty DataFrame with the specified indices and columns
    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df
