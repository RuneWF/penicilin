from pathlib import Path
import pandas as pd
import re
import matplotlib.pyplot as plt

import main as m

init = m.main()

def import_mid_to_endpoint_data():
    mid_to_endpoint_file_path = Path.joinpath(init.path_github, "data", r"ReCiPe2016_CFs_v1.1_20180117 .xlsx") 
    return pd.read_excel(mid_to_endpoint_file_path, sheet_name="mid_to_end", index_col=0)

def import_impact_results(results_df):
    results_df_mid = results_df[results_df.columns[:-3]]

    return results_df_mid.T, results_df.columns[-3:]

def midpoint_acronyms():
    impact_categories = init.lcia_impact_method()[:-3]

    midpoint_acronyms_lst = []
    # Process each midpoint category to create a shortened version for the plot x-axis
    for impact_category in impact_categories:
        ic = impact_category[2]
        string = re.findall(r'\((.*?)\)', ic)
        if 'ODPinfinite' in string[0]:
            string[0] = 'ODP'
        elif '1000' in string[0]:
            string[0] = 'GWP'
        midpoint_acronyms_lst.append(string[0])

    return midpoint_acronyms_lst

def x_tick_labels():
    return [
    "Ecosystem\nDamage",
    "Human\nHealth\nDamage",
    "Natural\nResources\nScarcity"
    ]

def get_mid_to_end_contribution_dct(results_df):
    dct = {}
    results_df_mid, endpoint_ic = import_impact_results(results_df)
    for col in results_df_mid.columns:
        df_temp = import_mid_to_endpoint_data()
        df_temp.columns = endpoint_ic
        for idx, row in results_df_mid.iterrows():
            df_temp.loc[str(idx)] *= row[col]
        
        for tcol in df_temp.columns:
            df_temp[tcol] /= df_temp[tcol].sum()

        dct[col] = df_temp


    for key, df in dct.items():
        idx_new = []
        for txt in midpoint_acronyms():
            for idx in df.index:
                if txt in idx:
                    idx_new.append(idx)
        df = df.reindex(idx_new)
        df.index = midpoint_acronyms()
        df.columns = x_tick_labels()
        dct[key] = df

    return dct



def mid_to_end_contribution_title_text(act, ax, a):
    title_identifier = [r"$\bf{Fig\ A:}$", r"$\bf{Fig\ B:}$"]
    if "V" in str(act):
        return ax.set_title(f"{title_identifier[a]} A SSD oral treatment ", loc="left")
    else:
        return ax.set_title(f"{title_identifier[a]} A SSD IV treatment", loc="left")
    
def mid_to_endpoint_contribution_plot(results_df):

    width_in, height_in, dpi = init.plot_dimensions()
    fig, axes = plt.subplots(1,2, figsize=(width_in, height_in), dpi=dpi)

    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(
        f'Midpoint to Endpoint contribution',  # Your global title
        fontsize=16, y=0.95        # y adjusts vertical position
    )
    dct = get_mid_to_end_contribution_dct(results_df)
    
    for a, (act, df) in enumerate(dct.items()):

        ax = axes[a]

        df = df.T
        bars = df.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            cmap="coolwarm",
            zorder=2
        )
        
        
        # Add outlines to each stacked bar segment
        for bar_container in bars.containers:
            for bar in bar_container:
                bar.set_edgecolor('black')  # Outline color
                bar.set_linewidth(0.5)        # Outline thickness


        mid_to_end_contribution_title_text(act, ax, a)
        ax.set_xticklabels(x_tick_labels(), rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
        ax.set_ylabel("Share of the impact")
        y_ticks = plt.gca().get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in y_ticks])
        ax.set_ylim(0,1.01)
    
        if a == 1:
            leg_color, _ = fig.gca().get_legend_handles_labels()
            leg_txt = list(df.T.index)
            leg_txt = leg_txt[::-1]
            leg_color = leg_color[::-1]
            ax.legend(
                leg_color,
                leg_txt,
                loc='upper left',
                bbox_to_anchor=(0.995, 0.98),
                ncol=1,
                frameon=False
                )
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    output_file = init.join_path(init.figure_folder, f"mid_to_endpoint_contribution.png")
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()