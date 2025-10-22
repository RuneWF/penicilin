from math import log
from bw2data import get_activity
import numpy as np
import pandas as pd
import bw2data as bd
import bw2calc as bc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path


import main as m

init = m.main()

try:
    # Optional helper to compute additional GSD from pedigree scores (recommended)
    import pedigree_matrix  # pip install pedigree_matrix
    HAVE_PED_LIB = True
except Exception:
    HAVE_PED_LIB = False


DB_NAME = init.db.name  # or e.g., "ecoinvent 3.9.1 cutoff"
ROOT = get_activity((DB_NAME, "ca75fd45911fc797bc543d0f564fa97d"))

# User knobs
MAX_DEPTH_BORROW = 5          # how far down we search for a descendant with uncertainty
BFS_BRANCH_LIMIT = 200        # limit to avoid walking a huge graph
DEFAULT_BASIC_GSD = 1.05      # fallback base uncertainty (geometric SD) if you have no table
DRY_RUN = False               # set True to see what would be changed without saving

# ---------- Utilities ----------
def is_lognormal(exc):
    return exc.get("uncertainty type", 0) == 1 and exc.get("scale") not in (None, 0)

def has_any_uncertainty(exc):
    ut = exc.get("uncertainty type", 0)
    if ut == 0:
        return False
    # Normal, lognormal, triangular, uniform etc.:
    # treat as usable if their scale or bounds are present
    if ut == 1:  # lognormal
        return exc.get("scale") not in (None, 0)
    if ut == 2:  # normal
        return exc.get("scale") not in (None, 0)
    if ut in (3, 4):  # uniform/triangular
        return exc.get("minimum") is not None and exc.get("maximum") is not None
    return False

def cv_from_exchange(exc):
    """Return (CV, kind) for exchange if we can infer it, else (None, None)."""
    ut = exc.get("uncertainty type", 0)
    if ut == 1:  # lognormal
        sigma = exc.get("scale")
        if sigma is None or sigma == 0:
            return None, None
        cv = ( (2.718281828 ** (sigma**2)) - 1 ) ** 0.5  # sqrt(exp(sigma^2)-1)
        return cv, "lognormal"
    elif ut == 2:  # normal
        mu, sigma = exc.get("loc"), exc.get("scale")
        if sigma is None or mu in (None, 0):
            return None, None
        if mu <= 0:
            # Can't make a relative measure if mean<=0; skip
            return None, None
        return sigma / mu, "normal"
    # For bounded distributions we could approximate with (max-min)/(2*sqrt(3)*mean), but skip for clarity
    return None, None

def ln_from_amount(amount):
    """Return loc=ln(|amount|) and negative flag for Brightway lognormal."""
    neg = amount < 0
    val = abs(amount)
    if val == 0:
        # Degenerate; nudge slightly to avoid log(0)
        val = 1e-30
    return log(val), neg

# ---- Pedigree → lognormal σ ----
def sigma_from_pedigree(exchange, default_basic_gsd=DEFAULT_BASIC_GSD):
    """Compute total lognormal sigma from pedigree scores; return None if not possible."""
    ped = exchange.get("pedigree") or exchange.get("pedigree matrix")
    if not ped:
        return None

    # 1) Basic uncertainty as a geometric standard deviation (GSD >= 1)
    gsd_basic = default_basic_gsd
    # TODO (recommended): Map gsd_basic from ecoinvent's basic uncertainty table by flow type
    # See: ecoinvent DQG explanation of 'basic uncertainty' + pedigree combination.  # (sources below)

    # 2) Additional multiplicative factors from the five pedigree scores
    try:
        if HAVE_PED_LIB:
            # This library implements the ecoinvent pedigree math; its API provides
            # factors/aggregation for the five scores into an additional GSD.
            # We'll compute an additional GSD (>=1) and combine with basic on log-scale.
            add_gsd = pedigree_matrix.additional_gsd_from_scores(  # <-- function name in your install
                reliability=ped.get("reliability"),
                completeness=ped.get("completeness"),
                temporal=ped.get("temporal correlation"),
                geographical=ped.get("geographical correlation"),
                technological=ped.get("further technological correlation"),
            )
        else:
            # Simple fallback if the library isn't available:
            # treat each score >1 as adding a modest factor. You should replace this with the real table
            # or install `pedigree_matrix`. This keeps the pipeline working.
            increments = []
            for k, s in [
                ("reliability", ped.get("reliability")),
                ("completeness", ped.get("completeness")),
                ("temporal correlation", ped.get("temporal correlation")),
                ("geographical correlation", ped.get("geographical correlation")),
                ("further technological correlation", ped.get("further technological correlation")),
            ]:
                if s is None:
                    continue
                # heuristic: each step above 1 multiplies GSD by 1.1
                increments.append(1.1 ** max(0, int(s) - 1))
            add_gsd = 1.0
            for g in increments:
                add_gsd *= g
    except Exception:
        return None

    # Combine on the log scale: total σ = sqrt( ln(GSD_basic)^2 + ln(GSD_additional)^2 )
    import math
    sigma = math.sqrt( math.log(gsd_basic)**2 + math.log(add_gsd)**2 )
    return sigma

# ---- Borrowing from descendants ----

def borrow_sigma_from_descendants(activity, max_depth=MAX_DEPTH_BORROW, branch_limit=BFS_BRANCH_LIMIT):
    """Breadth-first search down technosphere suppliers for exchanges with usable uncertainty.
       Return a representative log-sigma (weighted average over matches) or None."""
    from collections import deque
    import math
    q = deque()
    q.append((activity, 0))
    seen = set([activity.key])
    weighted_sum = 0.0
    total_weight = 0.0
    visited_edges = 0

    while q:
        node, depth = q.popleft()
        if depth >= max_depth:
            continue

        for exc in node.technosphere():
            visited_edges += 1
            if visited_edges > branch_limit:
                break

            if has_any_uncertainty(exc):
                cv, kind = cv_from_exchange(exc)
                if cv is not None:
                    weight = abs(exc.get("amount", 1.0)) or 1.0
                    weighted_sum += cv * weight
                    total_weight += weight

            supplier = exc.input
            if supplier.key not in seen:
                seen.add(supplier.key)
                q.append((supplier, depth + 1))

        if visited_edges > branch_limit:
            break
        # Stop BFS if we found any matches at this depth
        if total_weight > 0:
            break

    if total_weight == 0:
        return None

    # Weighted average CV
    cv = weighted_sum / total_weight
    # Convert CV to lognormal sigma
    sigma = math.sqrt(math.log(1 + cv**2))
    return sigma


# ---- Main: enrich one activity (and optionally its sub-suppliers) ----
def enrich_activity_uncertainties(activity, recurse=True, depth_limit=3, dry_run=DRY_RUN):
    """Set missing lognormal uncertainty for technosphere exchanges of `activity`:
       1) Try pedigree → sigma
       2) Else borrow sigma from nearest descendants
       Writes loc, scale, type, negative where needed.
    """
    changed = 0
    visited_acts = set()

    def _enrich_one(act, depth):
        nonlocal changed
        if act.key in visited_acts or depth > depth_limit:
            return
        visited_acts.add(act.key)

        for exc in act.technosphere():
            # Skip if already lognormal with scale
            if is_lognormal(exc):
                continue
            amount = exc.get("amount", 0.0) or 0.0

            # Step 1: pedigree
            sigma = sigma_from_pedigree(exc)
            # Step 2: borrow from descendants if pedigree not available
            if sigma is None:
                sigma = borrow_sigma_from_descendants(exc.input, max_depth=MAX_DEPTH_BORROW)

            if sigma is not None:
                loc, neg = ln_from_amount(amount)
                # Set uncertainty fields to lognormal
                exc["uncertainty type"] = 1  # lognormal
                exc["loc"] = loc
                exc["scale"] = float(sigma)
                exc["negative"] = bool(neg)  # important for negative amounts in lognormal
                if not dry_run:
                    exc.save()
                changed += 1

            # Recurse into suppliers
            if recurse:
                _enrich_one(exc.input, depth+1)

    _enrich_one(activity, 0)
    return changed

# --- Run it ---
def create_uncertainty_func_unit(samples):

    penG = get_activity((init.db.name, "ca75fd45911fc797bc543d0f564fa97d")) # production of Penicillin G
    penV = get_activity((init.db.name, "8a5478237433fca863df52bd6c71ffcf")) # production of Penicillin V

    penicllin_activities = [
        penG, 
        penV
        ]
    for pen in penicllin_activities:
        enrich_activity_uncertainties(pen, depth_limit=5)
    
    uncert_dct = {}
    uncert_func_unit = []

    for pen in penicllin_activities:
        uncert_dct[pen] = {}
        # uncert_func_unit[pen] = []
        for exc in pen.exchanges():
            if exc["type"] == "technosphere":
                
                s = exc.get('scale')
                l = exc.get('loc')
                # Initialize the key if it does not exist
                penicillin_qunantity = 1
                for up in penG.upstream():
                    if up.output != penG:
                        penicillin_qunantity = up["amount"]
                if exc.input not in uncert_dct[pen]:
                    uncert_func_unit.append({exc.input : exc["amount"]*penicillin_qunantity})
                uncert_dct[pen][exc.input] = np.random.lognormal(mean=l, sigma=s, size=samples)/exc["amount"]

    df_uncert = pd.DataFrame(uncert_dct)

    return df_uncert, uncert_func_unit

def update_index_colunms(df, df_uncert):
    try:
        df.index = [idx["name"] for idx in df.index]
    except TypeError:
        pass

    try:
        df_uncert.index = [idx["name"] for idx in df_uncert.index]
        df_uncert.columns = [col["name"] for col in df_uncert.columns]
    except TypeError:
        pass

def put_uncert_in_dataframe(pen_compact_idx, df_impact_sens, samples):
    df_categorized_uncert = pd.DataFrame(0, index=[key for key in pen_compact_idx.keys()], columns=df_impact_sens.columns, dtype=object)
    for col in df_categorized_uncert.columns:
        for _, row in df_categorized_uncert.iterrows():
            row[col] = [0] * samples
            

    return df_categorized_uncert

def category_impact(pen_compact_idx, df_impact_sens, df_impact):
    df_categorized = pd.DataFrame(0, index=[key for key in pen_compact_idx.keys()], columns=df_impact_sens.columns, dtype=object)
    df_categorized
    for col in df_impact.columns:
        for cat, st_lst in pen_compact_idx.items():
            for idx, row in df_impact.iterrows():
                if not any(st in idx for st in st_lst):
                    df_categorized.loc[cat, col] += row[col]
        # df_categorized.loc["Total", col] = df_impact[col].sum()
        print(df_impact[col].sum()*1000)

    return df_categorized     

def calc_uncertainty_impact(samples, calc):
    df_uncert, uncert_func_unit = create_uncertainty_func_unit(samples)
    init.database_setup(reload=True)
    path = init.results_path

    penicillin_background_uncert_file = Path.joinpath(path, "sensitivity", "penicillin_background_uncert.xlsx")

    bd.projects.set_current(init.bw_project)
    if not Path.exists(penicillin_background_uncert_file) or calc:
        df = pd.DataFrame(0, index=[list(fu.keys())[0] for fu in uncert_func_unit], columns=[init.lcia_impact_method()[1]], dtype=object)
        bd.Database(init.db.name)
        # # Set up and perform the LCA calculation
        calc_setup_name = f'penicillin production uncert'
        bd.calculation_setups[calc_setup_name] = {'inv': uncert_func_unit, 'ia': [init.lcia_impact_method()[1]]}
            
        mylca = bc.MultiLCA(calc_setup_name)
        res = mylca.results

        df = pd.DataFrame(res, index=[list(fu.keys())[0] for fu in uncert_func_unit], columns=[init.lcia_impact_method()[1]], dtype=object)

    elif Path.exists(penicillin_background_uncert_file) or not calc:
        df = init.import_LCIA_results(penicillin_background_uncert_file)

    update_index_colunms(df, df_uncert)

    df_impact_sens = pd.DataFrame(None, index=df_uncert.index, columns=df_uncert.columns, dtype=object)

    for col in df_uncert.columns:
        for idx, row in df_uncert.iterrows():
            lst = row[col]

            df_impact_sens.at[idx, col] = lst * df.loc[idx].to_numpy()[0]

    mask = df_uncert.notna()

    df_impact = pd.DataFrame(None, index=df_uncert.index, columns=df_uncert.columns, dtype=object)

    for col in df_impact.columns:
        for idx, row in df_impact.iterrows():
            row[col] = df.loc[idx].to_numpy()[0]

    df_impact = df_impact * mask
    
    pen_compact_idx = {
        "Fermentation": ["pharmamedia", "phenyl", "phenoxy", "glucose", "oxygen", "tap water", "sulfate"],
        "Extraction ": ["butyl", "sulfuric"],
        "Purification": ["sodium", "acetone"],
        "Energy": ["heat", "electricity"],
        "Waste": ["incineration", "penicillium"]
    }

    df_categorized_uncert = put_uncert_in_dataframe(pen_compact_idx, df_impact_sens, samples)
    for key, st in pen_compact_idx.items():
        for col in df_impact_sens.columns:
            for idx, row in df_impact_sens.iterrows():
                if any(s in idx for s in st):
                    lst = row[col]
                    # print(lst)
                    if lst is not None:
                        try:
                            for i, val in enumerate(lst):
                                df_categorized_uncert.at[key, col][i] += val
                        except TypeError as e:
                            # print(e)
                            pass
        df_categorized_uncert.at[key, col] = np.array(df_categorized_uncert.at[key, col])
    
    df_categorized = category_impact(pen_compact_idx, df_impact_sens, df_impact)

    for col in df_categorized_uncert.columns:
        for idx, row in df_categorized_uncert.iterrows():
            rest_sum = df_categorized.at[idx, col]
            # cat_impact = df_categorized.at[idx, col]
            row[col] = [(i + rest_sum )*1000 for i in row[col]]

    return df_categorized_uncert

def plot_monte_carlo_background_uncertainty(samples=1000, calc=False):
    df_categorized_uncert = calc_uncertainty_impact(samples, calc)
    title_identifier = [r"$\bf{Fig\ A:}$ ", r"$\bf{Fig\ B:}$ "]

    width_in, height_in, dpi = init.plot_dimensions()
    fig, axes = plt.subplots(1,len(df_categorized_uncert.columns), figsize=(width_in, height_in), dpi=dpi)
    plt.subplots_adjust(wspace=0.3)

    fig.suptitle(
        f'Monte Carlo simulation of background uncertainty (n={samples})',  # Your global title
        fontsize=16, y=0.95        # y adjusts vertical position
    )

    colors = init.color_range(colorname="coolwarm", color_quantity=2)
    blue = colors[0]
    red = colors[1]
    for a, col in enumerate(df_categorized_uncert.columns):
        ax = axes[a]
        data = df_categorized_uncert[col]
        bp = ax.boxplot(data, patch_artist = True, zorder=3)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-0)
        ax.set_xticklabels(list(df_categorized_uncert.index), rotation=45)
        ax.set_ylabel('grams of CO$_2$-eq per SSD treatment')
        ax.set_title(f'{title_identifier[a]}Penicillin {col[-1]} production', loc="left")
        
        for patch, _ in zip(bp['boxes'], blue):
            patch.set_facecolor(blue)

        for whisker in bp['whiskers']:
            whisker.set(color =blue,
                        linewidth = 2,
                        )

        # changing color and linewidth of
        for cap in bp['caps']:
            cap.set(color =blue,
                    linewidth = 2)

        for median in bp['medians']:
            median.set(color = red,
                    linewidth = 1)

        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker ='.',
                    color = blue,
                    markeredgecolor=red,
                    markersize=3 
                )
            
    plt.tight_layout()
    plt.savefig(
        Path.joinpath(init.figure_folder, "monte_carlo_penicillin_background_uncert.png"), 
        dpi=dpi, 
        format='png', 
        bbox_inches='tight'
        )
    
    plt.show()

