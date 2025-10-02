from math import log
from bw2data import get_activity
import numpy as np
import pandas as pd
import bw2data as bd
import bw2calc as bc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.ticker import PercentFormatter

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

def obtain_monte_carlo_samples(samples):
    penG = get_activity((init.db.name, "ca75fd45911fc797bc543d0f564fa97d"))
    penV = get_activity((init.db.name, "8a5478237433fca863df52bd6c71ffcf"))

    penicllin_activities = [
        penG, 
        penV
    ]
    for pen in penicllin_activities:
        num_changed = enrich_activity_uncertainties(pen, recurse=True, depth_limit=3, dry_run=False)
        print(f"Updated {num_changed} exchanges in {pen} with constructed lognormal uncertainty.")
    
    
    uncert_dct = {}
    uncert_func_unit = []
    uncert_func_unit.append({penG : 1})
    uncert_func_unit.append({penV : 1})
    

    for exc in penG.exchanges():
        if exc["type"] == "technosphere" and ("heat" in exc["name"] or "elec" in exc["name"]):

            uncert_func_unit.append({exc.input : exc["amount"]})
            s = exc.get('scale')
            l =  exc.get('loc')

            uncert_dct[exc.input] = np.random.lognormal(mean=l, sigma=s, size=samples)/exc["amount"]
    return uncert_func_unit, uncert_dct

def calculate_impact(samples):
    uncert_func_unit, uncert_dct = obtain_monte_carlo_samples(samples)
    init = m.main()
    init.database_setup(reload=True)
    
    bd.projects.set_current(init.bw_project)
    df = pd.DataFrame(0, index=[list(fu.keys())[0] for fu in uncert_func_unit], columns=[init.lcia_impact_method()[1]], dtype=object)
    bd.Database(init.db.name)
    # print(f'Total amount of calculations: {len(impact_categories) * len(idx_lst)}')
    # # Set up and perform the LCA calculation
    bd.calculation_setups[f'pen G energy uncert'] = {'inv': uncert_func_unit, 'ia': [init.lcia_impact_method()[1]]}
        
    mylca = bc.MultiLCA(f'pen G energy uncert')
    res = mylca.results

    df = pd.DataFrame(res, index=[list(fu.keys())[0] for fu in uncert_func_unit], columns=[init.lcia_impact_method()[1]], dtype=object)

    return df, uncert_dct

def plot_monte_carlo_energy(samples=1000):

    df, uncert_dct = calculate_impact(samples)
    elec_act = df.iloc[2].to_frame().columns[0]
    heat_act = df.iloc[3].to_frame().columns[0]

    elec = df.iloc[2,0]
    heat = df.iloc[3,0]

    penG_impact = df.iloc[0,0] - elec - heat
    penV_impact = df.iloc[1,0] - elec - heat
    
    penicillin_impact = [penG_impact, penV_impact]
    weight = [0.0006, 0.00066]

    elec = df.iloc[2,0]
    heat = df.iloc[3,0]
    lst = []
    width_in, height_in, dpi = init.plot_dimensions()
    colors = init.color_range(colorname="coolwarm", color_quantity=2)
    plt.figure(figsize=(width_in, height_in), dpi=dpi)
    for x, pen in enumerate(penicillin_impact):
        impact = [
            (pen + elec * uncert_dct[elec_act][sample] + heat * uncert_dct[heat_act][sample])*weight[x]*1000  for sample in range(samples)
            ]
        lst.append(impact)
        plt.hist(impact, bins=40, density=True, alpha=0.6, edgecolor='black', color=colors[x])
    legend = ["Pen G", "Pen V"]
    plt.legend(legend)
    plt.title('Monte Carlo simulation of the GWP for energy use in penicillin manufacturing')
    plt.xlabel(r"grams of CO$_2$-eq per treatment")
    plt.ylabel('Probability (%)')
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in y_ticks])
    ax.set_ylim(0,0.27)
    
    plt.tight_layout()
    output_file = init.join_path(
        init.path_github,
        r'figures\monte_carlo_energy.png'
    )
    plt.savefig(output_file, dpi=dpi, format='png', bbox_inches='tight')
    plt.show()

    print(ttest_ind(lst[0], lst[1]))