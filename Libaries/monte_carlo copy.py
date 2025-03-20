from collections import defaultdict, namedtuple
from typing import Optional, Union
import numpy as np
import brightway2 as bw
import bw2calc as bc
import brightway2 as bd
from bw2calc import MCRandomNumberGenerator, MonteCarloParameterManager
from time import time
import logging

log = logging.getLogger(__name__)

def monte_carlo_lca(cs_name, method):
    if cs_name not in bd.calculation_setups:
        raise ValueError("{} is not a known `calculation_setup`.".format(cs_name))

    state = {
        "cs_name": cs_name,
        "cs": bd.calculation_setups[cs_name],
        "seed": None,
        "cf_rngs": {},
        "CF_rng_vectors": {},
        "include_technosphere": True,
        "include_biosphere": True,
        "include_cfs": True,
        "include_parameters": True,
        "param_rng": None,
        "param_cols": ["row", "col", "type"],
        "tech_rng": None,
        "bio_rng": None,
        "cf_rng": None,
        "func_units": bd.calculation_setups[cs_name]["inv"],
        "rev_fu_index": {i: fu for i, fu in enumerate(bd.calculation_setups[cs_name]["inv"])},
        "activity_keys": [list(fu.keys())[0] for fu in bd.calculation_setups[cs_name]["inv"]],
        "activity_index": {key: index for index, key in enumerate([list(fu.keys())[0] for fu in bd.calculation_setups[cs_name]["inv"]])},
        "rev_activity_index": {index: key for index, key in enumerate([list(fu.keys())[0] for fu in bd.calculation_setups[cs_name]["inv"]])},
        "methods": bd.calculation_setups[cs_name]["ia"],
        "method_index": {m: i for i, m in enumerate(bd.calculation_setups[cs_name]["ia"])},
        "rev_method_index": {i: m for i, m in enumerate(bd.calculation_setups[cs_name]["ia"])},
        "A_matrices": list(),
        "B_matrices": list(),
        "CF_dict": defaultdict(list),
        "parameter_exchanges": list(),
        "parameters": list(),
        "parameter_data": defaultdict(dict),
        "results": list(),
        "lca": bc.LCA(demand=bd.calculation_setups[cs_name]["inv"], method=method)
    }

    return state

def unify_param_exchanges(data: np.ndarray, lca) -> np.ndarray:
    """Convert an array of parameterized exchanges from input/output keys
    into row/col values using dicts generated in bw.LCA object.

    If any given exchange does not exist in the current LCA matrix,
    it will be dropped from the returned array.
    """

    def key_to_rowcol(x) -> Optional[tuple]:
        if x["type"] in [0, 1]:
            row = lca.activity_dict.get(x["input"], None)
            col = lca.product_dict.get(x["output"], None)
        else:
            row = lca.biosphere_dict.get(x["input"], None)
            col = lca.activity_dict.get(x["output"], None)
        # if either the row or the column is None, return np.NaN.
        if row is None or col is None:
            return None
        return row, col, x["type"], x["amount"]

    # Convert the data and store in a new array, dropping Nones.
    converted = (key_to_rowcol(d) for d in data)
    unified = np.array(
        [x for x in converted if x is not None],
        dtype=[("row", "<u4"), ("col", "<u4"), ("type", "u1"), ("amount", "<f4")],
    )
    return unified

def load_data(lca, seed=None, include_technosphere=True, include_biosphere=True, include_cfs=True, include_parameters=True, methods=None):
    """Constructs the random number generators for all of the matrices that
    can be altered by uncertainty.

    If any of these uncertain calculations are not included, the initial
    amounts of the 'params' matrices are used in place of generating
    a vector.
    """
    lca.load_lci_data()

    tech_rng = (
        MCRandomNumberGenerator(lca.tech_params, seed=seed)
        if include_technosphere
        else lca.tech_params["amount"].copy()
    )
    bio_rng = (
        MCRandomNumberGenerator(lca.bio_params, seed=seed)
        if include_biosphere
        else lca.bio_params["amount"].copy()
    )

    cf_rngs = {}
    if lca.lcia:
        for m in methods:
            lca.switch_method(m)
            lca.load_lcia_data()
            cf_rngs[m] = (
                MCRandomNumberGenerator(lca.cf_params, seed=seed)
                if include_cfs
                else lca.cf_params["amount"].copy()
            )

    param_rng = MonteCarloParameterManager(seed=seed) if include_parameters else None

    lca.activity_dict_rev, lca.product_dict_rev, lca.biosphere_dict_rev = lca.reverse_dict()

    return {
        "tech_rng": tech_rng,
        "bio_rng": bio_rng,
        "cf_rngs": cf_rngs,
        "param_rng": param_rng,
        "activity_dict_rev": lca.activity_dict_rev,
        "product_dict_rev": lca.product_dict_rev,
        "biosphere_dict_rev": lca.biosphere_dict_rev,
    }

def calculate(lca, config, **kwargs):
    """Main calculate function for the MC LCA, allows fine-grained control
    over which uncertainties are included when running MC sampling.
    """
    start = time()
    seed = config.get("seed", bc.utils.get_seed())
    include_technosphere = kwargs.get("include_technosphere", True)
    include_biosphere = kwargs.get("include_biosphere", True)
    include_cfs = kwargs.get("include_cfs", True)
    include_parameters = kwargs.get("include_parameters", True)
    iterations = config.get("iterations", 10)
    func_units = config["func_units"]
    methods = config["methods"]
    rev_fu_index = config["rev_fu_index"]
    rev_method_index = config["rev_method_index"]
    param_cols = config["param_cols"]

    state = load_data(lca, seed=seed, include_technosphere=include_technosphere, include_biosphere=include_biosphere, include_cfs=include_cfs, include_parameters=include_parameters, methods=methods)

    results = np.zeros((iterations, len(func_units), len(methods)))

    # Reset GSA variables to empty.
    A_matrices = list()
    B_matrices = list()
    CF_dict = defaultdict(list)
    parameter_exchanges = list()
    parameters = list()

    # Prepare GSA parameter schema:
    if include_parameters:
        parameter_data = state["param_rng"].extract_active_parameters(lca)
        # Add a values field to handle all the sampled parameter values.
        for k in parameter_data:
            parameter_data[k]["values"] = []

    for iteration in range(iterations):
        tech_vector = state["tech_rng"].next() if include_technosphere else state["tech_rng"]
        bio_vector = state["bio_rng"].next() if include_biosphere else state["bio_rng"]
        if include_parameters:
            # Convert the input/output keys into row/col keys, and then match them against
            # the tech_ and bio_params
            data = state["param_rng"].next()
            param_exchanges = unify_param_exchanges(data, lca)

            # Select technosphere subset from param_exchanges.
            subset = param_exchanges[np.isin(param_exchanges["type"], [0, 1])]
            # Create index of where to insert new values from tech_params array.
            idx = np.argwhere(
                np.isin(
                    lca.tech_params[param_cols], subset[param_cols]
                )
            ).flatten()
            # Construct unique array of row+col+type combinations
            uniq = np.unique(lca.tech_params[idx][param_cols])
            # Use the unique array to sort the subset (ensures values
            # are inserted at the correct index)
            sort_idx = np.searchsorted(uniq, subset[param_cols])
            # Finally, insert the sorted subset amounts into the tech_vector
            # at the correct indexes.
            tech_vector[idx] = subset[sort_idx]["amount"]
            # Repeat the above, but for the biosphere array.
            subset = param_exchanges[param_exchanges["type"] == 2]
            idx = np.argwhere(
                np.isin(
                    lca.bio_params[param_cols], subset[param_cols]
                )
            ).flatten()
            uniq = np.unique(lca.bio_params[idx][param_cols])
            sort_idx = np.searchsorted(uniq, subset[param_cols])
            bio_vector[idx] = subset[sort_idx]["amount"]

            # Store parameter data for GSA
            parameter_exchanges.append(param_exchanges)
            parameters.append(state["param_rng"].parameters.to_gsa())
            # Extract sampled values for parameters, store.
            state["param_rng"].retrieve_sampled_values(parameter_data)

        lca.rebuild_technosphere_matrix(tech_vector)
        lca.rebuild_biosphere_matrix(bio_vector)

        # store matrices
        A_matrices.append(lca.technosphere_matrix)
        B_matrices.append(lca.biosphere_matrix)

        if not hasattr(lca, "demand_array"):
            lca.build_demand_array()
        lca.lci_calculation()

        # pre-calculating CF vectors enables the use of the SAME CF vector for each FU in a given run
        cf_vectors = {}
        for m in methods:
            cf_vectors[m] = state["cf_rngs"][m].next() if include_cfs else state["cf_rngs"][m]
            # store CFs for GSA (in a list defaultdict)
            CF_dict[m].append(cf_vectors[m])

        # iterate over FUs
        for row, func_unit in rev_fu_index.items():
            lca.redo_lci(func_unit)  # lca calculation

            # iterate over methods
            for col, m in rev_method_index.items():
                lca.switch_method(m)
                lca.rebuild_characterization_matrix(cf_vectors[m])
                lca.lcia_calculation()
                results[iteration, row, col] = lca.score

    log.info(
        f"Finished {iterations} iterations for {len(func_units)} reference flows and "
        f"{len(methods)} methods in {np.round(time() - start, 2)} seconds."
    )

    MonteCarloResult = namedtuple('MonteCarloResult', [
        'results', 'A_matrices', 'B_matrices', 'CF_dict', 'parameter_exchanges', 'parameters', 'parameter_data'
    ])

    return MonteCarloResult(
        results=results,
        A_matrices=A_matrices,
        B_matrices=B_matrices,
        CF_dict=CF_dict,
        parameter_exchanges=parameter_exchanges,
        parameters=parameters,
        parameter_data=parameter_data if include_parameters else None
    )