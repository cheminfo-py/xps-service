# -*- coding: utf-8 -*-
import os

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import wrapt_timeout_decorator
import logging
import pickle
import hashlib

from ase import Atoms
from quippy.descriptors import Descriptor

from .cache import soap_config_cache, model_cache
from .models import *
from .optimize import run_xtb_opt
from .settings import MAX_ATOMS_XTB, MAX_ATOMS_FF, TIMEOUT, transition_map
from .utils import (
    cache_hash,
    molfile2ase,
    molfile2smiles
)



def load_ml_model(transition_info):
    try:
        model_filepath = transition_info['model_filepath']
        with open(model_filepath, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.debug(f"Loaded ML model for {transition_info['element']} {transition_info['orbital']}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_filepath}.")
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to load model file {model_filepath}: {str(e)}")


def load_soap_config(transition_info):
    try:
        soap_config_filepath = transition_info['soap_config_filepath']
        with open(soap_config_filepath, 'rb') as soap_config_file:
            soap_config = pickle.load(soap_config_file)
        logging.debug(f"Loaded ML model for {transition_info['element']} {transition_info['orbital']}")
        return soap_config
    except FileNotFoundError:
        raise FileNotFoundError(f"SOAP descriptor not found at {soap_config_filepath}.")
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to load model file {soap_config_filepath}: {str(e)}")


def load_soap_configs_and_models(transition_map):
    #clear old cache    
    soap_config_cache.clear()
    model_cache.clear()
    
    for transition_key, transition_info in transition_map.items():
        try:
            
            # Load SOAP config and store in cache
            soap_config = load_soap_config(transition_info)
            soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
            print(soap_config_hashed_key)
            soap_config_cache.set(soap_config_hashed_key, soap_config)
            logging.debug(f"SOAP config for {transition_key} loaded")
            
            # Load ML model and store in cache
            ml_model = load_ml_model(transition_info)
            ml_model_hashed_key = cache_hash(transition_key, "ml_model_cache")
            model_cache.set(ml_model_hashed_key, ml_model)
            logging.debug(f"ML model for {transition_key} loaded")
            
        except Exception as e:
            logging.error(f"Error loading data for transition {transition_key}: {str(e)}")
            continue  # Optionally skip to the next iteration on error


def check_cache_status(transition_map) -> Dict[str, Dict[str, bool]]:
    cache_status = {}

    # Iterate through each transition in the transition_map
    for transition_key in transition_map:
        # Check the status of each cache for the transition key
        soap_config_loaded = cache_hash(transition_key, "soap_config_cache") in soap_config_cache
        model_loaded = cache_hash(transition_key, "ml_model_cache") in model_cache
        
        # Store the status in the cache_status dictionary
        cache_status[transition_key] = {
            "soap_config_loaded": soap_config_loaded,
            "model_loaded": model_loaded,
        }
    
    return cache_status


def has_any_cache_failure(cache_status: Dict[str, Dict[str, bool]]) -> bool:
    # Iterate through each transition and check each status
    for transition_key, status_dict in cache_status.items():
        for status_name, status_value in status_dict.items():
            if not status_value:
                # If any status is False, return True
                return True
    
    # If all statuses are True, return False
    return False


def get_max_atoms(method):
    if method == "GFNFF":
        return MAX_ATOMS_FF
    elif method == "GFN2xTB":
        return MAX_ATOMS_XTB
    elif method == "GFN1xTB":
        return MAX_ATOMS_XTB


def calculate_binding_energies(ase_mol: Atoms, transition_key):
    print(" entered calculate binding energies")
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in calculate_binding_energies expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    """
    Calculate binding energies for the atoms of a given molecule based on the loaded ML model and loaded descriptor.
    
    Parameters:
    mol: The molecule object.
    element: The element for which to calculate binding energies.
    transition_key: The specific transition_key follows the naming in transition_map from .models.py (e.g., "C1s" or "O1s").

    Returns:
    list: A list of binding energy predictions for the specified element and transition.
    """
    transition_info = transition_map[transition_key]
    element = transition_info['element']
    orbital = transition_info['orbital']

    # Check if the input transition is one of the keys in the transition_map
    if transition_key not in transition_map:
        raise KeyError(f"Transition '{transition_key}' is not a valid transition. Valid transitions are: {list(transition_map.keys())}")
    
    # Retrieve SOAP config from the cache
    soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
    soap_config = soap_config_cache.get(soap_config_hashed_key)
    if soap_config is None:
        logging.error(f"SOAP config not found in cache for transition {transition_key}")
        return []
    
    #Build SOAP descriptor from SOAP config loaded
    soap_descriptor = Descriptor(soap_config)
    if soap_descriptor is None:
        logging.error(f"SOAP descriptor could not be built from SOAP config for transition {transition_key}")
        return []
    
    # Retrieve ML model from the cache
    model_hashed_key = cache_hash(transition_key, "ml_model_cache")
    ml_model = model_cache.get(model_hashed_key)
    if ml_model is None:
        logging.error(f"ML model not found in cache for transition {transition_key}")
        return []
    
    # Calculate the SOAP descriptor for the molecule
    desc_data = soap_descriptor.calc(ase_mol)

    # Check if the descriptor data is available
    if 'data' not in desc_data:
        logging.error(f"No descriptor data found for molecule with transition {transition_key}")
        return []

    # Get the data from the descriptor object
    desc_molecule = desc_data['data']
    
    # Predict binding energies using the ML model
    be, std = ml_model.predict(desc_molecule, return_std=True)
    logging.info(f'Predicted {len(be)} binding energies for element {element}, orbital {orbital}, defined in {transition_key}')

    # Return a list of binding energy predictions
    return list(zip(be, std))


def run_xps_calculations(ase_mol: Atoms) -> dict:
    #Check the type of the input molecule
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in run_xps_calculation, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")

    # Dictionary to store predictions by element and transition
    be_predictions = {}

    # Iterate through the keys (transitions) in transition_map
    for transition_key in transition_map.keys():
        # Retrieve the element and orbital from the transition info
        transition_info = transition_map[transition_key]
        element = transition_info['element']

        # Check if the element is present in the ASE molecule
        if element in ase_mol.symbols:
            # Calculate binding energies for the element and transition
            predictions = calculate_binding_energies(ase_mol, transition_key)

            # Store the predictions in the dictionary using the transition_key as the key
            be_predictions[transition_key] = predictions
        else:
            logging.warning(f"No model found for element {element} in transition {transition_key}")

    return be_predictions


@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_molfile(molfile, method, fmax) -> XPSResult:

    # Convert molfile to smiles and ASE molecule
    smiles = molfile2smiles(molfile)
    ase_mol, mol = molfile2ase(molfile, get_max_atoms(method))
    
    # Optimize the ASE molecule
    opt_result = run_xtb_opt(ase_mol, fmax=fmax, method=method)
    opt_ase_mol = opt_result.atoms
    if not isinstance(opt_ase_mol, Atoms):
        raise TypeError(f"After xtb optimization, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    
    # Run XPS calculations
    be_predictions = run_xps_calculations(opt_ase_mol)
    
    # Extract binding energies and standard deviations
    binding_energies = []
    standard_deviations = []
    for transition_key, predictions in be_predictions.items():
        energies, stds = zip(*predictions)
        binding_energies.extend(list(energies))
        standard_deviations.extend(list(stds))

    # Create an instance of XPSResult
    xps_result = XPSResult(
        molfile=molfile,
        smiles=smiles,
        bindingEnergies=binding_energies,
        standardDeviations=standard_deviations
    )

    return xps_result


