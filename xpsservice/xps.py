# -*- coding: utf-8 -*-
import os
from functools import lru_cache
from typing import List, Tuple, Union, Dict
import numpy as np
import wrapt_timeout_decorator
import logging
import pickle
import hashlib
from fastapi.logger import logger
from ase import Atoms
from ase.build.tools import sort
from quippy.descriptors import Descriptor
from .cache import soap_config_cache, model_cache
from .models import *
from .optimize import run_xtb_opt
from .settings import MAX_ATOMS_XTB, MAX_ATOMS_FF, TIMEOUT
from .utils import (
    cache_hash,
    molfile2ase,
    molfile2smiles,
    compare_atom_order
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
        logging.debug(f"Loaded SOAP config for {transition_info['element']} {transition_info['orbital']}")
        return soap_config
    except FileNotFoundError:
        raise FileNotFoundError(f"SOAP descriptor not found at {soap_config_filepath}.")
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to load SOAP config file {soap_config_filepath}: {str(e)}")

def load_soap_configs_and_models(transition_map):
    # Clear old cache
    soap_config_cache.clear()
    model_cache.clear()
    
    for transition_key, transition_info in transition_map.items():
        try:
            # Load SOAP config and store in cache
            soap_config = load_soap_config(transition_info)
            soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
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

def calculate_binding_energies(ase_mol: Atoms, transition_key, transition_map):
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in calculate_binding_energies expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")

    transition_info = transition_map[transition_key]
    element = transition_info['element']
    orbital = transition_info['orbital']

    if transition_key not in transition_map:
        raise KeyError(f"Transition '{transition_key}' is not a valid transition. Valid transitions are: {list(transition_map.keys())}")

    soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
    soap_config = soap_config_cache.get(soap_config_hashed_key)
    if soap_config is None:
        logging.error(f"SOAP config not found in cache for transition {transition_key}")
        return []

    soap_descriptor = Descriptor(soap_config)
    if soap_descriptor is None:
        logging.error(f"SOAP descriptor could not be built from SOAP config for transition {transition_key}")
        return []

    model_hashed_key = cache_hash(transition_key, "ml_model_cache")
    ml_model = model_cache.get(model_hashed_key)
    if ml_model is None:
        logging.error(f"ML model not found in cache for transition {transition_key}")
        return []

    desc_data = soap_descriptor.calc(ase_mol)
    if 'data' not in desc_data:
        logging.error(f"No descriptor data found for molecule with transition {transition_key}")
        return []

    desc_molecule = desc_data['data']

    try:
        be, std = ml_model.predict(desc_molecule, return_std=True)
        logging.info(f'Predicted {len(be)} binding energies for element {element}, orbital {orbital}, defined in {transition_key}')
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return []

    if not hasattr(be, '__iter__') or not hasattr(std, '__iter__'):
        logging.error(f"Expected iterable outputs from predict method, got {type(be)} and {type(std)}")
        return []

    return list(zip(be, std))

def run_xps_calculations(ase_mol: Atoms, transition_map) -> dict:
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in run_xps_calculations, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")

    be_predictions = {}
    for transition_key in transition_map.keys():
        transition_info = transition_map[transition_key]
        element = transition_info['element']

        if element in ase_mol.get_chemical_symbols():
            predictions = calculate_binding_energies(ase_mol, transition_key, transition_map)
            be_predictions[transition_key] = predictions
        else:
            logging.warning(f"No model found for element {element} in transition {transition_key}")
     
    return be_predictions

@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_molfile(molfile, method, fmax, energy_reference, transition_map) -> XPSResult:
    
    ase_mol, mol = molfile2ase(molfile, get_max_atoms(method))
    
    opt_result = run_xtb_opt(ase_mol, fmax=fmax, method=method)
    opt_ase_mol = opt_result.atoms
    if not isinstance(opt_ase_mol, Atoms):
        raise TypeError(f"After xtb optimization, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    
    be_predictions = run_xps_calculations(opt_ase_mol, transition_map)
    
    #reference the binding enregy if queried
    if energy_reference == 'solid':
        referenced_be_predictions = reference_be_predictions(be_predictions, transition_map)
        be_predictions = referenced_be_predictions
    
    if isinstance(be_predictions, dict):
        ordered_predictions = reorder_predictions(be_predictions, opt_ase_mol, transition_map)
    else:
        logging.error(f"Unexpected format for be_predictions: {be_predictions}")
        ordered_predictions = []

    return ordered_predictions


def reorder_predictions(be_predictions: dict, ase_mol: Atoms, transition_map: dict) -> List[Prediction]:
    ordered_predictions = []
    
    atom_index_map = {i: atom for i, atom in enumerate(ase_mol)}

    # Create a dictionary to keep track of which atoms have predictions
    atom_predictions_map = {i: {} for i in range(len(ase_mol))}

    for transition_key, predictions in be_predictions.items():
        transition_info = transition_map[transition_key]
        element = transition_info['element']

        element_indices = [i for i, atom in enumerate(ase_mol) if atom.symbol == element]

        for idx, prediction in zip(element_indices, predictions):
            atom_predictions_map[idx][transition_key] = PredictionData(
                bindingEnergy=prediction[0],
                standardDeviation=prediction[1]
            )

    # Iterate over all atoms to ensure each one is included in the output
    for idx, atom in atom_index_map.items():
        position = dict(x=atom.position[0], y=atom.position[1], z=atom.position[2])
        prediction_data = atom_predictions_map[idx]
        ordered_predictions.append(Prediction(atom=atom.symbol, position=position, prediction=prediction_data))

    return ordered_predictions

# Energy referencing of the whole predictions
def reference_be_predictions(be_predictions: dict, transition_map: dict):
    
    referenced_be_predictions = {}
    
    for transition_key, predictions in be_predictions.items():
        # get regression coeffs
        transition_info = transition_map[transition_key]
        regression_coefficients = transition_info['regression_coefficients']
        
        referenced_predictions = []
        for prediction in predictions:
            referenced_predictions.append(reference_be_prediction(prediction, regression_coefficients))
        
        referenced_be_predictions[transition_key] = referenced_predictions
    
    return referenced_be_predictions

'''
# Energy referencing of a single prediction based on polynomial fitting of experimental data
def reference_be_prediction(be_prediction: float, regression_coefficients: list):
    
    referenced_be_prediction = be_prediction
    for i, coeff in enumerate(regression_coefficients):
        logger.debug(f"be_prediction format {be_prediction}")
        referenced_be_prediction[0] += coeff * (be_prediction[0] ** i)
    
    return referenced_be_prediction
'''

# Energy referencing of a single prediction based on polynomial fitting of experimental data
def reference_be_prediction(be_prediction: tuple, regression_coefficients: list):
    # Unpack the tuple
    prediction, std_dev = be_prediction
    
    # Initialize the referenced prediction with the original prediction value
    referenced_prediction = 0.0
    
    # Apply the polynomial correction
    for i, coeff in enumerate(regression_coefficients):
        referenced_prediction += coeff * (prediction ** i)
    
    # Return the modified tuple with the referenced prediction and original standard deviation
    return (referenced_prediction, std_dev)

