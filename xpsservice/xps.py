# -*- coding: utf-8 -*-
import os

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import wrapt_timeout_decorator
import logging
import pickle

from ase import Atoms
from quippy.descriptors import Descriptor

from .cache import soap_config_cache, soap_descriptor_cache, model_cache
from .models import *
from .optimize import run_xtb_opt
from .settings import MAX_ATOMS_FF, MAX_ATOMS_XTB, TIMEOUT
from .utils import (
    hash_atoms,
    hash_object,
    molfile2ase,
    molfile2smiles
)


#working MM
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


#working MM
def load_soap_config(transition_info):
    # Get the filepath for the SOAP configuration
    soap_filepath = transition_info['soap_filepath']
    
    # Check if the file exists
    if not os.path.exists(soap_filepath):
        raise FileNotFoundError(f"SOAP configuration file not found at path: {soap_filepath}")
    
    # Read the SOAP configuration file
    with open(soap_filepath, 'r') as file:
        # Read the content of the file and execute it as Python code
        exec(file.read())
        
    # Check if the variable `SOAP` is defined in the file's content
    if 'SOAP' not in locals():
        raise ValueError(f"SOAP configuration variable 'SOAP' not defined in the file: {soap_filepath}")
    
    logging.debug(f"Loaded SOAP config for {transition_info['element']} {transition_info['orbital']}")
    # Return the SOAP configuration
    return locals()['SOAP']



def load_models_and_descriptors(transition_map):
    print("Entered load")

    cutoff = 4.25; dc = 0.5; sigma = 0.5
    zeta = 6
    SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma])),
        "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma]))}
    
    for transition_key, transition_info in transition_map.items():
        try:
            # Load SOAP config and store in cache
            soap_config = load_soap_config(transition_info)
            soap_config_cache.set(transition_key, soap_config, expire=None)
            logging.debug(f"SOAP config for {transition_key} loaded")
            
            # Load SOAP descriptor and store in cache
            if transition_info['element'] == "C":
                soap_descriptor = Descriptor(SOAP["C"])
                
            elif transition_info['element'] == "O":
                soap_descriptor = Descriptor(SOAP["O"])
                
            #soap_descriptor = Descriptor(soap_config)
            soap_descriptor_cache.set(transition_key, soap_descriptor, expire=None)
            logging.debug(f"SOAP descriptor for {transition_key} loaded")
            
            # Load ML model and store in cache
            ml_model = load_ml_model(transition_info)
            model_cache.set(transition_key, ml_model, expire=None)
            logging.debug(f"ML model for {transition_key} loaded")
            
        except Exception as e:
            logging.error(f"Error loading data for transition {transition_key}: {str(e)}")
            continue  # Optionally skip to the next iteration on error


#working MM
def test_model_and_soap_loading_at_startup(transition_map):

    load_models_and_descriptors()
    print("entered test")
    # Create a list to store test results
    test_results = []
    
    logging.debug("entered test function for transition_map")
    
    # Iterate through all transitions in the transition_map
    for transition_key, transition_info in transition_map.items():
        logging.debug(f"entered loop for {transition_info['element']} {transition_info['orbital']}")
        
        try:
            # Load the ML model and SOAP configuration
            soap_config = soap_config_cache.get(transition_key)
            soap_descriptor = soap_descriptor_cache(transition_key)
            model = model_cache.get(transition_key)
        
            # Perform some basic checks
            if soap_config and soap_descriptor and model:
                # If SOAP config, soap descriptor and ML model are loaded, the test is successful
                test_results.append((transition_key, "Success"))
            else:
                # If either of the 3 is not loaded, the test fails
                test_results.append((transition_key, "Failure: Either SOAP config, SOAP descriptor or ML model not loaded"))
        except Exception as e:
            # Capture any exceptions during loading and mark the test as failed
            test_results.append((transition_key, f"Failure: {str(e)}"))

    return test_results



#working MM
#get soap and ML model for a given transition_key, as defined in transition_map from .models.py, i.e C1s
def get_soap_and_model(transition_key: str):
    # Validate the transition
    TransitionValidator(transition = transition_key)
    
    # Retrieve the SOAP and model from the mapping
    transition_info = transition_map[transition_key]
    
    # Check the cache for SOAP configuration, construct if not in cache
    soap_config = soap_config_cache.get(transition_key)
    if soap_config is None:
        soap_config = load_soap_config(transition_info)
        soap_config_cache.set(transition_key, soap_config, expire = None)
        print("set cache soap config")
    logging.info(f"soap config for {transition_key} loaded")

    # Check the cache for SOAP descriptor, construct if not in cache
    soap_descriptor = soap_descriptor_cache.get(transition_key)
    if soap_descriptor is None:
        soap_descriptor = Descriptor(soap_config_cache.get(transition_key))
        soap_descriptor_cache.set(transition_key, soap_descriptor, expire=None)
        print("set cache soap desxcript config")
    logging.info(f"SOAP descriptor for {transition_key} loaded")

    # Check the cache for ML model, construct if not in cache
    ml_model = model_cache.get(transition_key)
    if ml_model is None:
        ml_model = load_ml_model(transition_info)
        model_cache.set(transition_key, ml_model, expire = None)
        print("set cache ml model")
    logging.info(f"ML model for {transition_key} loaded")
    
    return soap_config, soap_descriptor, ml_model


#working MM
def test_model_and_soap_loading(transition_map):
    # Create a list to store test results
    test_results = []
    
    logging.debug("entered test function for transition_map")
    
    # Iterate through all transitions in the transition_map
    for transition_key, transition_info in transition_map.items():
        logging.debug(f"entered loop for {transition_info['element']} {transition_info['orbital']}")
        try:
            # Load the ML model and SOAP configuration
            soap_config, soap_descriptor, ml_model = get_soap_and_model(transition_key)
            soap_config_c = soap_config_cache.get(transition_key)
            soap_descriptor_c = soap_descriptor_cache.get(transition_key)
            ml_model_c = model_cache.get(transition_key)
        
            # Perform some basic checks
            if soap_config_c and soap_descriptor_c and ml_model_c:
                # If SOAP config, soap descriptor and ML model are loaded, the test is successful
                test_results.append((transition_key, "Success"))
            else:
                # If either of the 3 is not loaded, the test fails
                test_results.append((transition_key, "Failure: Either SOAP config, SOAP descriptor or ML model not loaded"))
        except Exception as e:
            # Capture any exceptions during loading and mark the test as failed
            test_results.append((transition_key, f"Failure: {str(e)}"))
        
    return test_results



def get_max_atoms(method):
    if method == "GFNFF":
        return MAX_ATOMS_FF
    elif method == "GFN2xTB":
        return MAX_ATOMS_XTB
    elif method == "GFN1xTB":
        return MAX_ATOMS_XTB


def calculate_binding_energies(ase_mol: Atoms, transition_key):
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
    
    
    
    '''here to 
    # Retrieve SOAP descriptor from the cache
    soap_descriptor = soap_descriptor_cache.get(transition_key)
    if soap_descriptor is None:
        logging.error(f"SOAP descriptor not found in cache for transition {transition_key}")
        return []

    # Retrieve ML model from the cache
    ml_model = model_cache.get(transition_key)
    if ml_model is None:
        logging.error(f"ML model not found in cache for transition {transition_key}")
        return []
    here replaced by loading directly the models'''
    
    cutoff = 4.25; dc = 0.5; sigma = 0.5
    zeta = 6
    SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma])),
        "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma]))}
    
    #new version from here
    # Load SOAP descriptor and store in cache
            
    if element == "C":
        soap_descriptor = Descriptor(SOAP["C"])
                
    elif element == "O":
        soap_descriptor = Descriptor(SOAP["O"])
            
    #Load ML model and store in cache
    ml_model = load_ml_model(transition_info)
    
    ###to here new version
    
    
    

    # Calculate the SOAP descriptor for the molecule
    desc_data = soap_descriptor.calc(ase_mol)

    # Check if the descriptor data is available
    if 'data' not in desc_data:
        logging.error(f"No descriptor data found for molecule with transition {transition_key}")
        return []

    # Get the data from the descriptor object
    desc_molecule = desc_data['data']
    
    #X_new = []
    #for desc_atom in desc_molecule:
    #    X_new.append(desc_atom)
    #be, std = ml_model.predict(X_new, return_std=True)
    

    # Predict binding energies using the ML model
    be, std = ml_model.predict(desc_molecule, return_std=True)
    logging.info(f'Predicted {len(be)} binding energies for element {element}, orbital {orbital}, defined in {transition_key}')

    # Return a list of binding energy predictions
    return list(zip(be, std))


def run_xps_calculations(ase_mol: Atoms) -> dict:
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in run_xps_calculation, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    """
    From ASE molecule to predicted binding energies.
    """
    logging.debug(f"ASE molecule: {ase_mol}")

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
            logging.warning(f"No model found for element {element} in transition {transition}")

    return be_predictions



#@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_molfile(molfile, method) -> XPSResult:
    print("entered calculate")
    # Convert molfile to smiles and ASE molecule
    smiles = molfile2smiles(molfile)
    ase_mol, mol = molfile2ase(molfile, get_max_atoms(method))
    
    print("converted molfile to ase")
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in calculate_from_molfile, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    
    # Optimize the ASE molecule
    opt_result = run_xtb_opt(ase_mol, fmax=0.2, method=method)
    opt_ase_mol = opt_result.atoms
    if not isinstance(opt_ase_mol, Atoms):
        raise TypeError(f"After xtb optimization, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    print("ran xtb opt")
    # Run XPS calculations
    be_predictions = run_xps_calculations(opt_ase_mol)
    print("ran be predictions")
    
    # Extract binding energies and standard deviations
    binding_energies = []
    standard_deviations = []
    for transition_key, predictions in be_predictions.items():
        print("entered loop")
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









##################################
##################################



def ir_hash(atoms, method):
    return hash_object(str(hash_atoms(atoms)) + method)
