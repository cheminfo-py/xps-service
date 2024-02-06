from rdkit import Chem
from rdkit.Chem import AllChem

from ase.io import read
from quippy.descriptors import Descriptor

import logging
import pickle

import os

MODEL = 'xps/MLmodels/XPS_GPR_C1s.pkl'

# First we create the descriptor object

Z = 6 
cutoff = 5; dc = 0.5; sigma = 0.5
zeta = 6
SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma])),
        "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma]))}

def xyz_to_soap_turbo(mol, element):
    if element not in ['C', 'O']:
        logging.info(f'Element "{element}" not implemented yet')
        return []

    desriptor =Descriptor(SOAP[element])

    elements = []
    if (element in mol.symbols) == True:
        descMol = desriptor.calc(mol) #descriptor for each molecule

        if 'data' in descMol:
           desc_data = descMol['data'] #get the data from the descriptor object if exist
           for element in desc_data:
               elements.append(element)
        return elements
    
    else:
        logging.info(f'Element "{element}" not in molecule')
        return []
    

def molfile_to_xyz(molfile:str):
    logging.info('molfile to xyz')
    temp_file = 'temp2.mol'
    with open(temp_file, 'w+') as f:
        f.write(molfile) #write molfile to temporary file
    mol = Chem.MolFromMolFile(temp_file) # Read temp file into RDKit molecule
    mol = Chem.AddHs(mol)# post-process molecule
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, temp_file)    # Write RDKit molecule to a temporary file
    molecule = read(temp_file) # Read the temporary file into ASE Atoms object
    return molecule

def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, 'temp.mol')    # Write RDKit molecule to a temporary file
    molecule = read('temp.mol') # Read the temporary file into ASE Atoms object
    return molecule

def soap_to_BE(soap, element):
    model_file = f'xps/MLmodels/XPS_GPR_{element}1s.pkl'

    model = pickle.load(open(model_file, 'rb'))
    logging.info('Model loaded')

    be = model.predict(soap)
    return be

def molfile_to_BE(molfile:str):
    bes = []
    mol = molfile_to_xyz(molfile)
    for element in ['C', 'O']:
        logging.info(element)
        soaps = xyz_to_soap_turbo(mol, element=element)
        if soaps != []:
            be = soap_to_BE(soaps, element)
            logging.info(be)
            for i in be:
                bes.append(i)
        
    return bes

def smiles_to_BE(smiles:str):
    mol = smiles_to_xyz(smiles)
    soaps = xyz_to_soap_turbo(mol, element= 'C')
    be = soap_to_BE(soaps)
    return be

