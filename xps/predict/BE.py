from rdkit import Chem
from rdkit.Chem import AllChem

from ase.io import read
from quippy.descriptors import Descriptor

import logging
import pickle

import os

MODEL = 'xps/MLmodels/XPS_GPR_C1s.pkl'
Z = 6 # compute environment around atoms of a given Z. Here C atoms
descriptor = Descriptor("soap atom_sigma=0.5 n_max=3 l_max=3 cutoff=3.0 Z={:d} n_species=3 species_Z='1 6 8'".format(Z))


def xyz_to_soap(mol):
    elements = []
    if ("C" in mol.symbols) == True:
        descMol = descriptor.calc(mol) #descriptor for each molecule

        if 'data' in descMol:
           desc_data = descMol['data'] #get the data from the descriptor object if exist
           print(len(desc_data))
           for element in desc_data:
               elements.append(element)
    return elements

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

def soap_to_BE(soap):
    model = pickle.load(open(MODEL, 'rb'))
    logging.info('Model loaded')

    be = model.predict(soap)
    return be

def molfile_to_BE(molfile:str):
    mol = molfile_to_xyz(molfile)
    soaps = xyz_to_soap(mol)
    be = soap_to_BE(soaps)
    return be

def smiles_to_BE(smiles:str):
    mol = smiles_to_xyz(smiles)
    soaps = xyz_to_soap(mol)
    be = soap_to_BE(soaps)
    return be

