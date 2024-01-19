from rdkit import Chem
from rdkit.Chem import AllChem

from ase.io import read
from quippy.descriptors import Descriptor


Z = 6 # compute environment around atoms of a given Z. Here C atoms
descriptor = Descriptor("soap atom_sigma=0.5 n_max=3 l_max=3 cutoff=3.0 Z={:d} n_species=3 species_Z='1 6 8'".format(Z))


def xyz_to_X(mol):
    elements = []
    if ("C" in mol.symbols) == True:
        descMol = descriptor.calc(mol) #descriptor for each molecule

        if 'data' in descMol:
           desc_data = descMol['data'] #get the data from the descriptor object if exist
           print(len(desc_data))
           for element in desc_data:
               elements.append(element)
    return elements


def smiles_to_xyz(smiles):
# smiles to mol
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, 'temp.mol')    # Write RDKit molecule to a temporary file
    molecule = read('temp.mol') # Read the temporary file into ASE Atoms object
    return molecule


def smiles_to_BE(smiles:str):
    mol = smiles_to_xyz(smiles)
    soaps = xyz_to_X(mol)
    return soaps