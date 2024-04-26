# xps-service

This docker returns predicted XPS binding energies (labelled BE) and predicted spectrum, based on the BEs fitting with Gaussians of standard deviation sigma. Currently only predicting C 1s and O 1s

Exposed functions of the API are:
/ping, a simple ping to test the API, returning pong
/SpectrumfromMolfile, expecting a "molfile" and optionally "sigma", returning the input molfile, the list of included and excluded elements, the calculated list of BE, and the gaussian broadened spectrum

/BEfromMolfile, expecting a "molfile", returning the input molfile, the list of included and excluded elements and the calculated list of BE

/SpectrumfromSMILES, expecting a "SMILES" and optionally "sigma", returning a molfile, the input smiles, the list of included and excluded elements, the calculated list of BE, and the gaussian broadened spectrum

/BEfromSMILES, expecting a "SMILES", returning a molfile, the input smiles, the list of included and excluded elements and the calculated list of BE
