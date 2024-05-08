# xps-service

This is a webservice built using [FastAPI](https://github.com/tiangolo/fastapi).

Allows to:

- Predict X-ray photoelectron binding energies of C 1s et O 1s, given a SMILES or molFile string
- Get conformers

## Usage

To be usable on Heroku or Dokku, which use `PORT` environmental variables, you need to either create this environmental variable or put it into an `.env` file. For local development, the `run_docker.sh` script uses `8091`.

```bash
./build_docker.sh # builds the docker image
./run_docker.sh # starts the service
```

For production, you may want to use Docker compose

```bash
docker-compose up
```

### Customization

You have the possibility to modify the photoemission transition that will be used a base of the prediction. This is done in settings.py. Several transition_maps could be defined, and the one to be used by the API should be loaded in xpsservice.py. Modification include:

-The option to import your own ML_model that should be pickled scikit GPR models
-The possibility to import your own descriptors which should be SOAP. A utiliy notebook allowing to convert a soap configuration to a saop file that could be further used to calculate the descriptor is saved in /SOAP_config

Note that adding elements to the transition_map implies adding dedicated GPR models as well as new SOAP. It also implies the modification of the existing SOAP config to account for the new interactions between the described elements

You have the option to customize the behavior of the app using environment variables:

- `MAX_ATOMS_XTB`: if the input contains more than this number of atoms, an error is thrown
- `TIMEOUT`: If the request takes longer than this time (in seconds) a `TimeOut` error is raised
- `CACHEDIR`: Sets the directory for the diskcache. It will be mounted by the docker container.

## Acknowledgments

This webservices heavily relies on [xtb](https://github.com/grimme-lab/xtb#citations) and ase. If you find this service useful and mention it in a publication, please also cite ase and xtb:

- [Ask Hjorth Larsen et al _J. Phys.: Condens. Matter_ **2017**, 29, 273002](https://iopscience.iop.org/article/10.1088/1361-648X/aa680e/meta)
- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher, S. Grimme
  _WIREs Comput. Mol. Sci._, **2020**, 11, e01493.
  DOI: [10.1002/wcms.1493](https://doi.org/10.1002/wcms.1493)
- S. Grimme, C. Bannwarth, P. Shushkov, _J. Chem. Theory Comput._, **2017**, 13, 1989-2009.
  DOI: [10.1021/acs.jctc.7b00118](https://dx.doi.org/10.1021/acs.jctc.7b00118)
- C. Bannwarth, S. Ehlert and S. Grimme., _J. Chem. Theory Comput._, **2019**, 15, 1652-1671.
  DOI: [10.1021/acs.jctc.8b01176](https://dx.doi.org/10.1021/acs.jctc.8b01176)
- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, _ChemRxiv_, **2019**, preprint.
  DOI: [10.26434/chemrxiv.8326202.v1](https://dx.doi.org/10.26434/chemrxiv.8326202.v1)
- S. Spicher and S. Grimme, _Angew. Chem. Int. Ed._, **2020**, 59, 15665–15673
  DOI: [10.1002/anie.202004239](https://doi.org/10.1002/anie.202004239)

It also uses RDKit and the conformer search proposed by Riniker/Landrum:

- RDKit: Open-source cheminformatics; http://www.rdkit.org
- S. Riniker and G. A. Landrum _J. Chem. Inf. Model._ **2015**, 55, 12, 2562–257 DOI: [10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)

## Docs

You find docs on `http://127.0.0.1:$PORT/v1/docs.`
