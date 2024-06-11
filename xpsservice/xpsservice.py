# -*- coding: utf-8 -*-
"""
xps-service.py
webservice providing xps calculations
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_versioning import VersionedFastAPI, version
from starlette.middleware import Middleware

from . import __version__
from .conformers import conformers_from_molfile, conformers_from_smiles
from .utils import smiles2molfile, molfile2smiles, smiles2ase, molfile2ase
from .errors import TooLargeError
from .models import ConformerLibrary, ConformerRequest
from .cache import *

from .xps import *
from .models import ConformerLibrary, ConformerRequest, XPSRequest, XPSResult

from .settings import MAX_ATOMS_FF, MAX_ATOMS_XTB, transition_map

import logging
import sys

ALLOWED_HOSTS = ["*"]


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


'''The desired transition-map should be defined here'''
selected_transition_map = transition_map

#Initial loading
#load_soap_configs_and_models(selected_transition_map)

app = FastAPI(
    title="EPFL-ISIC-XRDSAP: XPS webservice",
    description="Offers XPS binding energy prediction tool, based on simulation, and a Gaussian process ML model. Allowed elements/orbitals to be predicted are `C1s` and `O1s`. Hydrogens are taken into account but not predicted. Allowed methods for molecular geometry optimization are `GFNFF`, `GFN2xTB`, `GFN1xTB`",
    version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)


#load models and descriptors
@app.get("/load_soap_configs_and_models")
async def load_soap_configs_and_models_endpoint() -> bool:
    # Call the test function
    load_soap_configs_and_models(selected_transition_map)
     
     # Get the cache status from the function
    cache_status = check_cache_status(selected_transition_map)

    # Check if any cache failure exists
    result = not has_any_cache_failure(cache_status)
        
    return {"Loading successful:": result }
    

#Check the status of the cache / might have a mistake
@app.get("/check_cache_status")
def check_cache() -> Dict[str, Dict[str, bool]]:
    result = check_cache_status(selected_transition_map)
    return result



def max_atoms_error():
    return HTTPException(
        status_code=422,
        detail=f"This services only accepts structures with less than {MAX_ATOMS_FF} atoms for force-field calculations and {MAX_ATOMS_XTB} for xtb calculations.",
    )


# Predicts the binding energies
@app.post("/predict_binding_energies", response_model=XPSResult)
def predict_binding_energies_endpoint(request: XPSRequest):
    
    # Get the cache status from the function and optionnaly relaod
    cache_status = check_cache_status(selected_transition_map)
    if has_any_cache_failure(cache_status) == True:
        load_soap_configs_and_models(selected_transition_map)
       
    # Extract the input data
    smiles = request.smiles
    molfile = request.molFile
    method = request.method
    fmax = request.fmax
    energy_reference = request.energyReference
    
    # Perform conversions to molfile based on the provided input
    if smiles and not molfile:
        logging.debug("if smiles")
        smiles_input = True
        # Convert SMILES to molFile using your function
        molfile = smiles2molfile(smiles)
        logging.debug("smiles conversion")
    elif molfile and not smiles:
        smiles_input = False
    else:
        raise HTTPException(status_code=400, detail="Either SMILES or molFile must be provided.")
    print("converted format")
    # Perform calculations
    try:
        predictions = calculate_from_molfile(molfile, method, fmax, energy_reference, selected_transition_map)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if smiles_input == True:
        query = smiles
    else:
        query = molfile
    
     # Create an instance of XPSResult
    xps_result = XPSResult(
        query = query,
        predictions = predictions
    )

    # Return the result
    return xps_result


#checks version
@app.get("/app_version")
@version(1)
def read_version():
    return {"app_version": __version__}


app = VersionedFastAPI(
    app,
    version_format="{major}",
    prefix_format="/v{major}",
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_HOSTS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)