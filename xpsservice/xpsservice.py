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
from .utils import smiles2molfile, molfile2smiles
from .errors import TooLargeError
from .ir import ir_from_molfile, ir_from_smiles
from .models import ConformerLibrary, ConformerRequest, IRRequest, IRResult

from .xps import *
from .models import ConformerLibrary, ConformerRequest, XPSRequest, XPSResult, transition_map

from .settings import MAX_ATOMS_FF, MAX_ATOMS_XTB

import logging

ALLOWED_HOSTS = ["*"]


app = FastAPI(
    title="EPFL-ISIC-XRDSAP: XPS webservice",
    description="Offers XPS binding energy prediction tool, based on GW simulation, and a Gaussian process ML model. Allowed elements/orbitals to be predicted are `C1s` and `O1s`. Hydrogens are taken into account but not predicted. Allowed methods for molecular geometry optimization are `GFNFF`, `GFN2xTB`, `GFN1xTB`",
    version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)


def max_atoms_error():
    return HTTPException(
        status_code=422,
        detail=f"This services only accepts structures with less than {MAX_ATOMS_FF} atoms for force-field calculations and {MAX_ATOMS_XTB} for xtb calculations.",
    )

# Define the ping route
@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/app_version")
@version(1)
def read_version():
    return {"app_version": __version__}


#TEST THE loading of the ml model and soap
@app.get("/test_model_and_soap_loading")
async def test_loading():
    # Call the test function
    test_results = test_model_and_soap_loading(transition_map)
    
    # Convert the results to a list of dictionaries
    response = [{"transition": result[0], "result": result[1]} for result in test_results]
    
    # Return the results as a JSON response
    return response


@app.post("/calculate")
async def calculate(request: XPSRequest):
    # Extract SMILES and molFile from the request
    logging.debug("ENTERED")
    smiles = request.smiles
    molFile = request.molFile
    logging.debug("loaded")

    # Perform conversions based on the provided input
    if smiles and not molFile:
        logging.debug("if smiles")
        # Convert SMILES to molFile using your function
        molFile = smiles2molfile(smiles)
        logging.debug("smiles conversion")
    elif molFile and not smiles:
        # Convert molFile to SMILES using your function
        smiles = molfile2smiles(molFile)
    elif not smiles and not molFile:
        raise HTTPException(status_code=400, detail="Either SMILES or molFile must be provided.")

    # Perform your logic using the converted data (smiles and molFile)

    # For example, call your calculate_from_molfile function
    #result = calculate_from_molfile(molFile, request.method, hash(smiles))

    # Return the result
    #return {"result": result}
    return {"result smiles": smiles, "result molFile": molFile}


#MM
@app.post("/xps", response_model=XPSResult)
@version(1)
def post_get_xps_spectrum(xpsrequest: XPSRequest):
    try:
        if xpsrequest.smiles:
            xps = xps_from_smiles(xpsrequest.smiles, xpsrequest.method)
        elif xpsrequest.molFile:
            xps = xps_from_molfile(xpsrequest.molFile, xpsrequest.method)
        else:
            raise HTTPException(
                status_code=422,
                detail="You need to provide either `molFile` or `smiles`",
            )
    except TooLargeError:
        raise max_atoms_error()
    except TimeoutError:
        raise HTTPException(status_code=500, detail="Calculation timed out.")
    return {"hello"}


@app.get("/xps", response_model=XPSResult)
@version(1)
def get_xps_spectrum(smiles: str, method: str = "GFNFF"):
    try:
        xps = xps_from_smiles(smiles, method)
    except TooLargeError:
        raise max_atoms_error()
    except TimeoutError:
        raise HTTPException(status_code=500, detail="Calculation timed out.")
    return xps


@app.post("/conformers", response_model=ConformerLibrary)
@version(1)
def post_conformers(conformerrequest: ConformerRequest):
    try:
        if conformerrequest.smiles:
            conformers = conformers_from_smiles(
                conformerrequest.smiles,
                conformerrequest.forceField,
                conformerrequest.rmsdThreshold,
                conformerrequest.maxConformers,
            )
        elif conformerrequest.molFile:
            conformers = conformers_from_molfile(
                conformerrequest.molFile,
                conformerrequest.forceField,
                conformerrequest.rmsdThreshold,
                conformerrequest.maxConformers,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="You need to provide either `molFile` or `smiles`",
            )
    except TooLargeError:
        raise max_atoms_error()
    except TimeoutError:
        raise HTTPException(status_code=500, detail="Calculation timed out.")
    return conformers


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
