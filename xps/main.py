from typing import Union

from fastapi import FastAPI, HTTPException

from xps.models import XPS_prediction


app = FastAPI(
    title="XPS webservice",
    description="Offers xps predictions.",
    #version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)


@app.get("/xps", 
         response_model=XPS_prediction
         )
def get_xps_spectra(smiles: str):
    try:
        xps = XPS_prediction(
            energies=  [1,5],
            intensities = [1,6]
        )
    except:
        raise HTTPException(
                status_code=422,
                detail="You need to provide either `molFile` or `smiles`",
            )
    return xps