# -*- coding: utf-8 -*-
import os

from fastapi.logger import logger


# Define transition_map dictionary
# Add more entries for other orbitals as needed
transition_map = {
    "C1s": {
        "element": "C",
        "orbital": "1s",
        "soap_filepath": os.path.abspath("xpsservice/SOAP_turbo_C1s.txt"),
        "model_filepath": os.path.abspath("xpsservice/XPS_GPR_C1s_xtb.pkl")
    },
    "O1s": {
        "element": "O",
        "orbital": "1s",
        "soap_filepath": os.path.abspath("xpsservice/SOAP_turbo_O1s.txt"),
        "model_filepath": os.path.abspath("xpsservice/XPS_GPR_O1s_xtb.pkl")
    },   
}


MAX_ATOMS_XTB = int(os.getenv("MAX_ATOMS_XTB", 100))
MAX_ATOMS_FF = int(os.getenv("MAX_ATOMS_FF", 100))
TIMEOUT = int(os.getenv("TIMEOUT", 100))
ALLOWED_ELEMENTS = derive_allowed_elements(transition_map)
ALLOWED_METHODS = ("GFNFF", "GFN2xTB", "GFN1xTB")
ALLOWED_FMAX = (0.000001, 0.1)

# Derive allowed elements from transition_map
def derive_allowed_elements(transition_map: dict) -> Set[str]:
    allowed_elements = {info["element"] for info in transition_map .values()}
    return allowed_elements


logger.info(
    f"Settings: MAX_ATOMS_XTB: {MAX_ATOMS_XTB}, TIMEOUT: {TIMEOUT}, ALLOWED_ELEMENTS: {ALLOWED_ELEMENTS}, ALLOWED_METHODS: {ALLOWED_METHODS}, ALLOWED_FMAX: {ALLOWED_FMAX}"
)
