# -*- coding: utf-8 -*-
"""
xps-service
webservice providing xps calculations
"""
# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

# Add imports here
from .xpsservice import *
