# file: config.py
# description: Central configuration for dataset paths, loaded from environment.
# author: María Victoria Anconetani
# date: 24/06/2026

import os
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ.get("CA_DATA_ROOT", "E:/CA EN CMR")
