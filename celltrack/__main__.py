# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import os.path as op
import sys

path_to_script = op.dirname(op.abspath(__file__))
pth = op.join(path_to_script, "../../scaffan/")
sys.path.insert(0, pth)

from loguru import logger
import sys
import click
from pathlib import Path
import ast
from . import main_cli

main_cli.run()
