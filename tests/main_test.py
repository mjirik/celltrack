# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
# import click.testing
import shutil
import pytest
# import celltrack.main_cli
import celltrack
import celltrack.celltrack_app
import io3d
import io3d.datasets
import sys

from pathlib import Path
import os
from PyQt5 import QtWidgets
# from datetime import datetime

qapp = QtWidgets.QApplication(sys.argv)


def test_read_tiff():
    path = io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    ct = celltrack.celltrack_app.CellTrack()
    ct.set_input_file(path)
    ct.start_gui(qapp=qapp, skip_exec=True)
    ct.run()


    # assert Path("g:/") in Path(path).parents

