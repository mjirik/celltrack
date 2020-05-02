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

# from . import fixtures
# from fixtures import path_tubhistw, path_DIIVenus

qapp = QtWidgets.QApplication(sys.argv)

@pytest.fixture
def path_tubhistw():
    return io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)


@pytest.fixture
def path_DIIVenus():
    # return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    # return io3d.datasets.join_path("biology/orig/roots/examples/R2D2-20x-1.tif", get_root=True)
    # return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    return io3d.datasets.join_path("biology/orig/roots/2channel/20200305-r2d2-PRE3.tif", get_root=True)


def test_read_tiff(path_DIIVenus):
    path = path_DIIVenus
    # path = io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)
    logger.debug(f"file path={str(path)}")
    ct = celltrack.celltrack_app.CellTrack()
    ct.set_input_file(path)
    ct.start_gui(qapp=qapp, skip_exec=True)
    ct.run()




    # assert Path("g:/") in Path(path).parents

