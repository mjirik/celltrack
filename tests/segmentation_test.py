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
import glob
# from datetime import datetime

# from . import fixtures
# from fixtures import path_tubhistw, path_DIIVenus

qapp = QtWidgets.QApplication(sys.argv)

# @pytest.fixture
# def path_tubhistw():
#     return io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)


@pytest.fixture
def path_all_roots_data():
    # return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    # return io3d.datasets.join_path("biology/orig/roots/examples/R2D2-20x-1.tif", get_root=True)
    # return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    return Path(io3d.datasets.join_path("biology/orig/roots/", get_root=True))



def test_try_all(path_all_roots_data):
    path_all = path_all_roots_data
    logger.debug(path_all)
    fns = glob.glob(str(path_all / "**/*.tif"))
    logger.debug(fns)
    # path = io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)
    fnout = Path(__file__).parent / "test_try_all.xlsx"
    if fnout.exists():
        os.remove(fnout)

    for path in fns:
        logger.debug(f"file path={str(path)}")
        ct = celltrack.celltrack_app.CellTrack(skip_spreadshet_dump=False)
        ct.set_input_file(path)
        ct.set_parameter("Processing;Tracking;Frame Number", 5)
        ct.set_parameter("Output;Common Spreadsheet File", str(fnout))
        # ct.start_gui(qapp=qapp, skip_exec=True)
        ct.run()




    # assert Path("g:/") in Path(path).parents

