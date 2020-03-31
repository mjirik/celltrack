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

from pathlib import Path
import os

@pytest.fixture
def path_tubhistw():
    return io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)
                                    # biology\orig\general


def test_read_tiff(path_tubhistw):
    path = path_tubhistw
    logger.debug(f"pth={path}")
    # path=path_tubhistw
    # path = io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
    ct = celltrack.celltrack_app.CellTrack()
    ct.set_input_file(path)

    # assert Path("g:/") in Path(path).parents


