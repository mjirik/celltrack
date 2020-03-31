# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
# import click.testing
import shutil
import pytest
import io3d

@pytest.fixture
def path_tubhistw():
    return io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)


@pytest.fixture
def path_DIIVenus():
    return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)
