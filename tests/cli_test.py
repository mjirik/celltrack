# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import click.testing
import shutil
import pytest
import celltrack.main_cli
import io3d
from pathlib import Path
import os

@pytest.fixture
def path_tubhistw():
    return io3d.datasets.join_path("biology/orig/general/tubhiswt_C1.ome.tif", get_root=True)


@pytest.fixture
def path_DIIVenus():
    return io3d.datasets.join_path("biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True)


def test_cli_add_image_data(path_tubhistw):
    """
    Add image data to common spreadsheet file.
    :return:
    """
    pth = path_tubhistw
    # pth = io3d.datasets.join_path(
    #    "biology/orig/roots/examples/DIIVenus-20x-2.tif", get_root=True
    # )

    logger.debug(f"pth={pth}, exists={Path(pth).exists()}")
    common_xlsx = Path("test_data.xlsx")
    logger.debug(f"expected_pth={common_xlsx}, exists: {common_xlsx.exists()}")
    if common_xlsx.exists():
        logger.debug(f"Deleting file {common_xlsx} before tests")
        os.remove(common_xlsx)

    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    runner.invoke(
        celltrack.main_cli.run,
        ["nogui", "-i", pth, "-o", common_xlsx],
    )

    assert common_xlsx.exists()

def test_cli_print_params():
    """
    Add image data to common spreadsheet file.
    :return:
    """

    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    result = runner.invoke(
        celltrack.main_cli.run,
        ["nogui", "-pp", "-p", "Input;Time Axis", "1", "-p", "Processing;Report Level", "55"],
    )
    assert result.output.find("'Input;Time Axis': 1,") > 0

