# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import click.testing
import shutil
import pytest
import rootrack.main_cli
import io3d
from pathlib import Path
import os


def test_cli_add_image_data():
    """
    Add image data to common spreadsheet file.
    :return:
    """
    pth = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )

    logger.debug(f"pth={pth}, exists={Path(pth).exists()}")
    common_xlsx = Path("test_data.xlsx")
    logger.debug(f"expected_pth={common_xlsx}, exists: {common_xlsx.exists()}")
    if common_xlsx.exists():
        logger.debug(f"Deleting file {common_xlsx} before tests")
        os.remove(common_xlsx)

    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    runner.invoke(
        rootrack.main_cli.run,
        ["nogui", "-i", pth, "-o", common_xlsx, "-c", "#0000FF"],
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
        rootrack.main_cli.run,
        ["nogui", "-pp", "-p", "Processing;Intensity Normalization", "True", "-p", "Annotation;Upper Threshold", "1.5"],
    )
    assert result.output.find(" 'Annotation;Upper Threshold': 1.5,") > 0

def test_cli_print_params():
    """
    Add image data to common spreadsheet file.
    :return:
    """

    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    result = runner.invoke(
        rootrack.main_cli.run,
        ["gui", "-pp", "-p", "Processing;Intensity Normalization", "True", "-p", "Annotation;Upper Threshold", "1.5"],
    )
    assert result.output.find(" 'Annotation;Upper Threshold': 1.5,") > 0
