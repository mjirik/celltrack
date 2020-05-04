# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for celltrack application. It is usefull mainly for adding ndpi files to common xlsx file
"""

from loguru import logger
import sys
import click
from pathlib import Path
import ast

# print("start")
# from . import image

# print("start 5")
# print("start 6")

from celltrack import celltrack_app
from celltrack import app_tools
from celltrack import celltrack_app

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# print("Running __main__.py")
# @batch_detect.command(context_settings=CONTEXT_SETTINGS)
# @click.argument("image_stack_dir", type=click.Path(exists=True))
# @click.argument("working_dir", type=click.Path())
# @click.option("--create-icon", is_flag=True,
#               help="Create desktop icon"
#               )
@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option(
    "--log-level",
    "-ll",
    # type=,
    help="Set logging level",
    default="INFO",
)
@click.pass_context
def run(ctx, log_level, *args, **kwargs):
    logger.debug("CLI run...")
    if log_level is not None:
        logger.debug(f"changing log level to {log_level}")
        # try:
        #     log_level = int(log_level)
        # except ValueError as e:
        #     log_level = log_level.upper()
        # logger.remove()
        # i = logger.add(sys.stderr, level=log_level, colorize=True)
        # logger.debug("log level changed")
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        logger.debug("invoke subcommand gui")
        ctx.invoke(gui, *args, **kwargs)
        # a.main()
    else:
        logger.debug(f"invoke subcommand {ctx.invoked_subcommand} {args} {kwargs}")
        # Invoked automatically


@run.command(context_settings=CONTEXT_SETTINGS, help="Set persistent values")
@click.option(
    "--common-spreadsheet-file",
    help="Set path for common spreadsheet file.",
    type=click.Path(),
)
def set(common_spreadsheet_file=None):
    mainapp = celltrack_app.CellTrack()
    if common_spreadsheet_file is not None:
        mainapp.set_common_spreadsheet_file(path=common_spreadsheet_file)
        logger.info(f"Common spreadsheet file path is : {common_spreadsheet_file}")
        print(f"Common spreadsheet file path is : {common_spreadsheet_file}")


# def print_params(params):
#     algorithm.Scaffan().parameters.
#     params.


@run.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--params",
    "-p",
    multiple=True,
    default="",
    nargs=2,
    help='Set parameter. First argument is path to parameter separated by ";". Second is the value.'
    "python -m scaffan gui -p Processing,Show True",
)
@click.option("--print-params", "-pp", is_flag=True, help="Print parameters")
def gui(params, print_params):
    mainapp = celltrack_app.CellTrack()
    app_tools.set_parameters_by_path(mainapp.parameters, params)
    if print_params:
        import pprint
        pprint.pprint(app_tools.params_and_values(mainapp.parameters))
        exit()
    # for param in params:
    #     mainapp.set_parameter(param[0], value=ast.literal_eval(param[1]))
        # mainapp.parameters.param(*param[0].split(";")).setValue(ast.literal_eval(param[1]))
    mainapp.start_gui()


@run.command(
    context_settings=CONTEXT_SETTINGS, help="Create an icon on Windows platform"
)
def install():
    import platform

    print(platform.system)
    if platform.system() == "Windows":
        logger.info("Creating icon")

        logger.warning("TODO")
        from .app_tools import create_icon
        import pathlib

        logo_fn2 = pathlib.Path(__file__).parent / pathlib.Path("celltrack_icon512.ico")
        create_icon(
            "CellTrack", logo_fn2, conda_env_name="celltrack", package_name="celltrack"
        )


@run.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True),
    help="Path to input directory with video files.",
    default=None,
)
@click.option(
    "--common-xlsx",
    "-o",
    type=click.Path(),
    help="Path to common xlsx file.",
    default=None,
)
@click.option(
    "--params",
    "-p",
    multiple=True,
    default="",
    nargs=2,
    help='Set parameter. First argument is path to parameter separated by ";". Second is the value.'
    "python -m scaffan gui -p Processing,Show True",
)
@click.option("--print-params", "-pp", is_flag=True, help="Print parameters")
def nogui(input_path, common_xlsx, params, print_params):
    print("nogui")
    logger.debug(
        f"input path={input_path}, output_path={common_xlsx}, params={params}"
    )
    mainapp = celltrack_app.CellTrack()
    logger.debug(f"Celltrack created")
    app_tools.set_parameters_by_path(mainapp.parameters, params)
    if print_params:
        import pprint

        pprint.pprint(app_tools.params_and_values(mainapp.parameters))
        exit()
    # for param in params:
    #     logger.debug(f"param={param}")
    #     mainapp.parameters.param(*param[0].split(";")).setValue(ast.literal_eval(param[1]))

    logger.debug(f"common xlsx: {common_xlsx}")
    if common_xlsx is not None:
        mainapp.set_common_spreadsheet_file(common_xlsx)
    # logger.debug(f"common xlsx: {mainapp.report.com}")
    logger.debug(f"before input file: {input_path}")
    if input_path is not None:
        logger.debug(f"Setting new input file from CLI: {input_path}")
        mainapp.set_input_file(input_path)

    mainapp.run()


# def install():
