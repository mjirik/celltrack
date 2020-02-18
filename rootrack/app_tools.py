# /usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import os.path as op
import pyqtgraph
from loguru import logger
from typing import List
import ast


def create_icon(
    app_name: str, icon_filename=None, conda_env_name=None, package_name=None
):
    """

    :param app_name: Used for desktop icon name
    :param icon_filename: absolute path to icon usually:
        pathlib.Path(__file__).parent / pathlib.Path("app_icon512.ico")
    :param conda_env_name: conda environment. The app_name is used if conda_env_name is set to None.
    :param package_name: in `conda -m 'package_name'` . The app_name is used if conda_env_name is set to None.
    :return:
    """
    import platform

    # print(platform.system)
    if conda_env_name is None:
        conda_env_name = app_name

    # if icon_filename is None:
    #     icon_filename = app_name

    if Path(icon_filename).suffix == "":
        icon_filename += ".ico"

    if package_name is None:
        package_name = app_name

    if platform.system() == "Windows":

        # logo_fn2 = pathlib.Path(__file__).parent / pathlib.Path("scaffan_icon512.ico")

        # logo_fn = op.join(op.dirname(__file__), icon_filename)
        logo_fn = icon_filename
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")

        pth = Path.home()
        pth = pth / "Desktop" / Path(f"{app_name}.lnk")
        shortcut = shell.CreateShortcut(str(pth))
        # cmd
        # ln =  "call activate scaffan; {} -m scaffan".format(sys.executable)
        # C:\Windows\System32\cmd.exe /C "call activate anwaapp & pause &  python -m anwa & pause"
        # shortcut.TargetPath = sys.executable
        # shortcut.Arguments = f"-m {app_name}"
        shortcut.TargetPath = "cmd"
        # C:\Windows\System32\cmd.exe /C "call activate anwaapp & pause &  python -m anwa & pause"
        shortcut.Arguments = (
            f'/C "call activate {conda_env_name} & python -m {package_name} & pause" '
        )
        shortcut.IconLocation = "{},0".format(logo_fn)
        shortcut.Save()


def params_and_values(
    p: pyqtgraph.parametertree.Parameter, pth=None, dct={}, separator=";"
):
    """
    Get dict of all parameters. Key is the path to the parameter, value is value of the parameter.
    :param p:
    :param pth:
    :param dct:
    :param separator:  default ";"
    :return:
    """
    for name in p.getValues():
        # print(f"name: {name}, type {type(name)}")
        if pth is not None:
            pth_local = pth + separator + name
        else:
            pth_local = name
        # print(pth)
        ch = p.child(name)
        # print(f"name: {name}, type {type(ch)}")
        if type(ch) is pyqtgraph.parametertree.parameterTypes.SimpleParameter:
            dct[pth_local] = ch.value()
            # print(pth)
        else:
            params_and_values(ch, pth_local)

    return dct


def set_parameters_by_path(
    parameters: pyqtgraph.parametertree.Parameter,
    path_val_couple_list: List,
    parse_path=True,
    separator=";",
):
    """
    Set value to parameter.
    :param parameters: paramtree.Paramterr
    :param path_val_couple_list: list of couple [Path to parameter can be separated by ";", value]
    :param value:
    :param parse_path: Turn on separation of path by ";"
    :return:
    """

    for param_path, param_value in path_val_couple_list:
        if type(param_value) is str:
            value = ast.literal_eval(param_value)
        else:
            value = param_value
        logger.debug(f"param path={param_path}, ast value={value}")
        set_parameter_by_path(
            parameters,
            param_path,
            value=value,
            parse_path=parse_path,
            separator=separator,
        )


def set_parameter_by_path(
    parameters: pyqtgraph.parametertree.Parameter,
    param_path: str,
    value,
    parse_path=True,
    separator=";",
):
    """
    Set value to parameter.
    :param param_path: Path to parameter can be separated by ";"
    :param value:
    :param parse_path: Turn on separation of path by ";"
    :return:
    """
    logger.debug(f"Set {param_path} to {value}")
    if parse_path:
        param_path = param_path.split(separator)
    fnparam = parameters.param(*param_path)
    fnparam.setValue(value)
