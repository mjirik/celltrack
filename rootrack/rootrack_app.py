# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""

from loguru import logger
import sys
import os.path as op
path_to_script = op.dirname(op.abspath(__file__))
# pth = op.join(path_to_script, "../../scaffan/")
# sys.path.insert(0, pth)

from PyQt5 import QtGui

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QVBoxLayout,
    QSizePolicy,
    QMessageBox,
    QWidget,
    QPushButton,
)
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from exsu.report import Report
import sys
import datetime
from pathlib import Path
import io3d.misc
from io3d import cachefile
# import json
# import time
import platform
from typing import List, Union
import exsu
import rootrack
import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.widgets


class MicrAnt:
    def __init__(self):

        self.report: Report = Report()
        # self.report.level = 50

        self.qapp = None

        # import scaffan.texture as satex
        # self.glcm_textures = satex.GLCMTextureMeasurement()
        # self.slide_segmentation = scaffan.slide_segmentation.ScanSegmentation()
        # self.slide_segmentation.report = self.report

        # self.lobulus_processing.set_report(self.report)
        # self.glcm_textures.set_report(self.report)
        # self.skeleton_analysis.set_report(self.report)
        # self.evaluation.report = self.report
        self.win: QtGui.QWidget = None
        # self.win = None
        self.cache = cachefile.CacheFile("~/.rootrack_cache.yaml")
        # self.cache.update('', path)
        common_spreadsheet_file = self.cache.get_or_save_default(
            "common_spreadsheet_file",
            self._prepare_default_output_common_spreadsheet_file(),
        )
        logger.debug(
            "common_spreadsheet_file loaded as: {}".format(common_spreadsheet_file)
        )
        params = [
            {
                "name": "Input",
                "type": "group",
                "children": [
                    {"name": "File Path", "type": "str"},
                    {"name": "Select", "type": "action"},
                    {"name": "Data Info", "type": "str", "readonly": True},
                    {"name": "Pixel Size X", "type": "float", "value": 1.0},
                    {"name": "Pixel Size Y", "type": "float", "value": 1.0},
                    {"name": "Time Axis", "type": "int", "value": 0},
                    {"name": "X-Axis", "type": "int", "value": 3},
                    {"name": "Y-Axis", "type": "int", "value": 2},
                    {"name": "Z-Axis", "type": "int", "value": 1},
                    # {
                    #     "name": "Automatic Lobulus Selection",
                    #     "type": "bool",
                    #     "value": False,
                    #     "tip": "Skip selection based on annotation color and select lobulus based on Scan Segmentation. ",
                    # },
                    # {
                    #     "name": "Annotation Color",
                    #     "type": "list",
                    #     "tip": "Select lobulus based on annotation color. Skipped if Automatic Lobulus Selection is used.",
                    #     "values": {
                    #         "None": None,
                    #         "White": "#FFFFFF",
                    #         "Black": "#000000",
                    #         "Red": "#FF0000",
                    #         "Green": "#00FF00",
                    #         "Blue": "#0000FF",
                    #         "Cyan": "#00FFFF",
                    #         "Magenta": "#FF00FF",
                    #         "Yellow": "#FFFF00",
                    #     },
                    #     "value": 0,
                    # },
                    # {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
                    # {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
                    # BatchFileProcessingParameter(
                    #     name="Batch processing", children=[]
                    # ),
                    # {
                    #     "name": "Advanced",
                    #     "type": "group",
                    #     "children": [
                    #         dict(name="Ignore not found color",type="bool", value=False,
                    #              tip="No exception thrown if color not found in the data"),
                    #     ]
                    # }
                ],
            },
            {
                "name": "Output",
                "type": "group",
                "children": [
                    # {
                    #     "name": "Directory Path",
                    #     "type": "str",
                    #     "value": self._prepare_default_output_dir(),
                    # },
                    # {"name": "Select", "type": "action"},
                    {
                        "name": "Common Spreadsheet File",
                        "type": "str",
                        "value": common_spreadsheet_file,
                    },
                    {
                        "name": "Select Common Spreadsheet File",
                        "type": "action",
                        "tip": "All measurements are appended to this file in addition to data stored in Output Directory Path.",
                    },
                ],
            },
            # {
            #     "name": "Annotation",
            #     "type": "group",
            #     "children": [
            #         {"name": "Annotated Parameter", "type": "str", "value": "", "color":"#FFFF00"},
            #         {"name": "Upper Threshold", "type": "float", "value": 2},
            #         {"name": "Lower Threshold", "type": "float", "value": 0},
            #     ],
            # },
            {
                "name": "Processing",
                "type": "group",
                "children": [
                    # {"name": "Image Level", "type": "int", "value": 2},
                    # self.intensity_rescale.parameters,
                    {
                        "name": "Report Level",
                        "type": "int",
                        "value": 50,
                        "tip": "Control ammount of stored images. 0 - all debug imagess will be stored. "
                               "100 - just important images will be saved.",
                    },
                ],
            },
            # { "name": "Comparative Annotation",
            #   "type": "group",
            #   "children": [
            #       {
            #           "name": "Left is lower",
            #           "type": "action",
            #           "tip": "Annotated parameter on left image is lower than on right image",
            #       },
            #       {
            #           "name": "Right is lower",
            #           "type": "action",
            #           "tip": "Annotated parameter on right image is lower than on left image",
            #       },
            #   ]},
            {"name": "Run", "type": "action"},
            # {"name": "Run", "type": "action"},
            # {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
            #     {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True,
            #      'suffix': 'V'},
            #     {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
            #     {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True,
            #      'suffix': 'Hz'},
            #
            # ]},
        ]
        self.parameters = Parameter.create(name="params", type="group", children=params)
        # here is everything what should work with or without GUI
        self.parameters.param("Input", "File Path").sigValueChanged.connect(
            self._get_file_info
        )

    def set_parameter(self, param_path, value, parse_path=True):
        """
        Set value to parameter.
        :param param_path: Path to parameter can be separated by ";"
        :param value:
        :param parse_path: Turn on separation of path by ";"
        :return:
        """
        logger.debug(f"Set {param_path} to {value}")
        if parse_path:
            param_path = param_path.split(";")
        fnparam = self.parameters.param(*param_path)
        fnparam.setValue(value)

    def _prepare_default_output_dir(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        # timestamp = datetime.datetime.now().strftime("SA_%Y-%m-%d_%H:%M:%S")
        timestamp = datetime.datetime.now().strftime("MA_%Y%m%d_%H%M%S")
        default_dir = op.join(default_dir, timestamp)
        return default_dir

    def _prepare_default_output_common_spreadsheet_file(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        # timestamp = datetime.datetime.now().strftime("SA_%Y-%m-%d_%H:%M:%S")
        # timestamp = datetime.datetime.now().strftime("SA_%Y%m%d_%H%M%S")
        default_dir = op.join(default_dir, "rootrack_data.xlsx")
        return default_dir

    def _get_file_info(self):
        pass
        # fnparam = Path(self.parameters.param("Input", "File Path").value())
        # if fnparam.exists() and fnparam.is_file():
        #     anim = scaffan.image.AnnotatedImage(str(fnparam))
        #     self.parameters.param("Input", "Data Info").setValue(anim.get_file_info())

    def _show_input_files_info(self):
        msg = (
                f"Readed {self._n_readed_regions} regions from {self._n_files} files. "
                + f"{self._n_files_without_proper_color} without proper color."
        )
        logger.debug(msg)
        self.parameters.param("Input", "Data Info").setValue(msg)

    def run(self):
        logger.debug(self.report.df)
        self._dump_report()
        # self.report.init()
        pass

    def process_image(self, image:np.ndarray, resolution:np.ndarray, time_axis:int=None, z_axis:int=None, color_axis:int=None):

        # TODO implementation
        pass

    def select_file_gui(self):
        from PyQt5 import QtWidgets

        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        filter = "TIFF File(*.tiff)"
        # fn, mask = QtWidgets.QFileDialog.getOpenFileName(
        #     self.win,
        #     "Select Input File",
        #     directory=default_dir,
        #     filter=filter
        # )

        # filter = "TXT (*.txt);;PDF (*.pdf)"
        file_name = QtGui.QFileDialog()
        file_name.setFileMode(QtGui.QFileDialog.ExistingFiles)
        names, _ = file_name.getOpenFileNames(
            self.win, "Select Input Files", directory=default_dir, filter=filter
        )
        self.set_input_files(names)

    def set_input_files(self, names):
        self._n_readed_regions = 0
        self._n_files_without_proper_color = 0
        for fn in names:
            self.set_input_file(fn)
        self._n_files = len(names)

    def set_input_file(self, fn: Union[Path, str]):
        fn = str(fn)
        fnparam = self.parameters.param("Input", "File Path")
        fnparam.setValue(fn)
        logger.debug("Set Input File Path to : {}".format(fn))
        self.add_ndpi_file(fn)
        self._show_input_files_info()


    def _dump_report(self):
        common_spreadsheet_file = self.parameters.param(
            "Output", "Common Spreadsheet File"
        ).value()
        excel_path = Path(common_spreadsheet_file)
        # print("we will write to excel", excel_path)
        filename = str(excel_path)
        logger.debug(f"Saving to excel file: {filename}")
        exsu.report.append_df_to_excel(filename, self.report.df)
        self.report.init()

    def add_std_data_to_row(self, inpath: Path, annotation_id):
        datarow = {}
        datarow["Annotation ID"] = annotation_id

        # self.anim.annotations.
        fn = inpath.parts[-1]
        # fn_out = self.parameters.param("Output", "Directory Path").value()
        self.report.add_cols_to_actual_row(
            {
                "File Name": str(fn),
                "File Path": str(inpath),
                # "Annotation Color": self.parameters.param(
                #     "Input", "Annotation Color"
                # ).value(),
                "Datetime": datetime.datetime.now().isoformat(" ", "seconds"),
                "platform.system": platform.uname().system,
                "platform.node": platform.uname().node,
                "platform.processor": platform.uname().processor,
                "MicrAnt Version": rootrack.__version__,
                # "Output Directory Path": str(fn_out),
            }
        )
        # self.report.add_cols_to_actual_row(self.parameters_to_dict())

        self.report.add_cols_to_actual_row(datarow)


    # def select_output_dir_gui(self):
    #     logger.debug("Deprecated call")
    #     from PyQt5 import QtWidgets
    #
    #     default_dir = self._prepare_default_output_dir()
    #     if op.exists(default_dir):
    #         start_dir = default_dir
    #     else:
    #         start_dir = op.dirname(default_dir)
    #
    #     fn = QtWidgets.QFileDialog.getExistingDirectory(
    #         None,
    #         "Select Output Directory",
    #         directory=start_dir,
    #         # filter="NanoZoomer Digital Pathology Image(*.ndpi)"
    #     )
    #     # print (fn)
    #     self.set_output_dir(fn)

    def select_output_spreadsheet_gui(self):
        from PyQt5 import QtWidgets

        default_dir = self._prepare_default_output_dir()
        if op.exists(default_dir):
            start_dir = default_dir
        else:
            start_dir = op.dirname(default_dir)

        fn = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Select Common Spreadsheet File",
            directory=start_dir,
            filter="Excel File (*.xlsx)",
        )[0]
        # print (fn)
        self.set_common_spreadsheet_file(fn)

    def set_common_spreadsheet_file(self, path):
        path = str(path)
        logger.info(" -- common_spreadsheet_file set to {}".format(path))
        fnparam = self.parameters.param("Output", "Common Spreadsheet File")
        fnparam.setValue(path)
        logger.info(" --  -- common_spreadsheet_file set to {}".format(path))
        self.cache.update("common_spreadsheet_file", path)
        # try:
        #     self.cache.update("common_spreadsheet_file", path)
        # except Exception as e:
        #     logger.debug("Problem with cache update")
        #     import traceback
        #     logger.debug(traceback.format_exc())
        logger.info("common_spreadsheet_file set to {}".format(path))

    # def gui_set_image1(self):
    #     self.image1.setPixmap(QtGui.QPixmap(logo_fn).scaled(100, 100))
    #     self.image1.show()
    # self.image2 = QtGui.QLabel()
    # self.image2.setPixmap(QtGui.QPixmap(logo_fn).scaled(100, 100))
    # self.image2.show()





    def start_gui(self, skip_exec=False, qapp=None):

        from PyQt5 import QtWidgets
        # import scaffan.qtexceptionhook

        # import QApplication, QFileDialog
        if not skip_exec and qapp == None:
            qapp = QtWidgets.QApplication(sys.argv)

        self.parameters.param("Input", "Select").sigActivated.connect(
            self.select_file_gui
        )
        # self.parameters.param("Output", "Select").sigActivated.connect(
        #     self.select_output_dir_gui
        # )
        self.parameters.param(
            "Output", "Select Common Spreadsheet File"
        ).sigActivated.connect(self.select_output_spreadsheet_gui)
        self.parameters.param("Run").sigActivated.connect(self.run)

        # self.parameters.param("Processing", "Open output dir").setValue(True)
        t = ParameterTree()
        t.setParameters(self.parameters, showTop=False)
        t.setWindowTitle("pyqtgraph example: Parameter Tree")
        t.show()

        # print("run scaffan")
        win = QtGui.QWidget()
        win.setWindowTitle("RooTrack{}".format(rootrack.__version__))
        logo_fn = op.join(op.dirname(__file__), "rootrack_icon512.png")
        app_icon = QtGui.QIcon()
        # app_icon.addFile(logo_fn, QtCore.QSize(16, 16))
        app_icon.addFile(logo_fn)
        win.setWindowIcon(app_icon)
        # qapp.setWindowIcon(app_icon)
        layout = QtGui.QGridLayout()
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        # layout.setColumnStretch(2, 3)
        win.setLayout(layout)
        pic = QtGui.QLabel()
        pic.setPixmap(QtGui.QPixmap(logo_fn).scaled(50, 50))
        pic.show()

        # self.image1 = PlotCanvas()
        # self.image1.axes.set_axis_off()
        # self.image1.imshow(plt.imread(logo_fn))

        # self.image1.plot()
        self.image2 = PlotCanvas()
        self.image2.axes.text(0.1, 0.6, "Load Tiff file")
        self.image2.axes.text(0.1, 0.5, "Check pixelsize")
        self.image2.axes.text(0.1, 0.4, "Run")
        # self.image2.axes.text(0.1, 0.3, "Use Comparative Annotation (optimal in further iterations)")
        self.image2.axes.set_axis_off()
        self.image2.draw()
        # self.image2.plot()

        # self.addToolBar(NavigationToolbar(self.image1, self))
        # self.image1.setPixmap(QtGui.QPixmap(logo_fn).scaled(100, 100))
        # self.image1.show()
        # self.image2 = QtGui.QLabel()
        # self.image2.setPixmap(QtGui.QPixmap(logo_fn).scaled(100, 100))
        # self.image2.show()
        # layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
        layout.addWidget(pic, 1, 0, 1, 1)
        layout.addWidget(t, 2, 0, 1, 1)
        # layout.addWidget(NavigationToolbar(self.image2, win),1, 2, 1, 1)
        layout.addWidget(NavigationToolbar(self.image2, win),1, 1, 1, 1)
        layout.addWidget(self.image2, 2, 1, 1, 1)
        # layout.addWidget(self.image2, 2, 2, 1, 1)
        # layout.addWidget(t2, 1, 1, 1, 1)

        win.show()
        win.resize(800, 600)
        self.win = win
        # win.
        self.qapp = qapp
        if not skip_exec:

            qapp.exec_()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # self.plot()
        self.imshow_obj = None

    # not used anymore
    # def plot(self, text=None, title=None):
    #
    #     # ax = self.figure.add_subplot(111)
    #     ax = self.axes
    #     # ax = self.figure.add_subplot(111)
    #     if text is not None:
    #         ax.text(0.5, 0.5, "Set Annotation Parameter")
    #     else:
    #         data = [np.random.random() for i in range(25)]
    #         ax.plot(data, "r-")
    #     # ax.text(0.5, 0.5, "Set Annotation Parameter")
    #     if title is not None:
    #         ax.set_title("PyQt Matplotlib Example")
    #     self.draw()

    def imshow(self, *args, title="", **kwargs):
        # data = [np.random.random() for i in range(25)]
        # ax = self.figure.add_subplot(111)
        ax = self.axes
        # ax.plot(data, 'r-')
        if self.imshow_obj is None:
            self.imshow_obj = ax.imshow(*args, **kwargs)
        else:
            self.imshow_obj = ax.imshow(*args, **kwargs)
            # self.imshow_obj.set_data(args[0])
        ax.set_title(title)
        self.draw()





class NoAnnotationParameter(Exception):
    pass
