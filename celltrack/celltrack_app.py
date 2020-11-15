# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""

from loguru import logger
import sys
import os.path as op
path_to_script = op.dirname(op.abspath(__file__))
# pth = op.join(path_to_script, "../../celltrack/")
# sys.path.insert(0, pth)
import skimage.io
import skimage.measure
import seaborn as sns
from matplotlib import patches
from .tracker import FeatureTrackerManager, FeatureTracker

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
# import json
# import time
import platform
from typing import List, Union
import exsu
import celltrack
import numpy as np
import pandas as pd

from PIL import Image
from PIL.TiffTags import TAGS

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.widgets
import io3d.misc
from io3d import cachefile
from celltrack import tracker
logger.disable("exsu")
from PyQt5.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
pyqtRemoveInputHook()


class CellTrack:
    def __init__(self, skip_spreadshet_dump=False):

        self.report: Report = Report(check_version_of=["numpy", "scipy", "skimage"])
        self.report.set_persistent_cols({"celltrack_version": celltrack.__version__})
        # self.report.level = 50

        self.qapp = None

        # self.glcm_textures = satex.GLCMTextureMeasurement()
        # self.slide_segmentation.report = self.report
        self.tracker = tracker.Tracking(report=self.report)

        # self.lobulus_processing.set_report(self.report)
        # self.glcm_textures.set_report(self.report)
        # self.skeleton_analysis.set_report(self.report)
        # self.evaluation.report = self.report
        self.win: QtGui.QWidget = None
        self.skip_spreadsheet_dump = skip_spreadshet_dump
        self.patches = []
        # self.win = None
        self.cache = cachefile.CacheFile("~/.celltrack_cache.yaml")
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
                    {"name": "Pixel Size X", "type": "float", "value": 1.0,
                     "suffix": "m",
                     "siPrefix": True,
                     },
                    {"name": "Pixel Size Y", "type": "float", "value": 1.0,
                     "suffix": "m",
                     "siPrefix": True,
                     },
                    {"name": "Time Resolution", "type": "float", "value": 1.0,
                     "suffix": "s",
                     "siPrefix": False,
                     },
                    {"name": "Time Axis", "type": "int", "value": 0},
                    {"name": "X-Axis", "type": "int", "value": -2},
                    {"name": "Y-Axis", "type": "int", "value": -1},
                    # {"name": "Z-Axis", "type": "int", "value": 1},
                    {"name": "C-Axis", "type": "int", "value": 1, "tip": "Color axis"},
                    {"name": "Tracked Channel", "type": "int", "value": 0, "tip": "Channel used for tracking"},
                    {"name": "Preview Time", "type": "int", "value": -1, "tip": "Frame number used for preview"},

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
                "expanded": False,
                "children": [
                    # {"name": "Image Level", "type": "int", "value": 2},
                    self.tracker.parameters,
                    # self.intensity_rescale.parameters,
                    {
                        "name": "Report Level",
                        "type": "int",
                        "value": 50,
                        "tip": "Control ammount of stored images. 0 - all debug imagess will be stored. "
                               "100 - just important images will be saved.",
                    },
                    {
                        "name": "Debug Images",
                        "type": "bool",
                        "value": False,
                        # "suffix": "m",
                        "siPrefix": False,
                        "tip": "Show debug images",
                    },
                    {
                        "name": "Export Slices",
                        "type": "bool",
                        "value": False,
                        # "suffix": "m",
                        "siPrefix": False,
                        "tip": "Save slices as jpg files into the directory where the spreadsheet is located.",
                    },
                    {
                        "name": "Skip spreadsheet dump",
                        "type": "bool",
                        "value": skip_spreadshet_dump,
                        "tip": "Skip saving to output spreadsheet file.",
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
        self._n_files = None
        self.imagedata:np.ndarray = None
        self._should_clear_axes = True
        self.image2 = None

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
        default_dir = op.join(default_dir, "celltrack_data.xlsx")
        return default_dir

    def _get_file_info(self):
        pass
        # fnparam = Path(self.parameters.param("Input", "File Path").value())
        # if fnparam.exists() and fnparam.is_file():
        #     anim = scaffan.image.AnnotatedImage(str(fnparam))
        #     self.parameters.param("Input", "Data Info").setValue(anim.get_file_info())

    def _show_input_files_info(self):
        self.parameters.param("Input")
        sh = self.imagedata.shape if self.imagedata is not None else "None"
        msg = (
                f"Readed {self._n_files} files. Shape of first file={sh}"
        )
        logger.debug(msg)
        self.parameters.param("Input", "Data Info").setValue(msg)

    def run(self):
        logger.debug(f"report.df={self.report.df}")

        xaxis = self.parameters.param("Input", "X-Axis" ).value()
        yaxis = self.parameters.param("Input", "Y-Axis" ).value()
        taxis = self.parameters.param("Input", "Time Axis" ).value()
        caxis = self.parameters.param("Input", "C-Axis" ).value()
        cvalue= self.parameters.param("Input", "Tracked Channel" ).value()
        tvalue= self.parameters.param("Input", "Preview Time" ).value()
        # sl = list((np.asarray(self.imagedata.shape) - 1).astype(int)) # last image
        sl = list((np.asarray(self.imagedata.shape) / 2).astype(int)) # middle image
        time_resolution = float(self.parameters.param("Input", "Time Resolution").value())
        # sl = [0] * self.imagedata.ndim
        sl[int(caxis)] = int(cvalue)
        sl[int(xaxis)] = slice(None)
        sl[int(yaxis)] = slice(None)
        sl[int(taxis)] = slice(None)
        logger.debug(f"Channel={cvalue}, axis={caxis}")

        im = self.imagedata[tuple(sl)]
        xres = self.parameters.param("Input", "Pixel Size X").value()
        yres = self.parameters.param("Input", "Pixel Size Y").value()
        resolution = np.asarray([xres, yres], dtype=np.float)
        # self.image2.imshow(im)
        # self.report.init()
        self.add_std_data_to_rows_persistent()
        # self.report.add_cols_to_actual_row({})
        trackers = self.process_image(im, resolution=resolution, time_resolution=time_resolution)
        self.trackers_to_report(trackers, resolution, time_resolution, sl.copy(), int(caxis), int(taxis))
        logger.debug("trackers added to report")
        logger.debug("draw_output...")
        self._draw_output()

        skip_spreadsheet_dump = self.parameters.param("Processing", "Skip spreadsheet dump").value()
        if not skip_spreadsheet_dump:
            logger.debug("dump report...")
            self._dump_report()
        else:
            self.report.init()

    def _draw_output(self):
        if self.image2 is None:
            # no gui is initialized
            logger.debug("No output draw - no image")
            return
        dfs = self.report.df
        ax = self.image2.axes
        for ptch in ax.patches:
            ptch.remove()
        for txts in ax.texts:
            txts.remove()
        ax.patches=[]
        ax.texts=[]
        # remove all patches
        # for ptch in self.patches:
        #     ptch.remove()
            # ptch.set_visible(False)
        lns = ax.get_lines()
        for ln in lns:
            ln.remove()

        pal = sns.color_palette(None, len(dfs.id_obj.unique()))
        uu = sns.lineplot(data=self.report.df, x="x_px", y="y_px", hue="id_obj", legend=False, ax=ax, palette=pal)

        # logger.debug(f"type={type(uu)}")
        # import pdb;
        # pdb.set_trace()
        # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes

        dflast = dfs.groupby("id_obj", as_index=False).last()

        for i in range(len(dflast)):
            # dflast.bbox_0_x_px[i]
            # dflast.bbox_0_y_px[i]
            rect = patches.Rectangle(
                (dflast.bbox_left_px[i],
                 dflast.bbox_top_px[i]),
                dflast.bbox_right_px[i] - dflast.bbox_left_px[i],
                dflast.bbox_bottom_px[i] - dflast.bbox_top_px[i],
                linewidth=1, edgecolor=pal[i], facecolor='none'
            )
            self.patches.append(rect)
            # logger.debug(f"box {i} left={dflast.bbox_left_px}, top={dflast.bbox_top_px}")

            # ax = fig.axes[0]
            # Add the patch to the Axes
            ax.add_patch(rect)
            tx = ax.text(
                dflast.bbox_left_px[i],
                dflast.bbox_top_px[i],
                str(dflast.id_obj[i]),
                color=pal[i],
                fontsize="x-small"
            )
            self.patches.append(tx)
        self.image2.draw()

    def process_image(self, image:np.ndarray, resolution:np.ndarray, time_resolution:float): #, time_axis:int=None, z_axis:int=None, color_axis:int=None):
        """

        :param image: 3D image 1st axis is time, second and third is image space
        :param resolution:
        :param time_axis:
        :param z_axis:
        :param color_axis:
        :return:
        """
        logger.debug("calling process_image()")
        debug = self.parameters.param("Processing", "Debug Images").value()
        tvalue= self.parameters.param("Input", "Preview Time" ).value()
        trackers = self.tracker.process_image(
            image=image, resolution=resolution,
            time_resolution=time_resolution, qapp=self.qapp, debug=debug,
            preview_frame_id=tvalue
        )
        export = bool(self.parameters.param("Processing", "Export Slices").value())
        input_path = str(self.parameters.param("Input", "File Path").value())
        output_path = str(self.parameters.param("Output", "Common Spreadsheet File").value())
        if export:
            logger.debug("Export:")
            nm = Path(input_path).stem
            dire = Path(output_path).parent / nm
            dire.mkdir(parents=True, exist_ok=True)

            for idx, frame in enumerate(image):
                opth = dire / f"{nm}_{idx:05d}.jpg"
                logger.debug(opth)
                plt.imsave(opth, frame, cmap="gray")
        return trackers

    def trackers_to_report(self, trackers:FeatureTrackerManager,resolution:np.ndarray, time_resolution:float, sl:List[slice], caxis:int, taxis:int):

        if self.report:
            all_tracker_list = trackers.active_tracker_list.copy()
            all_tracker_list.extend(trackers.inactive_tracker_list)
            # for tr_id, tracker in enumerate(trackers.tracker_list):
            for tr_id, tracker in enumerate(all_tracker_list):
                logger.trace(f"tracker={tr_id}, len(tracker.frame)={len(tracker.region)}")
                for fr_i, fr in enumerate(tracker.region):
                    row = {
                        "id_obj": tracker.id,
                        "y_px": (tracker.region[fr_i].bbox[0] + tracker.region[fr_i].bbox[2])/2.,
                        "x_px": (tracker.region[fr_i].bbox[1] + tracker.region[fr_i].bbox[3])/2.,
                        "bbox_top_px": tracker.region[fr_i].bbox[0], # 0 y
                        "bbox_left_px": tracker.region[fr_i].bbox[1], # 0 x
                        "bbox_bottom_px": tracker.region[fr_i].bbox[2], # 1 y
                        "bbox_right_px": tracker.region[fr_i].bbox[3], # 1 x
                        "t_frame": tracker.frame[fr_i],
                        "area_px": tracker.region[fr_i].area,
                        # TODO prosím doplnit jméno předka
                        # "id_parent": str(tracker.parents),
                    }


                    #measure intensity
                    for c in range(self.imagedata.shape[caxis]):
                        sl[caxis] = c
                        sl[taxis] = tracker.frame[fr_i]
                        logger.trace(f"slice={sl}")
                        im = self.imagedata[tuple(sl)]
                        # region:skimage.measure.RegionProperties = tracker.region[fr_i]
                        region = tracker.region[fr_i]
                        # np.var(im[tracker.region])
                        minr, minc, maxr, maxc = region.bbox
                        imcr = im[minr:maxr, minc:maxc]

                        row[f"intensity mean in channel {c}"] = np.mean(imcr[region.image])
                        row[f"intensity var in channel {c}"] = np.var(imcr[region.image])

                    self.report.add_cols_to_actual_row(row)
                    self.report.finish_actual_row()
        df = self.report.df
        df["x [mm]"] = df["x_px"] * resolution[0]
        df["y [mm]"] = df["y_px"] * resolution[1]
        df["t [s]"] = df["t_frame"] * time_resolution
        df["bbox top [mm]"] = df["bbox_top_px"] * resolution[1]
        df["bbox right [mm]"] = df["bbox_right_px"] * resolution[0]
        df["bbox bottom [mm]"] = df["bbox_bottom_px"] * resolution[1]
        df["bbox left [mm]"] = df["bbox_left_px"] * resolution[0]
        df["area [mm^2]"] = df["bbox_left_px"] * resolution[0] * resolution[1]

    def select_file_gui(self):
        from PyQt5 import QtWidgets

        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser(r"d:\work\roots\data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        filter = "TIFF File(*.tiff, *.tif)"
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
        self._should_clear_axes = True
        self.show_image()

    def set_input_files(self, names):
        # self._n_readed_regions = 0
        # self._n_files_without_proper_color = 0
        self._n_files = len(names)
        for fn in names:
            self.set_input_file(fn)

    def set_input_file(self, fn: Union[Path, str]):
        fn = str(fn)
        fnparam = self.parameters.param("Input", "File Path")
        fnparam.setValue(fn)
        logger.debug("Set Input File Path to : {}".format(fn))
        self._add_tiff_file(fn)
        if self._n_files is None:
            self._n_files = 1
        self._show_input_files_info()

    def _add_tiff_file(self, fn:str ):
        with Image.open(fn) as img:
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag}
        key_value = [couplestring.split("=") for couplestring in meta_dict["ImageDescription"][0].split("\n")]
        image_description = {kv[0]: kv[1] for kv in key_value if len(kv) > 1}

        unit_multiplicator = 1
        if "unit" in image_description:
            if image_description["unit"] == "micron":
                unit_multiplicator = 0.000001
        try:
            xr = meta_dict["XResolution"]
            logger.debug(f"xr={xr}")
            xres = (xr[0][1] / xr[0][0]) * unit_multiplicator
            self.parameters.param("Input", "Pixel Size X").setValue(xres)
            yr = meta_dict["YResolution"]
            logger.debug(f"yr={xr}")
            yres = (yr[0][1] / yr[0][0]) * unit_multiplicator
            self.parameters.param("Input", "Pixel Size Y").setValue(yres)
        except Exception as e:
            logger.warning("Resolution not detected properly")

        img = skimage.io.imread(fn)
        self.imagedata:np.ndarray = img
        if self.imagedata.ndim < 4:
            self.set_parameter("Input;C-Axis", 0)

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

    def add_std_data_to_rows_persistent(self):
        # datarow = {}
        # datarow["Annotation ID"] = annotation_id
        inpath = Path(self.parameters.param("Input", "File Path").value())

        # self.anim.annotations.
        fn = inpath.parts[-1]
        # fn_out = self.parameters.param("Output", "Directory Path").value()
        # self.report.add_cols_to_actual_row(
        self.report.set_persistent_cols(
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
                "MicrAnt Version": celltrack.__version__,
                "timestamp": str(datetime.datetime.now())
                # "Output Directory Path": str(fn_out),
            }
        )
        # self.report.add_cols_to_actual_row(self.parameters_to_dict())

        # self.report.add_cols_to_actual_row(datarow)


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
    def _on_param_change(self):
        self.show_image()

    def show_image(self):
        if self.imagedata is None:
            return
        xaxis = self.parameters.param("Input", "X-Axis" ).value()
        yaxis = self.parameters.param("Input", "Y-Axis" ).value()
        taxis = self.parameters.param("Input", "Time Axis" ).value()
        caxis = self.parameters.param("Input", "C-Axis" ).value()
        cvalue= self.parameters.param("Input", "Tracked Channel" ).value()
        tvalue= self.parameters.param("Input", "Preview Time" ).value()
        # sl = list((np.asarray(self.imagedata.shape) / 2).astype(int)) # middle image
        sl = list((np.asarray(self.imagedata.shape) - 1).astype(int)) # last image
        # sl = [0] * self.imagedata.ndim
        sl[int(xaxis)] = slice(None)
        sl[int(yaxis)] = slice(None)
        sl[int(caxis)] = int(cvalue)
        sl[int(taxis)] = int(tvalue)

        im = self.imagedata[tuple(sl)]
        # self.image2.fig.clf()
        if self._should_clear_axes:
            self.image2.axes.cla()
            self.image2.axes.set_axis_off()
            self._should_clear_axes = False

        self.image2.imshow(im, cmap="gray")

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
        self.parameters.param("Input", "X-Axis").sigValueChanged.connect(self._on_param_change)
        self.parameters.param("Input", "Y-Axis").sigValueChanged.connect(self._on_param_change)
        self.parameters.param("Input", "Time Axis").sigValueChanged.connect(self._on_param_change)
        self.parameters.param("Input", "C-Axis").sigValueChanged.connect(self._on_param_change)
        self.parameters.param("Input", "Tracked Channel").sigValueChanged.connect(self._on_param_change)
        self.parameters.param("Input", "Preview Time").sigValueChanged.connect(self._on_param_change)

        # self.parameters.param("Processing", "Open output dir").setValue(True)
        t = ParameterTree()
        t.setParameters(self.parameters, showTop=False)
        t.setWindowTitle("pyqtgraph example: Parameter Tree")
        t.show()

        # print("run scaffan")
        win = QtGui.QWidget()
        win.setWindowTitle("CellTrack {}".format(celltrack.__version__))
        logo_fn = op.join(op.dirname(__file__), "celltrack_icon512.png")
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
        win.resize(1200, 800)
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
