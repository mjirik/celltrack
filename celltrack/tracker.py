from pyqtgraph.parametertree import Parameter
# from . import something
from exsu import Report
import numpy as np
import os
from pathlib import Path


path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))


class Tracking:
    def __init__(
            self,
            report: Report = None,
            pname = "Tracking",
            ptype = "group",
            pvalue = None,
            ptip = "Tracking parameters",

    ):

        # TODO Sem prosím všechny parametry.
        params = [

            {
                "name": "Example Integer Param",
                "type": "int",
                "value": 224,
                "suffix": "px",
                "siPrefix": False,
                "tip": "Value defines size of something",
            },
            {
                "name": "Example Float Param",
                "type": "float",
                "value": 0.00006,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Value defines size of something",
            },
        ]
        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        if report is None:
            report = Report()
            report.save = False
            report.show = False
        self.report: Report = report
        pass


    def init(self):
        model_path = path_to_script / 'models/my_best_model.model' #cesta k ulozenym modelum
        pass

    def process_image(self, image:np.ndarray, resolution:np.ndarray, time_resolution:float): #, time_axis:int=None, z_axis:int=None, color_axis:int=None):
    # def process_image(self, image:np.ndarray, resolution:np.ndarray, time_axis:int=None, z_axis:int=None, color_axis:int=None):
        """

        :param image: [z/t c x y] takhle ty dimenze? nebo jinak?
        :param resolution:
        :param time_axis:
        :param z_axis:
        :param color_axis:
        :return:
        """
        # TODO implementation


        # examples
        # get some parameter value
        sample_weight = float(self.parameters.param("Example Float Param").value())
        # self.report.

        out = {
            "id_obj": [1, 1, 2, 3],
            "x_px": [100, 100, 100, 100],
            "y_px": [100, 100, 105, 100],
            "t_frame": [1, 2, 2, 3],
            "id_parent": [None, None, [1], [1,2]],
        }
        return out
        # "x_mm": [0.1, 0.1, 0.100, 0.1],
        # "y_mm": [0.1, 0.1, 0.105, 0.1],
        # "t_s": [1.0, 2.0, 2.0, 3.0],
        # return (image > 0.5).astype(np.uint8)

