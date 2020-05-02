from pyqtgraph.parametertree import Parameter
# from . import something
from exsu import Report
import numpy as np
import os
from pathlib import Path

import skimage
import numpy as np
from pathlib import Path
import skimage.io
from scipy import ndimage as ndi
from typing import List

from skimage import data
from skimage import measure
from skimage.exposure import histogram, equalize_adapthist
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))


class Tracker:
    def __init__(self, bbox, frame, region, status):
        self.bbox = []
        self.frame = []
        self.region = []
        self.status = status
        self.bbox.append(bbox)
        self.frame.append(frame)
        self.region.append(region)

        # TODO občas má tracker dva záznamy v jednom framu a zjevně jde o chybu.

    def status_off(self):
        self.status = 0

    def new_frame(self, bbox, frame, region, status):
        self.status = status
        self.bbox.append(bbox)
        self.frame.append(frame)
        self.region.append(region)


class TrackerManager:
    def __init__(self):
        self.tracker_list:List[Tracker] = []
        self.tracker_count = 0
        self.current_frame = -1
        self.iou_mat = []

    def add_tracker(self, tracker):
        self.tracker_list.append(tracker)
        self.tracker_count += 1

    def get_active(self):
        return [tracker for tracker in self.tracker_list if tracker.status == True]

    def next_frame(self, regions_count):
        self.current_frame += 1
        self.iou_mat.append(np.zeros((regions_count, self.tracker_count)))

    def count(self):
        return self.tracker_count

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def compute_iou(self, bbox, region_id):

        for tracker_id, tracker in enumerate(self.tracker_list):
            iou = self.bb_intersection_over_union(tracker.bbox[-1], bbox)
            self.iou_mat[self.current_frame][region_id][tracker_id] = iou


    def update_trackers(self, regions):

        for row_id, row in enumerate(self.iou_mat[self.current_frame]):
            max_pos = np.argmax(row)
            if row[max_pos] == 0:
                # self.tracker_list[row_id].status_off()
                self.add_tracker(Tracker(regions[row_id].bbox, self.current_frame, regions[row_id], 1))
            else:
                # print(len(regions), len(self.tracker_list), row_id, max_pos)
                self.tracker_list[max_pos].new_frame(regions[row_id].bbox, self.current_frame, regions[row_id], 1)

    def get_iou_mat(self):
        return self.iou_mat


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

    def find_cells(self, frame, split=0.96, disk_r=7):

        if type(frame) != np.uint8:
            cells = np.uint8((frame / np.max(frame)) * 255)
        else:
            cells = frame

        hist, hist_centers = histogram(cells)
        sums = np.cumsum(hist / hist.max())

        thr_lo = hist_centers[np.argmax(sums > (split * sum(hist / hist.max())))]
        thr_hi = thr_lo + 1

        markers = np.zeros_like(cells)
        markers[cells < thr_lo] = 1
        markers[cells > thr_hi] = 2

        elevation_map = sobel(cells)
        segmentation = watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)

        selem = disk(disk_r)
        morph = opening(segmentation, selem)

        #     # Find contours at a constant value of 0.8
        #     contours = measure.find_contours(morph, 0.8)

        labeled_cells, _ = ndi.label(morph)
        # intensity image is added to provide data for further statistics computation
        regions = measure.regionprops(labeled_cells, intensity_image=frame)

        return regions

    def cell_tracker(self, frames, regions=0) -> TrackerManager:

        manager = TrackerManager()

        for frame_id, frame in enumerate(frames):

            # print(frame_id, len(frame))
            manager.next_frame(len(frame))
            print(manager.get_iou_mat()[frame_id].shape)

            for region_id, region in enumerate(frame):

                if frame_id == 0:

                    manager.add_tracker(Tracker(region.bbox, frame_id, region, 1))

                else:

                    manager.compute_iou(region.bbox, region_id)

            if frame_id > 0:
                manager.update_trackers(frame)


        return manager

    def tracker_to_report(self, trackers:TrackerManager):
        pass
        # if self.report:
        #     for tr_id, tracker in enumerate(trackers.tracker_list):
        #         for i, fr in enumerate(tracker.frame):
        #             row = {
        #                 "id_obj": tr_id,
        #                 "y_px": (tracker.bbox[i][0] + tracker.bbox[i][2])/2.,
        #                 "x_px": (tracker.bbox[i][1] + tracker.bbox[i][3])/2.,
        #                 "bbox_0_y_px": tracker.bbox[i][0],
        #                 "bbox_0_x_px": tracker.bbox[i][1],
        #                 "bbox_1_y_px": tracker.bbox[i][2],
        #                 "bbox_1_x_px": tracker.bbox[i][3],
        #                 "t_frame": tracker.frame[i],
        #                 # TODO prosím doplnit jméno předka
        #                 "id_parent": None,
        #             }
        #             self.report.add_cols_to_actual_row(row)
        #             self.report.finish_actual_row()

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
        # print(image.shape)
        frames = []
        frames_c = len(image) - 1
        for idx, frame in enumerate(image):
            regions = self.find_cells(frame[:, :])
            frames.append(regions)
            print('Frame ' + str(idx) + '/' + str(frames_c) + ' done. Found ' + str(len(regions)) + ' cells.')
        # #     debug first four frames
        #     if idx > 3:
        #         break

        trackers = self.cell_tracker(frames)
        # self.tracker_to_report(trackers)

        # out = {
        #     "id_obj": [1, 1, 2, 3],
        #     "x_px": [100, 100, 100, 100],
        #     "y_px": [100, 100, 105, 100],
        #     "t_frame": [1, 2, 2, 3],
        #     "id_parent": [None, None, [1], [1, 2]],
        # }
        return trackers


        # "x_mm": [0.1, 0.1, 0.100, 0.1],
        # "y_mm": [0.1, 0.1, 0.105, 0.1],
        # "t_s": [1.0, 2.0, 2.0, 3.0],
        # return (image > 0.5).astype(np.uint8)

