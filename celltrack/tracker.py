from pyqtgraph.parametertree import Parameter
# from . import something
from loguru import logger
from exsu import Report
import numpy as np
import os
# from pathlib import Path
import copy

# from matplotlib import pyplot as plt
# import cv2
# import skimage
import numpy as np
from pathlib import Path
# import skimage.io
from scipy import ndimage as ndi
from typing import List

# from skimage import data
from skimage import measure
# from skimage.exposure import histogram, equalize_adapthist
from skimage.filters import sobel, threshold_niblack, gaussian, threshold_otsu
# from skimage.segmentation import watershed
# from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
# from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.util import random_noise, img_as_ubyte

path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))


class Tracker:
    def __init__(self, tid, bbox, frame, region, status):
        self.id = tid  #tracker_id
        self.bbox = []
        self.frame = []
        self.region = []
        self.parents = []
        self.status = status  # 0 - inactive, #1 - active
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

    def add_parent(self, parent):
        self.parents.append(parent)

    def set_id(self, tid):
        self.id = tid


class TrackerManager:
    def __init__(self):
        self.tracker_list:List[Tracker] = []
        self.old_tracker_list:List[Tracker] = []
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
        self.iou_mat.append(np.zeros((regions_count, len(self.tracker_list))))

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
        if iou < 0.3:
            iou = 0.0
        return iou

    def compute_iou(self, bbox, region_id):

        for tracker_id, tracker in enumerate(self.tracker_list):
            iou = self.bb_intersection_over_union(tracker.bbox[-1], bbox)
            self.iou_mat[self.current_frame][region_id][tracker_id] = iou


    def update_trackers(self, regions):

        for row_id, row in enumerate(self.iou_mat[self.current_frame]):

            hit = np.sum(row != 0)

            if hit == 0:  # no tracker hits object - new tracker
                self.add_tracker(Tracker(self.tracker_count, regions[row_id].bbox, self.current_frame, regions[row_id], 1))
            elif hit == 1:
                hit_pos = np.argmax(row)
                best_hit = row[hit_pos]
                tracker_hits = np.sum(self.iou_mat[self.current_frame][:, hit_pos])
                if tracker_hits == best_hit:  # only one tracker intersection - continue
                    # print(len(regions), len(self.tracker_list), row_id, hit_pos)
                    self.tracker_list[hit_pos].new_frame(regions[row_id].bbox, self.current_frame, regions[row_id], 1)
                elif tracker_hits > best_hit: #more objects for one tracker - split
                    self.tracker_list[hit_pos].status_off()
                    splinter = copy.copy(self.tracker_list[hit_pos])
                    splinter.new_frame(regions[row_id].bbox, self.current_frame, regions[row_id], 1)
                    splinter.set_id(self.tracker_count)
                    self.add_tracker(splinter)
            elif hit > 1: # more trackers for one object - merge
                trackers_hit = np.argwhere(row != 0)
                parents = []
                for id in trackers_hit:
                    self.tracker_list[int(id)].status_off()
                    prev_parents = self.tracker_list[int(id)].parents
                    prev_parents.append(self.tracker_list[int(id)].id)
                    parents.append(prev_parents)
                merged = Tracker(self.tracker_count, regions[row_id].bbox, self.current_frame, regions[row_id], 1)
                merged.add_parent(parents)
                self.add_tracker(merged)

        for col_id, column in enumerate(self.iou_mat[self.current_frame].T):
            if np.sum(column) == 0:  # tracker not hit anything
                self.tracker_list[col_id].status_off()

        for tracker in self.tracker_list:
            if tracker.status == 0:
                self.old_tracker_list.append(tracker)
                self.tracker_list.remove(tracker)



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
                "name": "Disk Radius",
                "type": "float",
                "value": 0.000002,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            },
            {"name": "Frame Number", "type": "int", "value": -1,
             "tip": "Maximum number of processed frames. Use -1 for all frames processing."},
            {"name": "Min. object size", "type": "float", "value": 0.00000000002, "suffix":"m^2", "siPrefix":True,
             "tip": "Maximum number of processed frames. Use -1 for all frames processing."},
            {
                "name": "Gaussian noise mean",
                "type": "float",
                "value": 0,
                "tip": "Gaussian noise added to remove scanner noise.",
            },

            {
                "name": "Gaussian noise variation",
                "type": "float",
                "value": 0.01,
                "tip": "Gaussian noise added to remove scanner noise.",
            },

            {
                "name": "Gaussian denoise sigma",
                "type": "int",
                "value": 1,
                "tip": "Sigma for Gaussian denoising.",
            },

            {
                "name": "Window size",
                "type": "float",
                "value": 1/8,
                "tip": "Size of the averaging windows for adaptive thresholding.",
            },

            # {
            #     "name": "Example Integer Param",
            #     "type": "int",
            #     "value": 224,
            #     "suffix": "px",
            #     "siPrefix": False,
            #     "tip": "Value defines size of something",
            # },
            # {
            #     "name": "Example Float Param",
            #     "type": "float",
            #     "value": 0.00006,
            #     "suffix": "m",
            #     "siPrefix": True,
            #     "tip": "Value defines size of something",
            # },
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

    def find_cells(self, frame, disk_r=9, gaus_noise=(0, 0.1), gaus_denoise=1, window_size=1/8, min_size_px=64):

        # if type(frame) != np.uint8:
        #     cells = ((frame / np.max(frame)) * 255).astype(np.uint8)
        # else:
        #     cells = frame

        cells=frame
        # im_noise = random_noise(cells, mean=gaus_noise[0], var=gaus_noise[1])
        # im_denoise = img_as_ubyte(gaussian(im_noise, sigma=gaus_denoise))
        # imh, imw = cells.shape
        # window = (int(imh * window_size), int(imw * window_size))
        # if window[0] % 2 == 0:
        #     window = (window[0] + 1, window[1])
        # if window[1] % 2 == 0:
        #     window = (window[0], window[1] + 1)

        # im_denoise = cells
        # binary_adaptive = threshold_niblack(im_denoise, window_size=window, k=0)
        binary_adaptive = threshold_otsu(frame)
        binim = cells > binary_adaptive
        import skimage.morphology
        binim = skimage.morphology.remove_small_objects(binim, min_size=min_size_px)

        selem = disk(disk_r//2)

        binim_o = opening(binim, selem)

        # elevation_map = sobel(cells)
        # segmentation = watershed(elevation_map, binim_o + 1)
        # segmentation = ndi.binary_fill_holes(segmentation - 1)

        # selem = disk(disk_r)
        # morph = opening(segmentation, selem)
        labeled_cells, _ = ndi.label(binim_o)

        regions = measure.regionprops(labeled_cells, intensity_image=frame)

        return regions

    def cell_tracker(self, frames, regions=0) -> TrackerManager:
        """

        :param frames: regionprops per frame
        :param regions:
        :return:
        """

        manager = TrackerManager()

        for frame_id, frame in enumerate(frames):

            # print(frame_id, len(frame))
            manager.next_frame(len(frame))
            print(manager.get_iou_mat()[frame_id].shape)

            for region_id, region in enumerate(frame):

                if frame_id == 0:

                    manager.add_tracker(Tracker(manager.tracker_count, region.bbox, frame_id, region, 1))

                else:

                    manager.compute_iou(region.bbox, region_id)

            if frame_id > 0:
                manager.update_trackers(frame)


        return manager

    # def tracker_to_report(self, trackers:TrackerManager):
    #     pass
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
        # sample_weight = float(self.parameters.param("Example Float Param").value())
        gaussian_m = float(self.parameters.param("Gaussian noise mean").value())
        gaussian_v = float(self.parameters.param("Gaussian noise variation").value())
        gaussian_sigma = float(self.parameters.param("Gaussian denoise sigma").value())
        window_size = float(self.parameters.param("Window size").value())
        disk_r_m = float(self.parameters.param("Disk Radius").value())
        disk_r_px = int(disk_r_m / np.mean(resolution))
        min_size_m2  = float(self.parameters.param("Min. object size").value())
        min_size_px = int(min_size_m2 / np.prod(resolution))
        logger.debug(f"pixelsize={resolution}")
        logger.debug(f"disk_r_px={disk_r_px}")
        logger.debug(f"min_size_px={min_size_px}, min_size_m2={min_size_m2}")
        frame_number = int(self.parameters.param("Frame Number").value())
        # self.report.
        # print(image.shape)
        frame_number = frame_number if frame_number > 0 else len(image)
        frames = []
        frames_c = len(image) - 1
        for idx, frame in enumerate(image):
            regions_props = self.find_cells(
                frame[:, :],
                disk_r=disk_r_px,
                gaus_noise=(gaussian_m, gaussian_v),
                gaus_denoise=gaussian_sigma,
                window_size=window_size,
                min_size_px=min_size_px

            )
            frames.append(regions_props)
            print('Frame ' + str(idx) + '/' + str(frames_c) + ' done. Found ' + str(len(regions_props)) + ' cells.')
        #     debug first four frames
            if idx >= frame_number:
                break

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

