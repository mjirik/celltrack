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
from scipy.optimize import linear_sum_assignment

path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))


class Tracker:
    def __init__(self, tid, bbox, frame, region, status):
        self.id = tid  # tracker_id
        self.bbox = []
        self.frame = []
        self.region = []
        self.parents = []
        self.direction = []
        self.status = status  # 0 - inactive, #1 - active
        self.bbox.append(bbox)
        self.frame.append(frame)
        self.region.append(region)

        # TODO občas má tracker dva záznamy v jednom framu a zjevně jde o chybu.

    def status_off(self):
        self.status = 0

    def status_dec(self):
        self.status -= 1

    def new_frame(self, bbox, frame, region, status):
        self.status = status
        self.bbox.append(bbox)
        self.frame.append(frame)
        self.region.append(region)

    def add_parent(self, parent):
        self.parents.append(parent)

    def set_id(self, tid):
        self.id = tid


class FeatureTracker:
    def __init__(self, uid, frame_id, max_inactivity):
        self.id = uid
        self.region = []
        self.features = []
        self.frame = []
        self.start_frame = frame_id
        self.last_active_frame = frame_id
        self.status = max_inactivity
        self.max_time_off = max_inactivity

    def add_region(self, region, frame_id, new_features=None):
        self.region.append(region)
        self.last_active_frame = frame_id
        self.frame.append(frame_id)
        # self.add_features(new_features)
        self.status = self.max_time_off

    def change_status(self, change):
        self.status += change

    def add_features(self, new_features):
        self.features.append(new_features)


class FeatureTrackerManager:
    def __init__(self, region_size_limit, inactivity_time, distance_measure_method='combined'):
        self.active_tracker_list: List[FeatureTracker] = []
        self.inactive_tracker_list: List[FeatureTracker] = []
        self.id_tracker = 0
        self.id_frame = -1
        self.max_inactivity = inactivity_time
        self.region_limit = region_size_limit
        self.tracker_to_obj_map = []
        self.distance_measure_method = distance_measure_method

    def add_tracker(self):
        self.active_tracker_list.append(
            FeatureTracker(uid=self.id_tracker, frame_id=self.id_frame, max_inactivity=self.max_inactivity))
        self.id_tracker += 1

    def clean_tracker_list(self):
        for tracker in self.active_tracker_list:
            if tracker.status == 0:
                self.active_tracker_list.remove(tracker)
                self.inactive_tracker_list.append(tracker)

    def next_frame(self, new_objects):
        self.id_frame += 1
        self.tracker_to_obj_map.append(self.create_distance_matrix(new_objects))
        self.update_trackers(new_objects, distance_type=self.distance_measure_method)
        self.clean_tracker_list()

    def create_distance_matrix(self, new_objects):

        dist_mat = np.zeros((len(new_objects), len(self.active_tracker_list)))
        for obj_id, obj in enumerate(new_objects):
            for tracker_id, tracker in enumerate(self.active_tracker_list):
                dist_mat[obj_id][tracker_id] = self.dist_calc(obj, tracker, self.distance_measure_method)

        return dist_mat

    def dist_calc(self, obj, tracker, distance_type='distance'):

        if distance_type == 'distance':
            return np.linalg.norm(np.array(obj.centroid) - np.array(tracker.region[-1].centroid))
        elif distance_type == 'features':
            return np.linalg.norm(np.array(obj.intensity_image.flatten())
                                  - np.array(tracker.region[-1].intensity_image.flatten()))
        elif distance_type == 'combined':
            f_dist = np.linalg.norm(np.array(obj.intensity_image.flatten())
                                    - np.array(tracker.region[-1].intensity_image.flatten()))
            dist = np.linalg.norm(np.array(obj.centroid) - np.array(tracker.region[-1].centroid))
            dist = dist if dist <= self.region_limit else 12000
            return dist * f_dist

        else:
            return np.inf

    def update_trackers(self, new_objects, method='hungarian', distance_type='distance'):
        # if distance_type == 'distance':
        #     threshold = self.region_limit
        # elif distance_type == 'features':
        #     threshold = self.region_limit
        # elif distance_type == 'combined':
        #     threshold = self.region_limit
        threshold = self.region_limit
        if method == 'hungarian':
            row, col = linear_sum_assignment(self.tracker_to_obj_map[-1])
            if len(row > 0):
                for i in range(0, len(row)):
                    object_id = row[i]
                    tracker_id = col[i]
                    distance = self.tracker_to_obj_map[-1][object_id, tracker_id]

                    if distance <= (threshold * (
                            (new_objects[object_id].max_intensity
                             - new_objects[object_id].min_intensity))):
                        self.active_tracker_list[tracker_id].add_region(new_objects[object_id], self.id_frame)
                    else:
                        self.active_tracker_list[tracker_id].change_status(-1)
                        self.add_tracker()
                        self.active_tracker_list[-1].add_region(new_objects[object_id], self.id_frame)
                tracker_unused = list(range(0, len(self.active_tracker_list)))
                for tracker_id in col:
                    tracker_unused.remove(tracker_id)
                for tracker_id in tracker_unused:
                    self.active_tracker_list[tracker_id].change_status(-1)

            else:
                for i in range(0, len(new_objects)):
                    self.add_tracker()
                    self.active_tracker_list[-1].add_region(new_objects[i], self.id_frame)


class TrackerManager:
    def __init__(self):
        self.tracker_list: List[Tracker] = []
        self.old_tracker_list: List[Tracker] = []
        self.tracker_count = 0
        self.current_frame = -1
        self.iou_mat = []
        self.dist_mat = []
        self.max_distance = 10

    def set_max_distance(self, max_dist):
        self.max_distance = max_dist

    def add_tracker(self, tracker):
        self.tracker_list.append(tracker)
        self.tracker_count += 1

    def get_active(self):
        return [tracker for tracker in self.tracker_list if tracker.status == True]

    def next_frame(self, regions_count):
        self.current_frame += 1
        self.iou_mat.append(np.zeros((regions_count, len(self.tracker_list))))
        self.dist_mat.append(np.zeros((regions_count, len(self.tracker_list))))

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

    def compute_dist(self, centroid, region_id, move_dist):
        move_sign = np.sign(np.array(move_dist))
        for tracker_id, tracker in enumerate(self.tracker_list):
            dist = np.array(centroid) - np.array(tracker.region[-1].centroid)
            dist_move_sign = np.sign(dist)
            if tracker.direction:
                move_sign = np.sign(np.array(tracker.direction[-1]))
            if np.all(move_sign == dist_move_sign):
                dist = np.linalg.norm(dist)
            else:
                dist = 12000
            self.dist_mat[self.current_frame][region_id][tracker_id] = dist

    # def update_trackers(self, regions):
    #
    #     for row_id, row in enumerate(self.iou_mat[self.current_frame]):
    #
    #         hit = np.sum(row != 0)
    #
    #         if hit == 0:  # no tracker hits object - new tracker
    #             self.add_tracker(Tracker(self.tracker_count, regions[row_id].bbox, self.current_frame, regions[row_id], 1))
    #         elif hit == 1:
    #             hit_pos = np.argmax(row)
    #             best_hit = row[hit_pos]
    #             tracker_hits = np.sum(self.iou_mat[self.current_frame][:, hit_pos])
    #             if tracker_hits == best_hit:  # only one tracker intersection - continue
    #                 # print(len(regions), len(self.tracker_list), row_id, hit_pos)
    #                 self.tracker_list[hit_pos].new_frame(regions[row_id].bbox, self.current_frame, regions[row_id], 1)
    #             elif tracker_hits > best_hit: #more objects for one tracker - split
    #                 self.tracker_list[hit_pos].status_off()
    #                 splinter = copy.copy(self.tracker_list[hit_pos])
    #                 splinter.new_frame(regions[row_id].bbox, self.current_frame, regions[row_id], 1)
    #                 splinter.set_id(self.tracker_count)
    #                 self.add_tracker(splinter)
    #         elif hit > 1: # more trackers for one object - merge
    #             trackers_hit = np.argwhere(row != 0)
    #             parents = []
    #             for id in trackers_hit:
    #                 self.tracker_list[int(id)].status_off()
    #                 prev_parents = self.tracker_list[int(id)].parents
    #                 prev_parents.append(self.tracker_list[int(id)].id)
    #                 parents.append(prev_parents)
    #             merged = Tracker(self.tracker_count, regions[row_id].bbox, self.current_frame, regions[row_id], 1)
    #             merged.add_parent(parents)
    #             self.add_tracker(merged)
    #
    #     for col_id, column in enumerate(self.iou_mat[self.current_frame].T):
    #         if np.sum(column) == 0:  # tracker not hit anything
    #             self.tracker_list[col_id].status_off()
    #
    #     for tracker in self.tracker_list:
    #         if tracker.status == 0:
    #             self.old_tracker_list.append(tracker)
    #             self.tracker_list.remove(tracker)

    def update_trackers_iou(self, regions):

        iou_mat = self.iou_mat[self.current_frame].copy()
        current_tracker_list = self.tracker_list.copy()
        object_list = regions.copy()
        new_tracker_list = []

        while 1:
            if np.amax(iou_mat) > 0:
                x, y = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
                iou_mat = np.delete(iou_mat, x, axis=0)
                iou_mat = np.delete(iou_mat, y, axis=1)
                current_tracker_list[y].new_frame(object_list[x].bbox, self.current_frame, object_list[x], 1)
                new_tracker_list.append(current_tracker_list[y])
                current_tracker_list.pop(y)
                object_list.pop(x)
            else:
                break
            if not (iou_mat.shape[0] * iou_mat.shape[1]):
                break

        if len(object_list) > 0:
            for region in object_list:
                new_tracker_list.append(
                    Tracker(self.tracker_count, region.bbox, self.current_frame, region, 1))
                self.tracker_count += 1

        self.tracker_list = new_tracker_list.copy()

    def update_trackers(self, regions):

        comp_mat = self.dist_mat[self.current_frame].copy()
        current_tracker_list = self.tracker_list.copy()
        object_list = regions.copy()
        new_tracker_list = []
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(comp_mat)

        dist_norm = 7 * np.linalg.norm(np.array(self.max_distance))
        while 1:
            if np.amin(comp_mat) <= dist_norm:
                x, y = np.unravel_index(np.argmin(comp_mat, axis=None), comp_mat.shape)
                comp_mat = np.delete(comp_mat, x, axis=0)
                comp_mat = np.delete(comp_mat, y, axis=1)
                current_tracker_list[y].new_frame(object_list[x].bbox, self.current_frame, object_list[x], 1)
                # current_tracker_list[y].direction.append(np.array(object_list[x].centroid)
                #                                          - np.array(current_tracker_list[y].region[-2].centroid))
                new_tracker_list.append(current_tracker_list[y])
                current_tracker_list.pop(y)
                object_list.pop(x)
            else:
                break
            if not (comp_mat.shape[0] * comp_mat.shape[1]):
                break

        if len(object_list) > 0:
            for region in object_list:
                new_tracker_list.append(
                    Tracker(self.tracker_count, region.bbox, self.current_frame, region, 1))
                self.tracker_count += 1

        if len(current_tracker_list) > 0:
            for tracker in current_tracker_list:
                tracker.status_dec()
                if tracker.status > -3:
                    new_tracker_list.append(tracker)
                    current_tracker_list.remove(tracker)

        # self.old_tracker_list.extend(current_tracker_list)
        self.tracker_list = new_tracker_list

    def get_iou_mat(self):
        return self.iou_mat


class Tracking:
    def __init__(
            self,
            report: Report = None,
            pname="Tracking",
            ptype="group",
            pvalue=None,
            ptip="Tracking parameters",

    ):

        # TODO Sem prosím všechny parametry.
        gaussian_sigma_xy = 0.000001000
        gaussian_sigma_t = 1
        params = [
            {
                "name": "Gaussian Sigma XY",
                "type": "float",
                "value": 0.00000100,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Filtration parameter. Should be smaller than the cell size and bigger than the noise size.",
            },
            {
                "name": "Gaussian Sigma T",
                "type": "float",
                "value": 0.25,
                "suffix": "s",
                "siPrefix": True,
                "tip": "Filtration in the dimension of time.",
            },
            {
                "name": "Min. Distance",
                "type": "float",
                "value": 0.000006,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Minimal distance between cell centers.",
            },
            {
                "name": "Num. Peaks",
                "type": "int",
                "value": 0,
                # "suffix": "m",
                # "siPrefix": True,
                "tip": "Limit detection to some number of cells. Find all cells if 'Num. Peaks' is 0.",
            },
            # {
            #     "name": "Method",
            #     "type": "list",
            #     "value": "Graph-Cut",
            #     "values": ["Graph-Cut", "Threshold", "Auto Threshold"]
            #     # "suffix": "m",
            #     # "siPrefix": True,
            #     # "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            # },
            {
                "name": "Disk Radius",
                "type": "float",
                "value": 0.000004,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Radius of area for intensity evaluation. Should be comparable size as the cell radius.",
            },
            # {
            #     "name": "Graph-Cut Resize",
            #     "type": "bool",
            #     "value": True,
            #     # "suffix": "m",
            #     "siPrefix": False,
            #     # "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            # },
            # {
            #     "name": "Graph-Cut Pixelsize",
            #     "type": "float",
            #     "value": 0.0000015,
            #     "suffix": "m",
            #     "siPrefix": True,
            #     # "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            # },
            # {
            #     "name": "Graph-Cut Pairwise Alpha",
            #     "type": "int",
            #     "value": 20,
            #     # "suffix": "m",
            #     # "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            # },
            # {
            #     "name": "Graph-Cut Multiscale",
            #     "type": "bool",
            #     "value": False,
            #     # "suffix": "m",
            #     "siPrefix": False,
            #     "tip": "Size of morphologic element used on preprocessing. Should be comparable size as the cell.",
            # },
            {"name": "Frame Number", "type": "int", "value": -1,
             "tip": "Maximum number of processed frames. Use -1 for all frames processing."},
            {"name": "Min. object size", "type": "float", "value": 0.00000000002, "suffix": "m^2", "siPrefix": True,
             "tip": "Maximum number of processed frames. Use -1 for all frames processing."},
            {
                # "name": "Threshold Offset",
                # "type": "bool",
                # "value": True,

                "name": "Threshold Mode",
                "type": "list",
                "value": "offset",
                "values": ["absolute", "offset"],
                "tip": "Use the 'offset' value to use threshold as an offset to the automatic per-frame Otsu theshold selection.",
            },
            {
                "name": "Threshold",
                "type": "int",
                "value": 0,
                "tip": "Minimal intensity value for cell in 'absolute' Threshold Mode. " + \
                       "If Threshold Mode is set to 'offset', the value is relative to automatic threshold selection.",
            },
            # {
            #     "name": "Gaussian noise mean",
            #     "type": "float",
            #     "value": 0,
            #     "tip": "Gaussian noise added to remove scanner noise.",
            # },
            #
            # {
            #     "name": "Gaussian noise variation",
            #     "type": "float",
            #     "value": 0.01,
            #     "tip": "Gaussian noise added to remove scanner noise.",
            # },
            #
            # {
            #     "name": "Gaussian denoise sigma",
            #     "type": "int",
            #     "value": 1,
            #     "tip": "Sigma for Gaussian denoising.",
            # },

            # {
            #     "name": "Window size",
            #     "type": "float",
            #     "value": 1/8,
            #     "tip": "Size of the averaging windows for adaptive thresholding.",
            # },

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
        self.debug_image = True
        self._thr_image = None
        pass

    def init(self):
        model_path = path_to_script / 'models/my_best_model.model'  # cesta k ulozenym modelum
        pass

    def find_cells2(self, frame, *args, **kwargs):
        binim_o = frame > 0
        labeled_cells, _ = ndi.label(binim_o)

        regions = measure.regionprops(labeled_cells, intensity_image=frame)

        return regions, binim_o, labeled_cells

    def find_cells_per_frame(self, frame, disk_r=9, gaus_noise=(0, 0.1), gaus_denoise=1, window_size=1 / 8,
                             min_size_px=64):

        # if type(frame) != np.uint8:
        #     cells = ((frame / np.max(frame)) * 255).astype(np.uint8)
        # else:
        #     cells = frame

        cells = frame
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

        # for i in range(0, frame.)
        import skimage.morphology
        binim = skimage.morphology.remove_small_objects(binim, min_size=min_size_px)

        selem = disk(disk_r // 2)

        binim_o = opening(binim, selem)

        # elevation_map = sobel(cells)
        # segmentation = watershed(elevation_map, binim_o + 1)
        # segmentation = ndi.binary_fill_holes(segmentation - 1)

        # selem = disk(disk_r)
        # morph = opening(segmentation, selem)
        labeled_cells, _ = ndi.label(binim_o)

        regions = measure.regionprops(labeled_cells, intensity_image=frame)

        return regions, binim_o

    def direction_mapping(self, regions):

        frame_centroids = []
        for frame in regions:
            centroids = []
            for obj in frame:
                centroids.append(np.array(obj.centroid))
            centroids = np.array(centroids)
            frame_centroids.append(np.mean(centroids, axis=0))

        from scipy.stats import linregress

        x = np.array(frame_centroids)[:, 1]
        y = np.array(frame_centroids)[:, 0]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_reg = intercept + slope * x
        mean_move_y = np.mean(y[1:] - y[0:-1])
        mean_move_x = np.mean(x[1:] - x[0:-1])
        # from matplotlib import pyplot as plt
        # plt.imshow(np.zeros((1200, 1200)))
        # plt.plot(x, y, 'o', label='original data')
        # plt.plot(x, intercept + slope * x, 'r', label='fitted line')
        # plt.legend()
        # plt.show()

        return frame_centroids, intercept, slope, mean_move_y, mean_move_x

    def cell_tracker(self, frames, max_distance=10, regions=0) -> TrackerManager:
        """

        :param frames: regionprops per frame
        :param regions:
        :return:
        """

        logger.info("Tracking...")
        # manager = TrackerManager()
        manager = FeatureTrackerManager(region_size_limit=max_distance, inactivity_time=2)
        # manager.set_max_distance(max_distance)

        len_frames = len(frames)
        for frame_id, frame in enumerate(frames):
            # print(frame_id, len(frame))
            # manager.next_frame(len(frame))

            manager.next_frame(frame)
            logger.debug(f"processing frame {frame_id}/{len_frames} - shape={manager.tracker_to_obj_map[frame_id].shape}")

            # for region_id, region in enumerate(frame):
            #
            #     if frame_id == 0:
            #
            #         manager.add_tracker(Tracker(manager.tracker_count, region.bbox, frame_id, region, 1))
            #
            #     else:
            #
            #         # manager.compute_iou(region.bbox, region_id)
            #         manager.compute_dist(region.centroid, region_id, max_distance)
            #
            # if frame_id > 0:
            #     manager.update_trackers(frame)

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

    def process_image(self, image: np.ndarray, resolution: np.ndarray, time_resolution: float, qapp=None,
                      preview_frame_id=0,
                      debug=False,
                      ):  # , time_axis:int=None, z_axis:int=None, color_axis:int=None):
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

        from skimage import (
            color, feature, filters, io, measure, morphology, segmentation, util
        )
        logger.info("Detection...")
        # 500 nm
        # resolution = [1, 0.000000500, 0.000000500]
        # 5000 nm = 5 um
        # gaussian_sigma_xy = 0.000001000
        # gaussian_sigma_t = 1
        is_offset = str(self.parameters.param("Threshold Mode").value()) == 'offset'
        thr = int(self.parameters.param("Threshold").value())
        gaussian_sigma_xy = float(self.parameters.param("Gaussian Sigma XY").value())
        gaussian_sigma_t = float(self.parameters.param("Gaussian Sigma T").value())
        disk_r_m = float(self.parameters.param("Disk Radius").value())
        disk_r_px = int(disk_r_m / np.mean(resolution))
        min_dist_m = float(self.parameters.param("Min. Distance").value())
        min_dist_px = int(min_dist_m / np.mean(resolution))
        logger.debug(f"disk_r_px={disk_r_px}")
        logger.debug(f"min_dist_r_px={min_dist_px}")
        num_peaks = int(self.parameters.param("Num. Peaks").value())

        sigma = [gaussian_sigma_t, gaussian_sigma_xy, gaussian_sigma_xy] / np.asarray(
            [time_resolution, resolution[0], resolution[1]])

        imgf = filters.gaussian(image, sigma=sigma, preserve_range=True)

        # if thr < 0:
        #     thr = filters.threshold_otsu(imgf)

        # examples
        # get some parameter value
        # sample_weight = float(self.parameters.param("Example Float Param").value())
        # gaussian_m = float(self.parameters.param("Gaussian noise mean").value())
        # gaussian_v = float(self.parameters.param("Gaussian noise variation").value())
        # gaussian_sigma = float(self.parameters.param("Gaussian denoise sigma").value())
        # window_size = float(self.parameters.param("Window size").value())
        frame_number = int(self.parameters.param("Frame Number").value())
        # self.report.
        # print(image.shape)
        frame_number = frame_number if frame_number > 0 else len(image)
        # method = "Graph-Cut"
        #
        # method = str(self.parameters.param("Method").value())
        # logger.debug(f"method={method}")
        # if method == "Graph-Cut":
        #     seg = self.do_segmentation_with_graphcut(
        #         image, resolution, time_resolution, qapp
        #     )
        # elif method == "Threshold":
        #     seg = self.do_segmentation_with_connected_threshold(
        #         image, resolution, time_resolution, qapp
        #     )
        # elif method == "Auto Threshold":
        #     pass

        # Gaussian filter

        frames = []
        frames_c = len(image) - 1
        # debug_images = True
        if self.debug_image:
            self._thr_image = np.zeros_like(image, dtype=np.uint8)
        # for idx, frame in enumerate(seg):

        # import sed3
        # ed = sed3.sed3(seg)
        # ed.show()
        import matplotlib.pyplot as plt

        for idx, frame in enumerate(image):
            # frame = image[idx,:,:]
            imf = imgf[idx, :, :]

            num_peaks = np.inf if num_peaks < 1 else num_peaks
            if is_offset:
                used_thr = filters.threshold_otsu(imf) + thr
            else:
                used_thr = thr
            # imthr = (imf > thr).astype(np.uint8)
            local_maxi = feature.peak_local_max(imf, indices=False,
                                                min_distance=min_dist_px, threshold_abs=used_thr,
                                                num_peaks=num_peaks
                                                )
            markers = measure.label(local_maxi)
            local_maxi_big = morphology.binary_dilation(markers, morphology.disk(min_dist_px // 2 - 1))
            # markers_big = measure.label(local_maxi_big)

            regions_props, thr_image, labeled = self.find_cells2(local_maxi_big * imf)
            #     frame[:, :],
            #     # disk_r=disk_r_px,
            #     # # gaus_noise=(gaussian_m, gaussian_v),
            #     # # gaus_denoise=gaussian_sigma,
            #     # # window_size=window_size,
            #     # min_size_px=min_size_px
            #
            # )

            positive_preview_frame_id = preview_frame_id if preview_frame_id >= 0 else frame_number + preview_frame_id
            if debug & (idx == positive_preview_frame_id):
                fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(frame, cmap="gray")  #
                ax[0].set_title("Original")
                ax[1].imshow(imf, cmap="gray")
                ax[1].set_title("Filtered")
                ax[3].imshow(markers)
                ax[3].set_title("Markers")
                ax[4].imshow(labeled)
                ax[4].set_title("Labeled markers")
                ax[5].imshow(frame, cmap="gray")
                ax[5].contour(labeled)
                ax[5].set_title("Labeled in original")
                fig.show()
                # plt.imshow(imf)
                plt.show()
            frames.append(regions_props)
            logger.debug(
                'Frame ' + str(idx) + '/' + str(frames_c) + ' done. Found ' + str(len(regions_props)) + ' cells.')
            if self.debug_image:
                self._thr_image[idx, :, :] = thr_image
            #     debug first four frames
            if idx >= frame_number:
                break

        # direction of movement by linear regression of detected object centroid in each frame
        centroids, intercept, slope, mean_move_y, mean_move_x = self.direction_mapping(frames)

        trackers = self.cell_tracker(frames, max_distance=min_dist_px + np.linalg.norm([mean_move_y, mean_move_x]))
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

    def do_segmentation_with_non_maximum_suppression(self, image: np.ndarray, resolution: np.ndarray,
                                                     time_resolution: float, qapp=None):
        from skimage import (
            color, feature, filters, io, measure, morphology, segmentation, util
        )
        # 500 nm
        # resolution = [1, 0.000000500, 0.000000500]
        # 5000 nm = 5 um
        gaussian_sigma_xy = 0.000001000
        gaussian_sigma_t = 1

        sigma = [gaussian_sigma_t, gaussian_sigma_xy, gaussian_sigma_xy] / np.asarray(
            [time_resolution, resolution[0], resolution[1]])

        imgf = filters.gaussian(image, sigma=sigma, preserve_range=True)

    def do_segmentation_with_auto(self, image: np.ndarray, resolution: np.ndarray, time_resolution: float, qapp=None):
        disk_r_m = float(self.parameters.param("Disk Radius").value())
        disk_r_px = int(disk_r_m / np.mean(resolution))
        min_size_m2 = float(self.parameters.param("Min. object size").value())
        min_size_px = int(min_size_m2 / np.prod(resolution))
        logger.debug(f"pixelsize={resolution}")
        logger.debug(f"disk_r_px={disk_r_px}")
        logger.debug(f"min_size_px={min_size_px}, min_size_m2={min_size_m2}")

        for idx, frame in enumerate(image):
            frame = image[idx, :, :]
            regions_props, thr_image = self.find_cells_per_frame(
                disk_r=disk_r_px,
                # gaus_noise=(gaussian_m, gaussian_v),
                # gaus_denoise=gaussian_sigma,
                # window_size=window_size,
                min_size_px=min_size_px
            )

    def do_segmentation_with_graphcut(self, image: np.ndarray, resolution: np.ndarray, time_resolution: float,
                                      qapp=None):
        import imcut.pycut as pspc
        import seededitorqt.seed_editor_qt
        from seededitorqt.seed_editor_qt import QTSeedEditor
        import imma.image_manipulation

        gc_pxsz_mm = float(self.parameters.param("Graph-Cut Pixelsize").value()) * 1000
        gc_pairwise_alpha = int(self.parameters.param("Graph-Cut Pairwise Alpha").value())
        gc_resize = bool(self.parameters.param("Graph-Cut Resize").value())
        gc_msgc = bool(self.parameters.param("Graph-Cut Multiscale").value())
        vxsz_mm = np.array([1.0, resolution[0] * 1000, resolution[1] * 1000])
        new_vxsz_mm = np.array([1.0, gc_pxsz_mm, gc_pxsz_mm])
        logger.debug(f"vxsz_mm={vxsz_mm}")
        logger.debug(f"new_vxsz_mm={new_vxsz_mm}")
        logger.debug(f"rel={vxsz_mm / new_vxsz_mm}")
        logger.debug(f"new sz ={image.shape * vxsz_mm / new_vxsz_mm}")
        if gc_resize:
            im_resized = imma.image_manipulation.resize_to_mm(
                image,
                voxelsize_mm=vxsz_mm,
                new_voxelsize_mm=new_vxsz_mm
            )
        else:
            im_resized = image

        logger.debug(f"im_resized.shape={im_resized.shape}")

        segparams = {
            # 'method':'graphcut',

            'method': 'multiscale_graphcut',
            'use_boundary_penalties': False,
            'boundary_dilatation_distance': 1,
            'boundary_penalties_weight': 1,
            'block_size': 8,
            'tile_zoom_constant': 1,
            "pairwise_alpha": gc_pairwise_alpha,

        }
        if gc_msgc:
            segparams["method"] = 'multiscale_graphcut'
        else:
            segparams["method"] = 'graphcut'
        igc = pspc.ImageGraphCut(im_resized, voxelsize=new_vxsz_mm, segparams=segparams)

        logger.debug(f"segparams={igc.segparams}")
        # seeds = igc.interactivity(qt_app=qapp)
        logger.debug(f"qapp[{type(qapp)}]={qapp}")

        pyed = QTSeedEditor(
            igc.img,
            seeds=igc.seeds,
            modeFun=igc.interactivity_loop,
            voxelSize=igc.voxelsize * 1000,
            volume_unit='',
            init_brush_index=0,
        )

        pyed.voxel_label.setText(
            f"%.2f x %.2f x %.2f [µm]"
            % tuple(pyed.voxel_size[np.array(pyed.act_transposition)])
        )
        # seededitorqt.seed_editor_qt.VIEW_TABLE = {"time": (2, 1, 0), "X": (1, 0, 2), "Y": (2, 0, 1)}
        # pyed.actual_view = "time"

        logger.debug("exec_()")
        pyed.exec_()
        logger.debug("exec is done")
        logger.debug(f"stats={igc.stats}")
        logger.debug(f"GC time ={igc.stats['gc time']}")
        logger.debug(f"segmentation[{type(igc.segmentation)}]={igc.segmentation}")

        seg = imma.image_manipulation.resize_to_shape(
            igc.segmentation,
            shape=image.shape
        )
        import scipy.stats
        seg = (1 - seg).astype(np.int8)
        logger.debug(f"unique={np.unique(seg, return_counts=True)}")
        return seg

    def do_segmentation_with_connected_threshold(self, image: np.ndarray, resolution: np.ndarray,
                                                 time_resolution: float, qapp=None):
        from imtools import segmentation as imsegmentation

        params = {
            # 'threshold': threshold,
            'inputSigma': 0.15,
            'aoi_dilation_iterations': 0,
            'nObj': 1,
            'biggestObjects': False,
            'useSeedsOfCompactObjects': True,
            'interactivity': True,
            'binaryClosingIterations': 2,
            'binaryOpeningIterations': 0,
            # 'seeds': seeds,
        }
        # params.update(inparams)
        # logger.debug("ogran_label ", organ_label)
        # target_segmentation = (self.segmentation == self.nlabels(organ_label)).astype(np.int8)
        import imma.image_manipulation as ima
        # target_segmentation = ima.select_labels(
        #     self.segmentation, organ_label, self.slab
        # )
        vxsz_mm = np.array([1.0, resolution[0] * 1000, resolution[1] * 1000])
        outputSegmentation = imsegmentation.vesselSegmentation(
            image,
            voxelsize_mm=vxsz_mm,
            # target_segmentation,
            segmentation=np.ones(image.shape, dtype=np.uint8),
            aoi_label=1,
            # aoi_label=organ_label,
            # forbidden_label=forbidden_label,
            # slab=self.slab,
            # debug=self.debug_mode,
            **params
        )
