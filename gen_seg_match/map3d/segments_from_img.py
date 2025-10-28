import numpy as np
import matplotlib.pyplot as plt
import robotdatapy.transform as rdpt
import robotdatapy.data as rdpd
import robotdatapy.camera as rdpc
import os
import cv2 as cv
from typing import List, Tuple
# os.environ['ROMAN_WEIGHTS'] = '/Users/masonbp/weights'

from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.observation import Observation
from roman.params.fastsam_params import FastSAMParams
from roman.object.segment import Segment
from roman.viz import visualize_segment_on_img

from gen_seg_match.segment.segment_types import GeneralSegment, SegmentPoint, SegmentLine, SegmentPlane

def is_line(segment: Segment):
    if segment.linearity < 0.9:
        return False
    mean, C = segment.gaussian
    U, S, Vt = np.linalg.svd(C)
    # linear_vec = U[:, 0]
    mean_center_points = segment.points - mean
    isotropic_points = (U @ mean_center_points.T).T
    dist_from_line = np.linalg.norm(isotropic_points[:, 1:], axis=1)
    rms_dist_from_line = np.sqrt(np.mean(dist_from_line**2))
    if rms_dist_from_line > 1.0:
        return False
    return True

def is_plane(segment: Segment):
    if segment.planarity > 0.9:
        return True
    if segment.scattering > 0.5:
        return False
    if segment.linearity + segment.planarity < 0.9:
        return False
    mean, C = segment.gaussian
    U, S, Vt = np.linalg.svd(C)
    # linear_vec = U[:, 0]
    mean_center_points = segment.points - mean
    isotropic_points = (U @ mean_center_points.T).T
    dist_from_line = np.linalg.norm(isotropic_points[:, 1:], axis=1)
    rms_dist_from_line = np.sqrt(np.mean(dist_from_line**2))
    if rms_dist_from_line > 1.0:
        return True
    return False

def get_line(segment: Segment, occluded_points: np.ndarray):
    default_dist = 20.0
    
    assert segment.points is not None and len(segment.points) >= 2, "Segment must have at least two points."
    mean, C = segment.gaussian
    C = C
    U, S, Vt = np.linalg.svd(C)
    # linear_vec = U[:, 0]
    
    assert segment.points.shape[1] == 3
    mean_center_points = segment.points - mean
    axis_aligned_points = (U.T @ mean_center_points.T).T
    # fig, ax = plt.subplots()
    # ax.plot(axis_aligned_points[:, 0], axis_aligned_points[:, 1], '.')
    # ax.set_aspect('equal')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(axis_aligned_points[:, 0], axis_aligned_points[:, 2], '.')
    # ax.set_aspect('equal')
    # plt.show()

    mean = mean.reshape((3,1))
    num_endpoints = 0
    if len(occluded_points) == 0:
        end_pts = [None, None]
        end_pts_dist_from_center = [-default_dist, default_dist]
    else:
        axis_aligned_occluded_points = (U.T @ (occluded_points - mean.flatten()).T).T
        pt0_axis_aligned = np.array([np.min(axis_aligned_points[:, 0]), 0., 0.])
        pt1_axis_aligned = np.array([np.max(axis_aligned_points[:, 0]), 0., 0.])
        dist_to_occluded_pt0 = np.linalg.norm(axis_aligned_occluded_points - pt0_axis_aligned, axis=1)
        dist_to_occluded_pt1 = np.linalg.norm(axis_aligned_occluded_points - pt1_axis_aligned, axis=1)
        
        end_pts = []
        end_pts_dist_from_center = []
        
        if np.min(dist_to_occluded_pt0) > 1.0:
            end_pts.append(U @ pt0_axis_aligned.reshape((3,1)) + mean)
            end_pts_dist_from_center.append(pt0_axis_aligned[0])
            num_endpoints += 1
        else:
            end_pts.append(None)
            end_pts_dist_from_center.append(-default_dist)

        if np.min(dist_to_occluded_pt1) > 1.0:
            end_pts.append(U @ pt1_axis_aligned.reshape((3,1)) + mean)
            end_pts_dist_from_center.append(pt1_axis_aligned[0])
            num_endpoints += 1
        else:
            end_pts.append(None)
            end_pts_dist_from_center.append(default_dist)

    # length = np.max(axis_aligned_points[:, 0]) - np.min(axis_aligned_points[:, 0])
    # pt0_centered_unrotated = np.array([[-length / 2], [0], [0]])
    # pt1_centered_unrotated = np.array([[length / 2], [0], [0]])
    pts_center_unrotated = [np.array([[i], [0], [0]]) for i in np.arange(end_pts_dist_from_center[0], end_pts_dist_from_center[1], 0.1)]
    line_pts = [U @ pt.reshape((3,1)) + mean for pt in pts_center_unrotated]

    vector = U[:, 0]
    # if there is only one endpoint, need to make sure the vector points the right directions
    if num_endpoints == 1:
        offset = end_pts[0] if end_pts[0] is not None else end_pts[1]
        offset = offset.flatten()
        if np.dot(vector, offset - mean.flatten()) > 0:
            vector = -vector
    else:
        offset = mean.flatten()

    return end_pts, offset, vector, line_pts

def get_segments_with_occlusion(pose: np.ndarray, raw_observations: List[Observation], 
        depth_img: np.ndarray, camera_params: rdpc.CameraParams) -> List[Segment]:
    depth_mask_thresh = 6.0
    edge_pixel_thresh = 10
    edge_mask = np.zeros((depth_img.shape[0], depth_img.shape[1]), bool)
    edge_mask[:edge_pixel_thresh,:] = True
    edge_mask[-edge_pixel_thresh:,:] = True
    edge_mask[:,:edge_pixel_thresh] = True
    edge_mask[:,-edge_pixel_thresh:] = True

    # print(len(raw_observations))

    for observation in raw_observations:
        occluded_points = observation.point_cloud[
            rdpt.transform(np.linalg.inv(pose), observation.point_cloud)[:, 2] 
            > depth_mask_thresh, :]

        occluded_pixels = np.array(np.where(np.bitwise_and(observation.mask.astype(bool), edge_mask))).T # (y, x)
        occluded_pixels_depths = depth_img[occluded_pixels[:,0], occluded_pixels[:,1]] * 1e-3
        occluded_pixels_3d_cam = rdpc.pixel_depth_2_xyz(occluded_pixels[:,1], occluded_pixels[:,0], occluded_pixels_depths, camera_params.K).T
        occluded_pixels_3d_world = rdpt.transform(pose, occluded_pixels_3d_cam)

        occluded_points = np.vstack([occluded_points, occluded_pixels_3d_world]) if occluded_points.shape[0] > 0 else occluded_pixels_3d_world
        observation.occluded_points = occluded_points


    segments = [Segment(observation=observation, camera_params=camera_params) for observation in raw_observations]
    for seg, obs in zip(segments, raw_observations):
        seg.occluded_points = obs.occluded_points

    return segments

def roman_segments_to_general_segments(segments: List[Segment]) -> List[GeneralSegment]:

    general_segments = []
    for seg in segments:
        if is_line(seg):
            end_pts, offset, vector, _ = get_line(seg, seg.occluded_points)
            if end_pts[0] is not None:
                general_segments.append(SegmentPoint(id=0, point=end_pts[0].flatten(), cos_feature=seg.semantic_descriptor))
            if end_pts[1] is not None:
                general_segments.append(SegmentPoint(id=0, point=end_pts[1].flatten(), cos_feature=seg.semantic_descriptor))
            general_segments.append(SegmentLine(id=seg.id, point=offset.flatten(), direction=vector.flatten(), 
                                                endpoints=[pt.flatten() if pt is not None else None for pt in end_pts], 
                                                cos_feature=seg.semantic_descriptor))
        elif is_plane(seg):
            pass
        else:
            general_segments.append(SegmentPoint(id=seg.id, point=seg.center.flatten(), cos_feature=seg.semantic_descriptor))

    return general_segments

fastsam_params = FastSAMParams(
    semantics=None, 
    device='cpu', 
    max_depth=8.0, 
    plane_filter_params=tuple([np.inf, 1.0, 0.2]), 
    conf=0.5,
    iou=0.5,
    max_mask_len_div=1,
    erosion_size=6
)