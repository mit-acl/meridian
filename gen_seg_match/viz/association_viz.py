import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import List, Tuple
import robotdatapy as rdp
from robotdatapy.data import ImgData, PoseData
from enum import Enum
from functools import cached_property

from roman.viz import visualize_segment_on_img

from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import SegmentList, SegmentLine, SegmentPoint

# TODO: Figure out how to encode the height of the line
LINE_CAMERA_HEIGHT_OFFSET = 0.5


class FrameOrientation(Enum):
    SIDE_BY_SIDE = "side_by_side"
    ABOVE_AND_BELOW = "above_and_below"


@dataclass
class AssociationVizParams:
    aerial_img: np.ndarray
    ground_img_data: ImgData
    ground_pose_data: PoseData
    matched_aerial_segments: SegmentList
    matched_ground_segments: SegmentList
    output_path: str
    aerial_img_pixel_scale: float = 0.01
    aerial_img_crop: Tuple[int, int, int, int] = (0, 0, -1, -1)
    frame_orientation: FrameOrientation = FrameOrientation.ABOVE_AND_BELOW
    fps: int = 10
    line_width: int = 3
    min_segment_dist: float = 10.0
    show_segment_ids: bool = False
    connections_are_green: bool = True
    time_buffer: float = 5.0

    def __post_init__(self):
        if self.aerial_img_crop == (0, 0, -1, -1):
            self.aerial_img_crop = (
                0,
                0,
                self.aerial_img.shape[1],
                self.aerial_img.shape[0],
            )

    @cached_property
    def aerial_img_aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.aerial_img_crop
        return (x2 - x1) / (y2 - y1)  # width / height

    @cached_property
    def output_img_shape(self) -> Tuple[int, int]:
        if self.frame_orientation == FrameOrientation.SIDE_BY_SIDE:
            height = self.ground_img_data.height
            width = self.ground_img_data.width + self.aerial_viz_width
        elif self.frame_orientation == FrameOrientation.ABOVE_AND_BELOW:
            width = self.ground_img_data.width
            height = self.ground_img_data.height + self.aerial_viz_height
        else:
            raise ValueError(f"Unknown frame orientation: {self.frame_orientation}")
        return (height, width, 3)

    @cached_property
    def aerial_viz_width(self) -> int:
        if self.frame_orientation == FrameOrientation.SIDE_BY_SIDE:
            height = self.ground_img_data.height
            width = int(height * self.aerial_img_aspect_ratio)
        elif self.frame_orientation == FrameOrientation.ABOVE_AND_BELOW:
            width = self.ground_img_data.width
        else:
            raise ValueError(f"Unknown frame orientation: {self.frame_orientation}")
        return width

    @cached_property
    def aerial_viz_height(self) -> int:
        if self.frame_orientation == FrameOrientation.SIDE_BY_SIDE:
            height = self.ground_img_data.height
        elif self.frame_orientation == FrameOrientation.ABOVE_AND_BELOW:
            width = self.ground_img_data.width
            height = self.ground_img_data.height + int(
                width / self.aerial_img_aspect_ratio
            )
        else:
            raise ValueError(f"Unknown frame orientation: {self.frame_orientation}")
        return height

    @cached_property
    def ground_vid_pixel_origin(self) -> Tuple[int, int]:
        if self.frame_orientation == FrameOrientation.SIDE_BY_SIDE:
            return (self.aerial_viz_width, 0)
        elif self.frame_orientation == FrameOrientation.ABOVE_AND_BELOW:
            return (0, self.aerial_viz_height)
        else:
            raise ValueError(f"Unknown frame orientation: {self.frame_orientation}")

    @property
    def cropped_img(self):
        x1, y1, x2, y2 = self.aerial_img_crop
        return self.aerial_img[y1:y2, x1:x2]

    @property
    def cropped_img_pixel_scale(self) -> float:
        x1, y1, x2, y2 = self.aerial_img_crop
        return self.aerial_img_pixel_scale * (x2 - x1) / self.aerial_viz_width

    @property
    def aerial_crop_origin_m(self) -> Tuple[float, float]:
        x1, y1, _, _ = self.aerial_img_crop
        return (x1 * self.aerial_img_pixel_scale, y1 * self.aerial_img_pixel_scale)


class AssociationViz:
    def __init__(self, params: AssociationVizParams):
        self.params = params

    def create_video(self):
        t0 = (
            np.min([seg.first_seen for seg in self.params.matched_ground_segments])
            - self.params.time_buffer
        )
        tf = np.max([seg.last_seen for seg in self.params.matched_ground_segments])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(
            self.params.output_path,
            fourcc,
            self.params.fps,
            (self.params.output_img_shape[1], self.params.output_img_shape[0]),
        )

        aerial_img = self.params.cropped_img.copy()
        aerial_img = cv.resize(
            aerial_img, (self.params.aerial_viz_width, self.params.aerial_viz_height)
        )

        aerial_outlines = []
        # get segment outlines in aerial image
        for seg in self.params.matched_aerial_segments:
            if type(seg) is AerialSegment:
                convex_hull = seg.convex_hull_pixels(
                    img_pixel_scale=self.params.cropped_img_pixel_scale,
                    img_origin_m=self.params.aerial_crop_origin_m,
                )
                aerial_outlines.append(convex_hull)
                cv.polylines(
                    aerial_img,
                    [convex_hull],
                    True,
                    seg.viz_color[::-1],
                    self.params.line_width,
                )
            elif type(seg) is SegmentLine:
                if seg.num_endpoints == 2:
                    points = (
                        (
                            np.array(seg.endpoints)[:, :2]
                            - self.params.aerial_crop_origin_m
                        )
                        / self.params.cropped_img_pixel_scale
                    ).astype(np.int32)
                else:
                    pt = seg.get_point().flatten()[:2]
                    d = seg.get_direction().flatten()[:2]
                    far = 1e4
                    p1 = (
                        (pt - d * far) - self.params.aerial_crop_origin_m
                    ) / self.params.cropped_img_pixel_scale
                    p2 = (
                        (pt + d * far) - self.params.aerial_crop_origin_m
                    ) / self.params.cropped_img_pixel_scale
                    h, w = aerial_img.shape[:2]
                    ret, cp1, cp2 = cv.clipLine(
                        (0, 0, w, h), tuple(p1.astype(int)), tuple(p2.astype(int))
                    )
                    if not ret:
                        continue
                    points = np.array([cp1, cp2])
                aerial_outlines.append(points)
                aerial_img = cv.line(
                    aerial_img,
                    tuple(points[0]),
                    tuple(points[1]),
                    tuple(seg.color_from_id(order="bgr")),
                    self.params.line_width,
                )
            elif isinstance(seg, SegmentPoint):
                point = (
                    (
                        (seg.point.flatten()[:2] - self.params.aerial_crop_origin_m)
                        / self.params.cropped_img_pixel_scale
                    )
                    .astype(np.int32)
                    .flatten()
                )
                aerial_outlines.append(np.array([point]))
                aerial_img = cv.circle(
                    aerial_img,
                    tuple(point),
                    radius=5,
                    color=tuple(seg.color_from_id(order="bgr")),
                    thickness=-1,
                )

        # iterate through ground images
        for t in np.arange(t0, tf, 1 / self.params.fps):
            combined_img = np.zeros(self.params.output_img_shape, dtype=np.uint8)
            combined_img[: aerial_img.shape[0], : aerial_img.shape[1]] = aerial_img

            ground_pose_t = self.params.ground_pose_data.pose(t)
            ground_img = self.params.ground_img_data.img(t)[:, :, :3].copy()
            seg_seen = [False for _ in self.params.matched_ground_segments]
            ground_outlines = [None for _ in self.params.matched_ground_segments]

            # get ground segment outlines
            for i in range(len(self.params.matched_ground_segments)):
                seg = self.params.matched_ground_segments[i]
                if (
                    False
                    and (  # TODO: decide whether to support when segments are not lines
                        np.linalg.norm(
                            seg.center.flatten()
                            - self.params.ground_pose_data.position(t).flatten()
                        )
                        < self.params.min_segment_dist
                    )
                ):
                    seg_outline = seg.outline_2d(ground_pose_t)
                    if seg_outline is None:
                        continue
                    ground_img = visualize_segment_on_img(
                        seg,
                        ground_pose_t,
                        ground_img,
                        show_id=self.params.show_segment_ids,
                    )
                    seg_seen[i] = True
                    ground_outlines[i] = (
                        seg_outline + self.params.ground_vid_pixel_origin
                    )
                    ground_outlines[i] = ground_outlines[i].astype(np.int32)
                if seg.first_seen - self.params.time_buffer <= t <= seg.last_seen:
                    # if True:
                    if isinstance(seg, SegmentLine):
                        endpt3d = (np.zeros((3,)), np.zeros((3,)))
                        for ii in [0, 1]:
                            endpt3d[ii][0:2] = seg.endpoints[ii][0:2]
                            endpt3d[ii][2] = seg.height
                        res = self.get_line_pts_on_img(
                            endpt3d[0],
                            endpt3d[1],
                            ground_pose_t,
                        )
                        if res is None:
                            continue
                        pt1, pt2 = res
                        seg_seen[i] = True
                        ground_outlines[i] = (
                            np.array([pt1, pt2]) + self.params.ground_vid_pixel_origin
                        )
                        ground_outlines[i] = ground_outlines[i].astype(np.int32)
                        ground_img = cv.line(
                            ground_img,
                            tuple(pt1.astype(np.int32)),
                            tuple(pt2.astype(np.int32)),
                            seg.color_from_id(order="bgr"),
                            self.params.line_width,
                        )
                    elif isinstance(seg, SegmentPoint):
                        pt = self.get_point_on_img(
                            np.concatenate([seg.point.flatten(), [seg.height]]),
                            ground_pose_t,
                        )
                        if pt is None:
                            continue
                        seg_seen[i] = True
                        ground_outlines[i] = (
                            np.array([pt]) + self.params.ground_vid_pixel_origin
                        )
                        ground_outlines[i] = ground_outlines[i].astype(np.int32)
                        ground_img = cv.circle(
                            ground_img,
                            tuple(pt.astype(np.int32)),
                            radius=5,
                            color=seg.color_from_id(order="bgr"),
                            thickness=-1,
                        )

            combined_img[
                self.params.ground_vid_pixel_origin[1] :,
                self.params.ground_vid_pixel_origin[0] :,
            ] = ground_img

            # draw connections when both aerial and ground segments are seen
            for i, seg in enumerate(self.params.matched_ground_segments):
                if not seg_seen[i]:
                    continue

                # use the nearest pixels between aerial and ground outlines
                # to draw the connection line between
                # nearest_pixels = (aerial_outlines[i][0], ground_outlines[i][0])
                # for pixels_ii in aerial_outlines[i]:
                #     for pixels_jj in ground_outlines[i]:
                #         if np.linalg.norm(pixels_ii - pixels_jj) < np.linalg.norm(
                #             nearest_pixels[0] - nearest_pixels[1]
                #         ):
                #             nearest_pixels = (pixels_ii, pixels_jj)

                nearest_pixels = (
                    np.mean(aerial_outlines[seg.id], axis=0),
                    np.mean(ground_outlines[i], axis=0),
                )
                color = (
                    (0, 255, 0)
                    if self.params.connections_are_green
                    else seg.color_from_id(order="bgr")
                )
                cv.line(
                    combined_img,
                    tuple(nearest_pixels[0].astype(np.int32)),
                    tuple(nearest_pixels[1].astype(np.int32)),
                    color,
                    self.params.line_width,
                )

            out.write(combined_img)

        out.release()
        cv.destroyAllWindows()

    def get_point_on_img(self, point3d_odom, T_odom_cam):
        """Projects a 3D point in the odometry frame into a 2D point on the camera image."""
        if (
            np.linalg.norm(point3d_odom[:2] - T_odom_cam[:2, 3])
            > self.params.min_segment_dist
        ):
            return None

        point3d_cam = rdp.transform.transform(np.linalg.inv(T_odom_cam), point3d_odom)

        if point3d_cam[2] <= 0:  # behind camera
            return None

        point2d_cam = rdp.camera.xyz_2_pixel(
            point3d_cam.flatten(), self.params.ground_img_data.K
        )

        return point2d_cam.flatten()

    def get_line_pts_on_img(self, point3d_odom_1, point3d_odom_2, T_odom_cam):
        """Projects a 3D line in the odometry frame into a 2D line on the camera image."""
        point3d_odom_1[2] = (
            T_odom_cam[2, 3] - LINE_CAMERA_HEIGHT_OFFSET
        )  # TODO: Figure out how to encode the height of the line
        point3d_odom_2[2] = T_odom_cam[2, 3] - LINE_CAMERA_HEIGHT_OFFSET
        point3d_cam_1 = rdp.transform.transform(
            np.linalg.inv(T_odom_cam), point3d_odom_1
        )
        point3d_cam_2 = rdp.transform.transform(
            np.linalg.inv(T_odom_cam), point3d_odom_2
        )

        if point3d_cam_1[2] <= 0 and point3d_cam_2[2] <= 0:  # behind camera
            return None

        closest_line_point = SegmentLine.from_endpoints(
            -1, point3d_cam_1, point3d_cam_2
        ).closest_point_to_point(np.array([0.0, 0.0, 0.0]))
        if np.linalg.norm(closest_line_point) > self.params.min_segment_dist:
            return None
        if point3d_cam_1[2] <= 0 or point3d_cam_2[2] <= 0:  # one point behind camera
            # find the point on the line that intersects the image plane z=0
            # make point3d_cam_1 the point in front of the camera
            if point3d_cam_1[2] <= 0:
                point3d_behind = point3d_cam_1
                point3d_cam_1 = point3d_cam_2
                point3d_cam_2 = point3d_behind
            t = point3d_cam_1[2] / (point3d_cam_1[2] - point3d_cam_2[2])
            point3d_cam_2 = point3d_cam_1 + t * 0.99 * (point3d_cam_2 - point3d_cam_1)

        point2d_cam_1 = rdp.camera.xyz_2_pixel(
            point3d_cam_1.flatten(), self.params.ground_img_data.K
        )
        point2d_cam_2 = rdp.camera.xyz_2_pixel(
            point3d_cam_2.flatten(), self.params.ground_img_data.K
        )

        return point2d_cam_1.flatten(), point2d_cam_2.flatten()

    def project_point_to_image(self, point3d_odom, t):
        # get point3d_cam
        T_odom_cam = self.params.ground_pose_data.pose(t)
        point3d_cam = rdp.transform.transform(np.linalg.inv(T_odom_cam), point3d_odom)

        if point3d_cam[2] <= 0:  # behind camera
            return None

        # project to image
        point2d_cam = rdp.camera.xyz_2_pixel(
            point3d_cam.flatten(), self.params.ground_img_data.K
        )
        return point2d_cam
