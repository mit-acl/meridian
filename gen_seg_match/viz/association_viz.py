import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import List, Tuple
from robotdatapy.data import ImgData, PoseData
from enum import Enum
from functools import cached_property

from roman.viz import visualize_segment_on_img
from roman.object.segment import Segment

from aerial_segment import AerialSegment


class FrameOrientation(Enum):
    SIDE_BY_SIDE = "side_by_side"
    ABOVE_AND_BELOW = "above_and_below"


@dataclass
class AssociationVizParams:
    aerial_img: np.ndarray
    ground_img_data: ImgData
    ground_pose_data: PoseData
    matched_aerial_segments: List[AerialSegment]
    matched_ground_segments: List[Segment]
    output_path: str
    aerial_img_pixel_scale: float = 0.01
    aerial_img_crop: Tuple[int, int, int, int] = (0, 0, -1, -1)
    frame_orientation: FrameOrientation = FrameOrientation.ABOVE_AND_BELOW
    fps: int = 10
    line_width: int = 3
    min_segment_dist: float = 10.0
    show_segment_ids: bool = False

    @cached_property
    def aerial_img_aspect_ratio(self) -> float:
        if self.aerial_img_crop == (0, 0, -1, -1):
            self.aerial_img_crop = (
                0,
                0,
                self.aerial_img.shape[1],
                self.aerial_img.shape[0],
            )
        return self.aerial_img.shape[1] / self.aerial_img.shape[0]  # width / height

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
        t0 = np.min([seg.first_seen for seg in self.params.matched_ground_segments])
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
        for seg in self.params.matched_aerial_segments:
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

        for t in np.arange(t0, tf, 1 / self.params.fps):
            combined_img = np.zeros(self.params.output_img_shape, dtype=np.uint8)
            combined_img[: aerial_img.shape[0], : aerial_img.shape[1]] = aerial_img

            ground_pose_t = self.params.ground_pose_data.pose(t)
            ground_img = self.params.ground_img_data.img(t)
            seg_seen = [False for _ in self.params.matched_ground_segments]
            ground_outlines = [None for _ in self.params.matched_ground_segments]
            for i in range(len(self.params.matched_ground_segments)):
                seg = self.params.matched_ground_segments[i]
                if (
                    np.linalg.norm(
                        seg.center.flatten()
                        - self.params.ground_pose_data.position(t).flatten()
                    )
                    < self.params.min_segment_dist
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

            combined_img[
                self.params.ground_vid_pixel_origin[1] :,
                self.params.ground_vid_pixel_origin[0] :,
            ] = ground_img

            for i, seg in enumerate(self.params.matched_ground_segments):
                if not seg_seen[i]:
                    continue

                nearest_pixels = (aerial_outlines[i][0], ground_outlines[i][0])
                for pixels_ii in aerial_outlines[i]:
                    for pixels_jj in ground_outlines[i]:
                        if np.linalg.norm(pixels_ii - pixels_jj) < np.linalg.norm(
                            nearest_pixels[0] - nearest_pixels[1]
                        ):
                            nearest_pixels = (pixels_ii, pixels_jj)
                cv.line(
                    combined_img,
                    tuple(nearest_pixels[0].astype(np.int32)),
                    tuple(nearest_pixels[1].astype(np.int32)),
                    seg.viz_color[::-1],
                    self.params.line_width,
                )

            out.write(combined_img)

        out.release()
        cv.destroyAllWindows()
