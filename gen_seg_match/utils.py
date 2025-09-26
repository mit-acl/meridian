from copy import deepcopy

def segment_map_3d_to_2d(map_3d):
    """Convert a 3D map to a 2D map by averaging the z-axis."""
    map_2d = deepcopy(map_3d)
    to_rm = []
    for seg in map_2d.segments:
        seg.points[:,2] = 0.0
        seg._cleanup_points()
        if len(seg.points) < 2:
            to_rm.append(seg)
            continue
        try:
            seg.final_cleanup()
        except Exception as e:
            to_rm.append(seg)
    return map_2d