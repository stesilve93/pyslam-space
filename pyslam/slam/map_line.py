# MapLine: a persistent 3D line segment landmark
import numpy as np
from typing import Dict

class MapLine:
    _next_id = 0

    def __init__(self, p0, p1, id=None):
        if id is None:
            id = MapLine._next_id
            MapLine._next_id += 1

        self.id = id
        self.p0 = np.asarray(p0, dtype=float).reshape(3)
        self.p1 = np.asarray(p1, dtype=float).reshape(3)

        # Plücker representation
        self.d = self.p1 - self.p0
        self.m = np.cross(self.p0, self.d)

        # observations: dict of keyframe -> line_idx
        self.observations: Dict[object, int] = {}

        self.is_bad = False

    def add_observation(self, keyframe, line_idx):
        self.observations[keyframe] = line_idx

    def remove_observation(self, keyframe):
        if keyframe in self.observations:
            del self.observations[keyframe]

    def num_observations(self):
        return len(self.observations)

    def endpoints(self):
        return self.p0.copy(), self.p1.copy()

    def reproject_to_keyframe(self, kf):
        """Project both endpoints into image coords of keyframe kf."""
        pts = np.vstack([self.p0, self.p1])  # shape (2,3)
        uvs, zs = kf.project_points(pts)
        if uvs is None or np.any(zs <= 0):
            return None, None
        return uvs[0], uvs[1]

    def as_segment(self):
        return np.vstack([self.p0, self.p1])

