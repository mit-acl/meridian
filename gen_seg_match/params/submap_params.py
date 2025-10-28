from dataclasses import dataclass
import yaml
from typing import List


@dataclass
class SubmapParams:

    creation_method: str = 'force_fill'              # Method for creating submaps: ('force_fill', 'adaptive', 'set_times')
    max_size: int = 40                              # Maximum number of segments in a submap (to save computation)
    segment_avg_time: bool = True                   # If true, use (first_seen + last_seen) / 2 for each segment reference time 
                                                    #    (only applicable if creation_method == 'force_fill' or if
                                                    #    creation_method == 'adaptive' and pruning_method == 'time')
    descriptor: str = None                          # Type of submap descriptor. Either 'none' or 'mean_semantic'.

    # the following is applicable only if creation_method == 'force_fill'
    overlap: int = int(0.5 * max_size)              # Number of overlapping segments between submaps

    # the following are applicable only if creation_method == 'adaptive' -----------
    radius: float = 15.0                             # Radius of submap in meters. If set to None, segments 
                                                    #    are never excluded from submaps based on distance 
                                                    #    (though they may still be pruned)
    center_dist: float = 10.0                        # Distance between submap centers in meters
    center_time: float = 50.0                        # time threshold between segments and submap center times
    pruning_method: str = 'distance'                # Metric for pruning segments in a submap: 
                                                    #    ('time', 'distance') -> max gets pruned
    # ----------------------------------------------------------------------------

    # the following are applicable only if creation_method == 'set_times' -----------
    submap_times: List[float] = None                 # List of center times for submaps

    def __post_init__(self):
        if type(self.descriptor) == str and self.descriptor.lower() == 'none':
            self.descriptor = None

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            params = yaml.full_load(f)
        return cls(**params)