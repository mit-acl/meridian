from dataclasses import dataclass
import yaml


@dataclass
class RomanConversionParams:
    # parameters for plane —————————————

    plane_max_e2_e1: float = 0.2  # Maximum e[2]/e[1] threshold to be considered a plane
    plane_max_e2_e0: float = 0.1  # Maximum e[2]/e[0] threshold to be considered a plane
    plane_rms_threshold: float = (
        0.5  # Minimum RMS distance from plane to still be considered a plane
    )

    # parameters for line ——————————————

    line_max_e1_e0: float = 0.1  # Maximum e[1]/e[0] threshold to be considered a line
    line_max_e2_e0: float = 0.1  # Maximum e[2]/e[0] threshold to be considered a line
    line_rms_threshold: float = (
        1.0  # Maximum RMS distance from line to still be considered a line
    )

    # ——————————————————————————————————

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            params = yaml.full_load(f)
        return cls(**params)
