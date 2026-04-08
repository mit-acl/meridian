import yaml
import os


class ParamsBase:
    params_key = None

    def __init__(self):
        pass

    @classmethod
    def load(cls, source: str, run: str = None):
        if os.path.isdir(source):
            return cls.from_yaml(f"{source}/{cls.params_key}.yaml", run=run)
        else:
            return cls.from_yaml(source, run=run)

    @classmethod
    def from_yaml(cls, yaml_file, run: str = None):
        assert cls.params_key is not None, "Class params key must be set"
        with open(yaml_file, "r") as f:
            params = yaml.full_load(f)

        if cls.params_key in params:
            params = params[cls.params_key]

        if run is not None and run in params:
            params = params[run]

        if run is not None:
            os.environ["RUN"] = run

        try:
            return cls(**params)
        except Exception:
            import warnings

            warnings.warn(f"Loading {cls.params_key} params failed, using defaults")
            return cls()
