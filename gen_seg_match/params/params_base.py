import yaml


class ParamsBase:
    params_key = None

    def __init__(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            params = yaml.full_load(f)

        if cls.params_key and cls.params_key in params:
            params = params[cls.params_key]

        return cls(**params)
