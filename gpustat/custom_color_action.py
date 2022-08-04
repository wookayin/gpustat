import argparse


class CustomColorAction(argparse.Action):
    def __call__(self, _parser, namespace, values, _option_strings=None):
        param_dict = getattr(namespace, self.dest)
        if param_dict is None:
            param_dict = {}

        try:
            k, v = values.split("=")
            param_dict[k] = v
        except ValueError:
            pass
        setattr(namespace, self.dest, param_dict)
