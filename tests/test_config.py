import copy
import unittest
from unittest.mock import patch

import yaml

from config import DEFAULT, load_config


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.defaults = yaml.safe_load(DEFAULT.read_text())

    def load(self, data):
        with patch("config.yaml.safe_load", side_effect=[self.defaults, data]):
            return load_config()

    def test_default_configuration(self):
        config = load_config()
        self.assertEqual((config.coarse.iters, config.final.iters), (800, 1500))
        self.assertEqual(config.seed, 42)

    def test_invalid_numeric_settings_are_rejected(self):
        cases = (
            (("seed",), 2**32),
            (("seed",), -1),
            (("coarse", "position_lr_init"), -0.001),
            (("final", "position_lr_final"), 0),
            (("final", "opacity_lr"), -0.01),
            (("coarse", "density_start_iter"), 500),
            (("ergo", "gaussian_kernel_size"), 4),
            (("ergo", "gaussian_kernel_size"), 0),
            (("ergo", "gaussian_sigma"), 0),
            (("ergo", "lambda_rgb"), -1),
            (("final", "sds_min_timestep_ratio"), 2),
            (("final", "final_prune_min_opacity"), 1),
            (("final", "max_gaussian_anisotropy"), 0.5),
            (("final", "max_gaussian_scale"), -1),
            (("coarse", "iters"), True),
            (("ergo", "eps"), float("nan")),
        )
        for path, value in cases:
            with self.subTest(path=path, value=value):
                data = copy.deepcopy(self.defaults)
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_unknown_fields_are_rejected(self):
        data = copy.deepcopy(self.defaults)
        data["unused"] = 1
        with self.assertRaises(ValueError):
            self.load(data)

    def test_missing_fields_are_rejected(self):
        data = copy.deepcopy(self.defaults)
        del data["ergo"]["gaussian_sigma"]
        with self.assertRaises(ValueError):
            self.load(data)

    def test_valid_boundary_and_disabled_constraints(self):
        data = copy.deepcopy(self.defaults)
        data["seed"] = 2**32 - 1
        data["final"]["opacity_lr"] = 0
        data["final"]["max_gaussian_scale"] = 0
        data["final"]["max_gaussian_anisotropy"] = 0
        config = self.load(data)
        self.assertEqual(config.seed, 2**32 - 1)


if __name__ == "__main__":
    unittest.main()
