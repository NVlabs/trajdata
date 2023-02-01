import unittest
from collections import defaultdict

import numpy as np
import torch

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import NP_STATE_TYPES, TORCH_STATE_TYPES
from trajdata.dataset import UnifiedDataset

AgentStateArray = NP_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,h"]
AgentObsArray = NP_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,s,c"]
AgentStateTensor = TORCH_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,h"]
AgentObsTensor = TORCH_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,s,c"]


class TestStateTensor(unittest.TestCase):
    def test_construction(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = torch.rand(8).as_subclass(AgentStateTensor)
        c = AgentObsTensor(torch.rand(5, 9))

    def test_class_propagation(self):
        # TODO(bivanovic): We want to test the following commented code, but...
        # https://github.com/pytorch/pytorch/issues/47051
        # a = AgentStateTensor(torch.rand(2, 8))
        # self.assertTrue(isinstance(a.to("cpu"), AgentStateTensor))

        a = AgentStateTensor(torch.rand(2, 8))
        self.assertTrue(isinstance(a.cpu(), AgentStateTensor))

        b = AgentStateTensor(torch.rand(2, 8))
        self.assertTrue(isinstance(a + b, AgentStateTensor))

        b = torch.rand(2, 8)
        self.assertTrue(isinstance(a + b, AgentStateTensor))

        a += 1
        self.assertTrue(isinstance(a, AgentStateTensor))

    def test_property_access(self):
        a = AgentStateTensor(torch.rand(2, 8))
        position = a[..., :3]
        velocity = a[..., 3:5]
        acc = a[..., 5:7]
        h = a[..., 7:]

        self.assertTrue(torch.allclose(a.position3d, position))
        self.assertTrue(torch.allclose(a.velocity, velocity))
        self.assertTrue(torch.allclose(a.acceleration, acc))
        self.assertTrue(torch.allclose(a.heading, h))

    def test_heading_conversion(self):
        a = AgentStateTensor(torch.rand(2, 8))
        h = a[..., 7:]
        hv = a.heading_vector
        self.assertTrue(torch.allclose(torch.atan2(hv[..., 1], hv[..., 0])[:, None], h))

    def test_as_format(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a.as_format("x,y,z,xd,yd,xdd,ydd,s,c")
        self.assertTrue(isinstance(b, AgentObsTensor))
        self.assertTrue(torch.allclose(a, b.as_format(a._format)))

    def test_tensor_ops(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a[0] + a[1]
        c = torch.mean(b)
        self.assertFalse(isinstance(c, AgentStateTensor))
        self.assertTrue(isinstance(c, torch.Tensor))


class TestStateArray(unittest.TestCase):
    def test_construction(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        c = np.random.rand(5, 9).view(AgentObsArray)

    def test_property_access(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        position = a[..., :3]
        velocity = a[..., 3:5]
        acc = a[..., 5:7]
        h = a[..., 7:]

        self.assertTrue(np.allclose(a.position3d, position))
        self.assertTrue(np.allclose(a.velocity, velocity))
        self.assertTrue(np.allclose(a.acceleration, acc))
        self.assertTrue(np.allclose(a.heading, h))

    def test_property_setting(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        a.heading = 0.0
        self.assertTrue(np.allclose(a[..., -1], np.zeros([2, 1])))

    def test_heading_conversion(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        h = a[..., 7:]
        hv = a.heading_vector
        self.assertTrue(np.allclose(np.arctan2(hv[..., 1], hv[..., 0])[:, None], h))

    def test_as_format(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        b = a.as_format("x,y,z,xd,yd,xdd,ydd,s,c")
        self.assertTrue(isinstance(b, AgentObsArray))
        self.assertTrue(np.allclose(a, b.as_format(a._format)))

    def test_tensor_ops(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        b = a[0] + a[1]
        c = np.mean(b)
        self.assertFalse(isinstance(c, AgentStateArray))
        self.assertTrue(isinstance(c, float))


class TestDataset(unittest.TestCase):
    def test_default_datatypes_agent(self):
        dataset = UnifiedDataset(
            desired_data=["lyft_sample-mini_val"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=36,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "lyft_sample": "~/datasets/lyft_sample/scenes/sample.zarr",
            },
        )

        elem: AgentBatchElement = dataset[0]
        self.assertIsInstance(elem.curr_agent_state_np, dataset.np_state_type)
        self.assertIsInstance(elem.agent_history_np, dataset.np_obs_type)
        self.assertIsInstance(elem.agent_future_np, dataset.np_obs_type)
        self.assertIsInstance(elem.robot_future_np, dataset.np_obs_type)

    def test_default_datatypes_agent(self):
        dataset = UnifiedDataset(
            desired_data=["lyft_sample-mini_val"],
            centric="scene",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=36,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "lyft_sample": "~/datasets/lyft_sample/scenes/sample.zarr",
            },
        )

        elem: SceneBatchElement = dataset[0]
        self.assertIsInstance(elem.centered_agent_state_np, dataset.np_state_type)
        self.assertIsInstance(elem.agent_histories[0], dataset.np_obs_type)
        self.assertIsInstance(elem.agent_futures[0], dataset.np_obs_type)
        self.assertIsInstance(elem.robot_future_np, dataset.np_obs_type)


if __name__ == "__main__":
    unittest.main()
