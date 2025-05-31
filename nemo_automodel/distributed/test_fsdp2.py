import pytest
from unittest import mock
from nemo_automodel.distributed.fsdp2 import FSDP2Manager

class DummyModel:
    pass

def setup_dist_available_initialized():
    patcher1 = mock.patch('torch.distributed.is_available', return_value=True)
    patcher2 = mock.patch('torch.distributed.is_initialized', return_value=True)
    patcher1.start()
    patcher2.start()
    return [patcher1, patcher2]

def teardown_patchers(patchers):
    for p in patchers:
        p.stop()

def test_fsdp2manager_init_and_device_mesh():
    patchers = setup_dist_available_initialized()
    with mock.patch('torch.distributed.device_mesh.init_device_mesh') as mock_mesh:
        mock_mesh.return_value = mock.MagicMock()
        mgr = FSDP2Manager(dp_size=2, tp_size=1, cp_size=1, world_size=2)
        assert mgr.dp_size == 2
        assert mgr.tp_size == 1
        assert mgr.cp_size == 1
        assert hasattr(mgr, 'device_mesh')
    teardown_patchers(patchers)

def test_fsdp2manager_parallelize_nvfsdp_false():
    patchers = setup_dist_available_initialized()
    with mock.patch('torch.distributed.device_mesh.init_device_mesh') as mock_mesh, \
         mock.patch('nemo_automodel.distributed.parallelizer.fsdp2_strategy_parallelize') as mock_fsdp2:
        mock_mesh.return_value = mock.MagicMock()
        mgr = FSDP2Manager(dp_size=2, tp_size=1, cp_size=1, world_size=2, nvfsdp=False)
        model = DummyModel()
        mgr.device_mesh = mock.MagicMock()
        result = mgr.parallelize(model)
        mock_fsdp2.assert_called_once()
    teardown_patchers(patchers)

def test_fsdp2manager_parallelize_nvfsdp_true():
    patchers = setup_dist_available_initialized()
    with mock.patch('torch.distributed.device_mesh.init_device_mesh') as mock_mesh, \
         mock.patch('nemo_automodel.distributed.parallelizer.nvfsdp_strategy_parallelize') as mock_nvfsdp:
        mock_mesh.return_value = mock.MagicMock()
        mgr = FSDP2Manager(dp_size=2, tp_size=1, cp_size=1, world_size=2, nvfsdp=True, nvfsdp_unit_modules=['torch.nn.Linear'])
        model = DummyModel()
        mgr.device_mesh = mock.MagicMock()
        mock_nvfsdp.return_value = 'wrapped_model'
        result = mgr.parallelize(model)
        mock_nvfsdp.assert_called_once()
        assert result == 'wrapped_model'
    teardown_patchers(patchers)

def test_fsdp2manager_invalid_dist():
    with mock.patch('torch.distributed.is_available', return_value=False):
        with pytest.raises(RuntimeError):
            FSDP2Manager(dp_size=2, tp_size=1, cp_size=1, world_size=2)
    with mock.patch('torch.distributed.is_available', return_value=True), \
         mock.patch('torch.distributed.is_initialized', return_value=False):
        with pytest.raises(RuntimeError):
            FSDP2Manager(dp_size=2, tp_size=1, cp_size=1, world_size=2)

def test_fsdp2manager_invalid_group_sizes():
    patchers = setup_dist_available_initialized()
    with mock.patch('torch.distributed.device_mesh.init_device_mesh') as mock_mesh:
        mock_mesh.return_value = mock.MagicMock()
        with pytest.raises(AssertionError):
            FSDP2Manager(dp_size=0, tp_size=1, cp_size=1, world_size=2)
        with pytest.raises(AssertionError):
            FSDP2Manager(dp_size=2, tp_size=0, cp_size=1, world_size=2)
        with pytest.raises(AssertionError):
            FSDP2Manager(dp_size=2, tp_size=1, cp_size=0, world_size=2)
    teardown_patchers(patchers) 