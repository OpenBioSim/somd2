from unittest.mock import MagicMock, patch
from somd2.runner._dynamics_cache import DynamicsCache


def _mock_opencl(vendors):
    """Build a mock pyopencl module exposing one GPU per vendor string."""
    devices = []
    for v in vendors:
        dev = MagicMock()
        dev.vendor = v
        devices.append(dev)

    mock_platform = MagicMock()
    mock_platform.get_devices.return_value = devices

    mock_cl = MagicMock()
    mock_cl.get_platforms.return_value = [mock_platform]
    return mock_cl


def _mock_pynvml(used=1024, free=2048, total=3072):
    """Build a mock pynvml module returning fixed memory values."""
    mock_mem = MagicMock(used=used, free=free, total=total)
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
    return mock_pynvml


def test_in_range_nvidia_device_queries_pynvml():
    """When OpenCL finds the device and identifies NVIDIA, query pynvml by physical index."""
    mock_cl = _mock_opencl(["NVIDIA Corporation", "NVIDIA Corporation"])
    mock_pynvml = _mock_pynvml(1024, 2048, 3072)

    with patch.dict("sys.modules", {"pyopencl": mock_cl, "pynvml": mock_pynvml}):
        result = DynamicsCache._check_device_memory(device=1)

    assert result == (1024, 2048, 3072)
    mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_out_of_range_opencl_index_falls_back_to_pynvml():
    """When OpenCL returns fewer devices than device_index, fall back cleanly to pynvml."""
    mock_cl = _mock_opencl(["NVIDIA Corporation"])
    mock_pynvml = _mock_pynvml(512, 1536, 2048)

    with patch.dict("sys.modules", {"pyopencl": mock_cl, "pynvml": mock_pynvml}):
        result = DynamicsCache._check_device_memory(device="2")

    assert result == (512, 1536, 2048)
    mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_empty_opencl_devices_falls_back_to_pynvml():
    """When OpenCL returns no devices, fall back cleanly to pynvml."""
    mock_cl = _mock_opencl([])
    mock_pynvml = _mock_pynvml(256, 768, 1024)

    with patch.dict("sys.modules", {"pyopencl": mock_cl, "pynvml": mock_pynvml}):
        result = DynamicsCache._check_device_memory(device=0)

    assert result == (256, 768, 1024)
    mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_uuid_device_queries_pynvml_by_uuid():
    """UUID strings skip OpenCL indexing and query pynvml directly by UUID."""
    gpu_uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    mock_cl = _mock_opencl(["NVIDIA Corporation"])
    mock_pynvml = _mock_pynvml(2048, 2048, 4096)

    with patch.dict("sys.modules", {"pyopencl": mock_cl, "pynvml": mock_pynvml}):
        result = DynamicsCache._check_device_memory(device=gpu_uuid)

    assert result == (2048, 2048, 4096)
    mock_pynvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(gpu_uuid.encode())
    mock_pynvml.nvmlShutdown.assert_called_once()
