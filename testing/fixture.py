
import os
from collections.abc import Generator
import pytest
import parsl
from parsl.config import Config
from parsl import DataFlowKernel

from ifum_agent.parsl import get_parsl_config

@pytest.fixture(autouse=True, scope='session')
def local_parsl(tmp_path_factory) -> Generator[DataFlowKernel]:
    run_dir = tmp_path_factory.mktemp("parsl")
    config = get_parsl_config("htex-local", run_dir, 8)

    with parsl.load(config) as dfk:
        yield dfk

@pytest.fixture
def data_dir() -> str:
    current_path = os.path.abspath(__file__)
    testing_dir = os.path.dirname(current_path)
    return os.path.join(testing_dir, "data")