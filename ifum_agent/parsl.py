from __future__ import annotations

import os
import pickle

from parsl.addresses import address_by_hostname
from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher
from parsl.usage_tracking.levels import LEVEL_1
from parsl.monitoring.monitoring import MonitoringHub
from parsl.data_provider.files import File
from parsl.dataflow.memoization import id_for_memo
from parsl.dataflow.dependency_resolvers import DEEP_DEPENDENCY_RESOLVER
from parsl.utils import get_all_checkpoints

def get_htex_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label='htex-local',
        max_workers_per_node=workers_per_node,
        address=address_by_hostname(),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, max_blocks=1),
    )
    return Config(
        executors=[executor],
        monitoring=MonitoringHub(
            workflow_name="IFU-M",
            resource_monitoring_enabled=False,
            logging_endpoint=f"sqlite:///{os.path.join(run_dir, 'monitoring.db')}"
        ),
        run_dir=run_dir,
        initialize_logging=False,
        dependency_resolver=DEEP_DEPENDENCY_RESOLVER,
        exit_mode="skip",
        checkpoint_mode = 'task_exit',
        checkpoint_files = get_all_checkpoints(run_dir),
    )


def get_htex_aurora_cpu_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    # Get the number of nodes:
    node_file = os.getenv('PBS_NODEFILE')
    assert node_file is not None, 'PBS_NODEFILE must be set'
    with open(node_file, 'r') as f:
        node_list = f.readlines()
        num_nodes = len(node_list)

    executor = HighThroughputExecutor(
        max_workers_per_node=workers_per_node,
        # Increase if you have many more tasks than workers
        prefetch_capacity=0,
        # Options that specify properties of PBS Jobs
        provider=LocalProvider(
            # Number of nodes job
            nodes_per_block=num_nodes,
            launcher=MpiExecLauncher(
                bind_cmd='--cpu-bind',
                overrides='--ppn 1',
            ),
            init_blocks=1,
            max_blocks=1,
        ),
    )

    return Config(
        executors=[executor],
        monitoring=MonitoringHub(
            workflow_name="IFU-M",
            resource_monitoring_enabled=False,
            logging_endpoint=f"sqlite:///{os.path.join(run_dir, 'monitoring.db')}"
        ),
        run_dir=run_dir,
        initialize_logging=False,
        dependency_resolver=DEEP_DEPENDENCY_RESOLVER,
    )


def get_htex_aurora_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label='htex-local',
        max_workers_per_node=workers_per_node,
        address=address_by_interface('hsn0'),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, max_blocks=1),
    )
    return Config(
        executors=[executor],
        monitoring=MonitoringHub(
            workflow_name="IFU-M",
            resource_monitoring_enabled=False,
            logging_endpoint=f"sqlite:///{os.path.join(run_dir, 'monitoring.db')}"
        ),
        run_dir=run_dir,
        initialize_logging=False,
        dependency_resolver=DEEP_DEPENDENCY_RESOLVER,
    )

def get_midway_cpu_config(
    run_dir: str,
    workers_per_node: int
) -> Config:
    # Get the number of nodes:
    num_nodes = os.getenv('SLURM_NNODES')
    assert num_nodes is not None, 'SLURM_NNODES must be set'

    config = Config(
        run_dir=run_dir,
        executors=[
            HighThroughputExecutor(
                label='Midway_HTEX_multinode',
                address=address_by_hostname(),
                worker_debug=True,
                max_workers_per_node=workers_per_node,
                provider=LocalProvider(
                    min_blocks=1,
                    # max_blocks=4,
                    parallelism=1,
                    nodes_per_block=num_nodes, # match requested nodes
                    launcher=SrunLauncher(),
                    worker_init='source $(conda info --base)/etc/profile.d/conda.sh; conda activate /home/babnigg/conda_envs/ifum_parsl'
                    ),
            )
        ],
        monitoring=MonitoringHub(
            workflow_name="IFU-M",
            resource_monitoring_enabled=False,
            logging_endpoint=f"sqlite:///{os.path.join(run_dir, 'monitoring.db')}"
        ),
        usage_tracking=LEVEL_1,
        dependency_resolver=DEEP_DEPENDENCY_RESOLVER,
    )

    return config

PARSL_CONFIGS = {
    'htex-local': get_htex_local_config,
    'htex-aurora-cpu': get_htex_aurora_cpu_config,
    'htex-aurora-local': get_htex_aurora_local_config,
    'htex-midway-cpu': get_midway_cpu_config,
}

def get_parsl_config(name: str, run_dir: str, workers_per_node: int):
    return PARSL_CONFIGS[name](
        os.path.join(run_dir, 'parsl'),
        workers_per_node=workers_per_node,
    )


@id_for_memo.register(File)
def id_for_memo_serialize(obj, output_ref=False):
    return pickle.dumps(obj)