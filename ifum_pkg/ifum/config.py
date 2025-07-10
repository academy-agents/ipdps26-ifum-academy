# https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html#midway-rcc-uchicago

from parsl.addresses import address_by_interface, address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider, LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.monitoring.monitoring import MonitoringHub



def local_config():
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                provider=LocalProvider(min_blocks=1,
                                    max_blocks=1,
                                    parallelism=1),
            )
        ],
        monitoring=MonitoringHub(
        hub_address=address_by_hostname(),
        monitoring_debug=False,
        resource_monitoring_interval=10,
        )
    )

    return config



# def midway_config():
#     config = Config(
#         executors=[
#             HighThroughputExecutor(
#                 label='Midway_HTEX_multinode',
#                 address=address_by_interface('bond0'),
#                 worker_debug=False,
#                 max_workers_per_node=2,
#                 provider=SlurmProvider(
#                     'caslake',  # Partition name, e.g 'broadwl'
#                     launcher=SrunLauncher(),
#                     nodes_per_block=2,
#                     init_blocks=1,
#                     min_blocks=1,
#                     max_blocks=1,
#                     # string to prepend to #SBATCH blocks in the submit
#                     # script to the scheduler eg: '#SBATCH --constraint=knl,quad,cache'
#                     scheduler_options='',
#                     # Command to be run before starting a worker, such as:
#                     # 'module load Anaconda; source activate parsl_env'.
#                     worker_init='conda activate parsl_py38',
#                     walltime='00:30:00'
#                 ),
#             )
#         ],
#         usage_tracking=LEVEL_1,
#     )

#     return config



def midway_config():
    config = Config(
        executors=[
            HighThroughputExecutor(
                label='Midway_HTEX_multinode',
                address=address_by_interface('bond0'),
                worker_debug=False,
                max_workers_per_node=2,
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                    parallelism=1,
                    launcher=SrunLauncher(),
                    worker_init='module load Python; conda activate /home/babnigg/conda_envs/parsl_py38'
                    ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    return config