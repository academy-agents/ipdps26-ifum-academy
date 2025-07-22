# https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html#midway-rcc-uchicago

from parsl.addresses import address_by_interface, address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider, LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.monitoring.monitoring import MonitoringHub

import numpy as np

def prepare_inputs(bad_blues,bad_reds,wcs_stars,mode,wavelength):
    bad_masks = [np.array(bad_blues)-1,np.array(bad_reds)-1]
    wcs_stars = np.array(wcs_stars)
    if mode == "STD":
        total_masks = 552
        mask_groups = 12
        hex_dims = (23,24)
    elif mode == "HR":
        total_masks = 864
        mask_groups = 16
        hex_dims = (27,32)
    else:
        print("invalid mode")

    if wavelength == "far red":
        bins = np.arange(7000,10000,1)
    elif wavelength == "blue":
        bins = np.arange(4000,6100,1)
    else:
        print("invalid wavelength")

    return bad_masks,wcs_stars,total_masks,mask_groups,hex_dims,bins 


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

def midway_config():
    config = Config(
        run_dir="/home/babnigg/globus/IFU-M/runinfo",
        executors=[
            HighThroughputExecutor(
                label='Midway_HTEX_multinode',
                address=address_by_hostname(),
                worker_debug=False,
                # max_workers_per_node=2,
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                    parallelism=1,
                    launcher=SrunLauncher(),
                    worker_init='source $(conda info --base)/etc/profile.d/conda.sh; conda activate /home/babnigg/conda_envs/ifum_parsl'
                    ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    return config