from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Dict, List, NamedTuple, Tuple
import asyncio

import os
import uuid
from astropy.io import fits
import numpy as np
from scipy.optimize import minimize

from academy.agent import Agent
from academy.agent import action
from academy.handle import Handle
from academy.exchange import ExchangeFactory

from parsl.app.app import join_app

class AlignmentParameters(NamedTuple):
    h_shift: float
    v_shift: float
    scale: float
    std: float

    def __sub__(self, other: AlignmentParameters):
        return AlignmentParameters(
            self.h_shift - other.h_shift,
            self.v_shift - other.v_shift,
            self.scale / other.scale,
            max(self.std, other.std),
        )

class AlignmentAgent(Agent):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger("Alignment")
        
        # Index corresponds to time
        # None means we haven't seen the timestep before
        self.params_by_time: defaultdict[int, AlignmentParameters | None] = defaultdict()
        self.params_by_time[0] = AlignmentParameters(0, 0, 1, 0)

    def initial_guess(
        self, 
        target_timestep: int,                 
        reference_timestep: int,
    ) -> Tuple[bool, AlignmentParameters]:
        
        if target_timestep in self.params_by_time and reference_timestep in self.params_by_time:
            return False, self.params_by_time[target_timestep] - self.params_by_time[reference_timestep]
        
        if reference_timestep in self.params_by_time:
            while target_timestep > reference_timestep:
                target_timestep -= 1
                if target_timestep in self.params_by_time:
                    return True, self.params_by_time[target_timestep] - self.params_by_time[reference_timestep]
        
        return True, AlignmentParameters(0, 0, 1, float('inf'))
    
    def shift_and_scale(self, image, v_shift, h_shift, scale=1.0, order=1):
        shifted_image = np.ndimage.shift(
            image,
            (v_shift,h_shift),
            order=order,
            mode='constant',
            cval=0.0
        )
        
        return shifted_image * scale
    
    def subtract_ims(self, v_shift, h_shift, scale, refdata, data, margin, cutoff):
        shifted_data = self.shift_and_scale(data, v_shift, h_shift, scale)
        
        subtracted = refdata[margin:-margin, margin:-margin] - shifted_data[margin:-margin, margin:-margin]
        extreme_mask = (subtracted < (-1*cutoff)) | (subtracted > cutoff)
        subtracted[extreme_mask] = np.nan

        return subtracted
    
    def optimize_cmray_params(
        self, 
        cut_data, 
        cut_data_,
        margin,
        cutoff,
        x0 = np.array([0,0,1]),
        bounds = [(-2, 2), (-2, 2), (0.5, 1.5)]
    ):
        """Optimize the alignment between two patches."""
        def optimize_function(params):
            v_shift, h_shift, scale = params
            result = self.subtract_ims(v_shift, h_shift, scale, cut_data, cut_data_, margin, cutoff)
            return np.nanstd(result)
        
        result = minimize(
            optimize_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-6}
        )

        best_v, best_h, best_s = result.x
        best_std = result.fun

        return best_v, best_h, best_s, best_std
    
    def cmray_mask_minimize(
        self,
        reference_file,
        target_file, 
        area=300, 
        x0=np.array([0, 0, 1]), 
        bounds=[(-2, 2), (-2, 2), (0.5, 1.5)],
    ):
        
        
        margin = int(np.max([np.abs(bounds[0][1]), np.abs(bounds[1][1])])+1)

        with fits.open(reference_file) as refdata:
            data = refdata[0].data
            std_cutoff = np.std(np.percentile(data.flatten(), [1, 60]))

            # make random_x and random_y in the center 50% of the image
            random_x = np.random.randint(data.shape[1]*0.25, data.shape[1]*0.75-area-2*margin)
            random_y = np.random.randint(data.shape[0]*0.25, data.shape[0]*0.75-area-2*margin)

            cut_data = data[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]
            cutoff = 2*np.nanstd(cut_data)

        with fits.open(target_file) as sfile:
            data_ = sfile[0].data

            cut_data_ = data_[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

        best_v, best_h, best_s, best_std = self.optimize_cmray_params(
            cut_data,
            cut_data_,
            margin,
            cutoff,
            x0=x0,
            bounds=bounds
        )

        return best_v, best_h, best_s, best_std, std_cutoff
    
    def optimize_parameters(
        self, 
        reference_file: str, 
        target_file: str,
        initial_guess: AlignmentParameters,
        max_iter: int = 20
    ) -> Dict:
        """
        Run optimization until std_value is lower than std_cutoff
        """
        # Initialize the best parameters
        best_params = initial_guess

        success = False
        for iter in range(max_iter):
            bounds = [
                (-2, 2),
                (-2, 2),
                (0.5, 1.5)
            ]

            # Call the minimize method
            best_v, best_h, best_s, std_value, std_cutoff = self.cmray_mask_minimize(
                reference_file,
                target_file,
                area=300,
                x0=np.array([best_params[0].v_shift, best_params[0].h_shift, best_params[0].scale]),
                bounds=bounds
            )

            # Store result
            result_key = f"{reference_file}_{target_file}_{iter}"
            self.optimization_results[result_key] = {
                "v_shift": best_v,
                "h_shift": best_h,
                "scale": best_s,
                "std_value": std_value,
                "std_cutoff": std_cutoff
            }

            # Update best parameters if this result is better
            if std_value < best_params:
                best_params =  AlignmentParameters(
                    best_h,
                    best_v,
                    best_s,
                    std_value,
                )

            # Success condition: std is lower than cutoff
            if std_value < std_cutoff:
                success = True
                break
        
        return {
            "success": success,
            "params": best_params,
            "iterations": max_iter
        }

    @action
    async def calculate_alignment(
        self, 
        reference_timestep: int,
        reference_file: str,
        target_timestep: int,
        target_file: str,
    ) -> AlignmentParameters:
        _id = uuid.uuid1()
        self.logger.info(f"START alignment-exec {_id}")
        if reference_timestep > target_timestep:
            result, _ = await self.calculate_alignment(
                target_timestep,
                target_file,
                reference_timestep,
                reference_file,
            )
            return AlignmentParameters(0, 0, 1) - result
        
        optimize, initial_guess = self.initial_guess(target_timestep, reference_timestep)
        if not optimize:
            return initial_guess
        
        result = self.optimize_parameters(reference_file, target_file, initial_guess)
        if reference_timestep in self.params_by_time:
            self.params_by_time[target_timestep] = self.params_by_time[reference_timestep] + result["params"]
        self.logger.info(f"END alignment-exec {_id}")
        return result["params"]
        

async def calculate_alignment(
    agent: Handle[AlignmentAgent],
    factory: ExchangeFactory[Any],
    reference_timestep: int,
    reference_file: str,
    target_timestep: int,
    target_file: str,
) -> AlignmentParameters:
    async with await factory.create_user_client():
        return await agent.calculate_alignment(
            reference_timestep,
            reference_file,
            target_timestep,
            target_file,
        )

@join_app
def calculate_alignment_app(
    agent: Handle[AlignmentAgent],
    factory: ExchangeFactory[Any],
    reference_timestep: int,
    reference_file: str,
    target_timestep: int,
    target_file: str,
) -> AlignmentParameters:
    loop = asyncio.get_event_loop()
    future = asyncio.run_coroutine_threadsafe(
        calculate_alignment(
            agent,
            factory,
            reference_timestep,
            reference_file,
            target_timestep,
            target_file,
        ),
        loop,
    )
    return future