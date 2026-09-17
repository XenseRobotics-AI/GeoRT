# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np 
from pathlib import Path
from geort.utils.path import get_human_data 


class ReplayMocap:
    def __init__(self, human_data):
        explicit = Path(human_data).expanduser()
        human_data_path = explicit if explicit.is_file() else get_human_data(human_data)
        if human_data_path is None:
            raise FileNotFoundError(f'Human recording not found: {human_data}')
        self.data_path = Path(human_data_path).resolve()
        self.human_points = np.load(human_data_path, allow_pickle=False) # [T, N, 3]
        self.t = 0
        self.T = len(self.human_points)
        return 

    def get(self):
        result = self.human_points[self.t]
        self.t = (self.t + 1) % self.T 
        if self.t == 0:
            print("[ReplayMocap] I am returning to start!")
        return {"result": result, "status": "recording"}
