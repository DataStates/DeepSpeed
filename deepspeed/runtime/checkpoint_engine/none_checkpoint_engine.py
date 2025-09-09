# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.utils import log_dist, logger
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
import os, time

class NoneCheckpointEngine(CheckpointEngine):

    def __init__(self):
        super().__init__()
        return

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def create(self, tag):
        return None

    def save(self, state_dict, path: str):
        return None

    def load(self, path: str, map_location=None):
        return None

    def commit(self, tag):
        return None

    def wait(self):
        return None
