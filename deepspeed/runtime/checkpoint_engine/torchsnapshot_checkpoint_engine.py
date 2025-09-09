# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from torchsnapshot import Snapshot, StateDict
import os, time

class TorchSnapshotCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.snapshot = []

    def create(self, tag):
        log_dist(f"[TorchSnapshot] Checkpoint {tag} is about to be saved!", ranks=[0])

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def save(self, state_dict, path: str):
        t = time.time()
        # debloated_state_dict = clone_tensors_for_torch_save(state_dict)
        a = {"app_state": StateDict(ckpt=state_dict)}
        snapshot = Snapshot.async_take(path=path, app_state=a, replicated=[])
        self.snapshot.append(snapshot)
        logger.info(f"[TorchSnapshot] CkptTime {path} {time.time()-t}.")
        return None

    def load(self, path: str, map_location=None):
        snapshot = Snapshot(path=path)
        logger.info(f"loading checkpoint {path}")
        a = StateDict(ckpt={})
        snapshot.restore(app_state=a)
        res = a["app_state"].data['ckpt']
        logger.info('Restored state dict keys: {}'.format(res.keys()))
        return res

    def commit(self, tag):
        logger.info(f"[TorchSnapshot] Checkpoint {tag} is ready now!")
        return True

    def wait(self):
        t = time.time()
        snapshots = []
        for s in self.snapshot:
            x = s.wait()
            assert s.done() == True, "Snapshot should be done after wait"
            snapshots.append(x) # Append the snapshot to keep temporary reference
        self.snapshot = []
        logger.info(f"[TorchSnapshot] Waited for all snapshots to complete in {time.time() - t:.6f}s")
        del snapshots
        return True
    
    def __del__(self):
        self.wait()
        return True