import os
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint


class PEFTAdapterCheckpoint(ModelCheckpoint):
    """
    Save only the PEFT adapter instead of the whole .ckpt file for checkpointing.
    """

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )

    def _is_peft_module(self, module: torch.nn.Module) -> bool:
        if hasattr(module, "peft_config"):
            return True
        return False

    def _find_peft_module(self, module: torch.nn.Module) -> Optional[torch.nn.Module]:
        """
        Recursively search for the PEFT adapter module in the model.
        """
        if self._is_peft_module(module):
            return module

        for child in module.children():
            peft_module = self._find_peft_module(child)
            if peft_module is not None:
                return peft_module

        return None

    def _save_checkpoint(self, trainer, filepath) -> None:
        """
        Override the save checkpoint method to save PEFT adapter weights instead
        of the full model checkpoint.
        """
        peft_model = self._find_peft_module(trainer.lightning_module)

        if peft_model is None:
            raise ValueError(
                "Could not find PEFT adapter module in provided model.  "
                "Please make sure the model has a PEFT adapter module."
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        adapter_path = str(Path(filepath).with_suffix(""))
        peft_model.save_pretrained(adapter_path)

        if self.verbose:
            print(f"Saved PEFT adapter to {adapter_path}")

        self._last_checkpoint = adapter_path

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Don't load any checkpoint data since we're only saving adapters
        """
        raise NotImplementedError("Loading PEFT adapter checkpoints is not supported.")
