import torch

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [(min(1.0, self._step_count / self.warmup_steps)) * base_lr for base_lr in self.base_lrs]
        else:
            # Exponential decay phase
            decay_factor = (self.min_lr / self.base_lrs[0]) ** ((self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            return [decay_factor * base_lr for base_lr in self.base_lrs]


# import torch
# import math
# from torch.optim.lr_scheduler import _LRScheduler

# class WarmupLR(_LRScheduler):
#     """
#     Implements a learning rate scheduler with warmup followed by exponential decay to minimum LR

#     Args:
#         optimizer (torch.optim.Optimizer): Optimizer instance
#         warmup_steps (int): Number of warmup steps
#         total_steps (int): Total number of training steps
#         min_lr (float): Minimum learning rate to decay to
#         last_epoch (int): Index of last epoch
#         verbose (bool): Print learning rate updates
#     """
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         warmup_steps: int,
#         total_steps: int,
#         min_lr: float,
#         last_epoch=-1,
#         verbose=False,
#     ):
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.min_lr = min_lr
#         super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

#     def get_lr(self):
#         """Calculate learning rate for current step"""
#         if self._step_count < self.warmup_steps:
#             # Linear warmup phase
#             warmup_factor = min(1.0, self._step_count / self.warmup_steps)
#             return [warmup_factor * base_lr for base_lr in self.base_lrs]
#         else:
#             # Exponential decay phase
#             # Calculate decay factor to reach min_lr by the end of training
#             remaining_steps = self.total_steps - self.warmup_steps
#             current_step = self._step_count - self.warmup_steps

#             if remaining_steps <= 0:
#                 # Avoid division by zero
#                 decay_factor = self.min_lr / self.base_lrs[0]
#             else:
#                 # Exponential decay formula
#                 decay_factor = math.pow(
#                     self.min_lr / self.base_lrs[0],
#                     current_step / remaining_steps
#                 )

#             return [decay_factor * base_lr for base_lr in self.base_lrs]
