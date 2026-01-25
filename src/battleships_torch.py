from enum import Enum, auto
from typing import NamedTuple, Tuple
import random
import numpy as np
import torch

class StepOut(NamedTuple):
    active: torch.Tensor
    hit: torch.Tensor
    won: torch.Tensor
    all_finished: bool

class BattleshipsBoardCollection:
    ''' Batchable battleships board built with torch.Tensor. '''

    def __init__(self, batch_size=100, size=10, ships=(5, 4, 3, 2, 2), device='cpu'):
        self.B: int = batch_size
        self.N: int = size
        self.ships_spec = tuple(ships)
        self.device = torch.device(device)
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        B, N = self.B, self.N
        self.ships = torch.zeros((B, N, N), dtype=torch.bool, device=self.device)
        self.attacked = torch.zeros((B, N, N), dtype=torch.bool, device=self.device)
        self.remaining = torch.full((B,), sum(self.ships_spec), dtype=torch.int64, device=self.device)
        self.done = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self.move_count = torch.zeros((B,), dtype=torch.int64, device=self.device)
        self._randomly_place_ships()

    @torch.no_grad()
    def _randomly_place_ships(self, retries: int = 10_000) -> None:
        N = self.N
        for b in range(self.B):
            for L in self.ships_spec:
                placed = False
                for _ in range(retries):
                    horiz = random.getrandbits(1) == 1
                    if horiz:
                        y = random.randrange(N)
                        x = random.randrange(N - L + 1)
                        if not self.ships[b, y, x:x+L].any():
                            self.ships[b, y, x:x+L] = True
                            placed = True
                            break
                    else:
                        y = random.randrange(N - L + 1)
                        x = random.randrange(N)
                        if not self.ships[b, y:y+L, x].any():
                            self.ships[b, y:y+L, x] = True
                            placed = True
                            break
                if not placed:
                    raise RuntimeError(f"Failed to place ship length {L} on board {b}.")

    @torch.no_grad()
    def mask(self) -> torch.Tensor:
        mask = ~self.attacked # [B,N,N]
        if self.done.any():
            mask = mask.clone()
            mask[self.done] = True
        return mask.view(self.B, -1)

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        hits = self.ships & self.attacked # [B,N,N]
        misses = self.attacked ^ hits # [B,N,N]
        return torch.stack([hits.float(), misses.float()], dim=1) # [B,2,N,N]

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> StepOut:
        ''' Carry out a batched step of the game. '''

        # get active board mask
        active_mask = ~self.done

        # select indices of active boards
        active_indices = torch.arange(self.B, device=self.device)[active_mask] # [B,] (0 to B-1)

        # get attack coordinates for active baords
        actions_dim_0 = actions[active_indices] // self.N # y axis
        actions_dim_1 = actions[active_indices] % self.N # x axis

        # apply attacks and see which were hits
        self.attacked[active_indices, actions_dim_0, actions_dim_1] = True
        hit = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        hit[active_indices] = self.ships[active_indices, actions_dim_0, actions_dim_1]     

        # update tracking tensors
        self.remaining[active_indices] -= hit[active_indices].to(self.remaining.dtype)
        self.done = self.done | (self.remaining == 0)
        self.move_count[active_indices] += 1

        return StepOut(active=active_mask, hit=hit, won=self.done.clone(), all_finished=self.done.all())

    @torch.no_grad()
    def is_finished(self) -> torch.Tensor:
        return self.done.clone()

    @torch.no_grad()
    def get_total_moves(self) -> torch.Tensor:
        return self.move_count.clone()

    @torch.no_grad()
    def reveal_ships(self) -> torch.Tensor:
        return self.ship.clone()



