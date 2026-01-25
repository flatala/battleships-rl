from enum import Enum, auto
from typing import NamedTuple, Tuple
import numpy as np
import torch

class StepOut(NamedTuple):
    hit: torch.Tensor
    finished: torch.Tensor

class BattleshipsTorchBoard:
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
                    horiz = bool(torch.randint(0, 2, (), device=self.device).item())

                    if horiz:
                        y = int(torch.randint(0, N, (), device=self.device).item())
                        x = int(torch.randint(0, N - L + 1, (), device=self.device).item())
                        if not self.ships[b, y, x : x + L].any():
                            self.ships[b, y, x : x + L] = True
                            placed = True
                            break
                    else:
                        y = int(torch.randint(0, N - L + 1, (), device=self.device).item())
                        x = int(torch.randint(0, N, (), device=self.device).item())
                        if not self.ships[b, y : y + L, x].any():
                            self.ships[b, y : y + L, x] = True
                            placed = True
                            break
                if not placed:
                    raise RuntimeError(f"Failed to place ship length {L} on board {b}.")

    @torch.no_grad()
    def mask(self) -> torch.Tensor:
        m = ~self.attacked # [B,N,N]
        if self.done.any():
            # broadcasts [B,N,N] & [B,1,1] -> [B,N,N]
            m = m & (~self.done).view(B, 1, 1)
        return m.view(self.B, -1) # [B,N*N]

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        hits = self.ships & self.attacked # [B,N,N]
        misses = self.attacked - hits # [B,N,N]
        return torch.stack([hits.float(), misses.float()], dim=1) # [B,2,N,N]

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> StepOut:
        b = torch.arange(self.B, device=self.device) # [B,] (0 to B-1)
        actions_dim_0 = actions // self.N # y axis
        actions_dim_1 = actions % self.N # x axis
        self.attacked[b, actions_dim_0, actions_dim_1] = True
        hit = self.ship[b, actions_dim_0, actions_dim_1]         
        self.remaining[hit] -= 1
        self.done = self.remaining == 0
        self.move_count += 1
        return StepOut(hit=hit, done=self.done.clone())

    @torch.no_grad()
    def is_finished(self) -> torch.Tensor:
        return self.done.clone()

    @torch.no_grad()
    def get_total_moves(self) -> torch.Tensor:
        return self.move_count.clone()

    @torch.no_grad()
    def reveal_ships(self) -> torch.Tensor:
        return self.ship.clone()



