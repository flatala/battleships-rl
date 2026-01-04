from enum import Enum, auto
from typing import NamedTuple, Tuple
import numpy as np
import torch

class Direction(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()

class AttackResult(NamedTuple):
    hit: bool
    finished: bool

class BattleshipsBoard:
    ''' Class representing a Battleships game board. '''

    def __init__(self, size=10, ships=None):
        ''' Initialize the Battleships board with given size and ship lengths. '''

        # default ship lengths (avoiding mutable default argument)
        ships = [5, 4, 3, 3, 2] if ships is None else ships

        # check for negative size
        if size <= 0:
            raise ValueError("Board size must be a positive integer.")

        # ensure all lengths are defined correctly
        if not all(length > 0 and length <= size for length in ships):
            raise ValueError("All ship lengths must be positive integers and less than or equal to the board size.")  

        # initialize board  
        self.size = size
        self.ships = ships
        self.remaining = sum(ships)
        self.placing_index = 0
        self.board = np.zeros((size, size), dtype=int)
        self.attacks = np.zeros((size, size), dtype=int)

    def is_finished(self) -> bool:
        ''' Check if all ships have been sunk. '''
        return self.remaining == 0

    def get_board(self) -> np.ndarray:
        ''' Get the current state of the board. '''
        return self.board.copy()

    def get_attacks(self) -> np.ndarray:
        ''' Get the current state of attacks on the board. '''
        return self.attacks.copy()

    def get_next_length(self) -> int:
        ''' Get the length of the next ship to be placed. '''

        # check if all ships have been placed
        if self.placing_index >= len(self.ships):
            raise IndexError("All ships have been placed.")

        # get next ship length
        length = self.ships[self.placing_index]
        self.placing_index += 1
        return length

    def get_mask_and_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Get the mask of valid attack positions and the current state tensor. '''
        
        # create mask of valid attack positions
        mask = (self.attacks == 0).astype(np.float32).reshape(-1)

        # create state tensor with two channels: board and attacks
        hits = torch.tensor(self.attacks == 1, dtype=torch.float32)
        misses = torch.tensor(self.attacks == -1, dtype=torch.float32)
        state = torch.stack([hits, misses], dim=0).squeeze(2)

        return torch.tensor(mask, dtype=torch.bool), state

    def is_valid_placement(self, origin_ax_0: int, origin_ax_1: int, length: int, direction: Direction) -> bool:
        ''' Check if a ship can be placed at the given position with the specified length and direction. '''

        # ensure direction is not None
        if direction is None:
            raise ValueError("Direction must be specified.")

        # verify parameter corectness
        out_of_bounds = not (0 <= origin_ax_0 < self.size and 0 <= origin_ax_1 < self.size)
        bad_length = not (1 <= length <= self.size)
        if out_of_bounds or bad_length:
            return False

        # verify placemnt if horizontal
        if direction == Direction.HORIZONTAL:
            if origin_ax_1 + length > self.size:
                return False
            if np.any(self.board[origin_ax_0, origin_ax_1 : origin_ax_1 + length] != 0):
                return False

        # verify placemnt if vertical
        if direction == Direction.VERTICAL:
            if origin_ax_0 + length > self.size:
                return False
            if np.any(self.board[origin_ax_0 : origin_ax_0 + length, origin_ax_1] != 0):
                return False

        return True

    def place_ship(self, origin_ax_0: int, origin_ax_1: int, length: int, direction: Direction) -> np.ndarray:
        ''' Place a ship at the given position with the specified length and direction. '''

        # ensure placement is valid
        if not self.is_valid_placement(origin_ax_0, origin_ax_1, length, direction):
            raise ValueError("Invalid ship placement.")

        # place ship if horizontal
        if direction == Direction.HORIZONTAL:
            self.board[origin_ax_0, origin_ax_1 : origin_ax_1 + length] = 1

        # place ship if vertical
        if direction == Direction.VERTICAL:
            self.board[origin_ax_0 : origin_ax_0 + length, origin_ax_1] = 1

        return self.board

    def receive_attack(self, ax_0: int, ax_1: int) -> AttackResult:
        ''' Process an attack at the given coordinates. '''

        # ensure all ships have been placed
        if self.placing_index != len(self.ships):
            raise ValueError("All ships must be placed before attacks are allowed.")

        # check if attack is within bounds
        if not (0 <= ax_0 < self.size and 0 <= ax_1 < self.size):
            raise ValueError("Attack coordinates are out of bounds.")

        # check if the position has already been attacked
        if self.attacks[ax_0, ax_1] != 0:
            raise ValueError("This position has already been attacked.")

        # hit
        if self.board[ax_0, ax_1] == 1:
            self.attacks[ax_0, ax_1] = 1
            self.remaining -= 1
            return AttackResult(hit=True, finished=(self.remaining == 0))

        # miss
        else:
            self.attacks[ax_0, ax_1] = -1
            return AttackResult(hit=False, finished=(self.remaining == 0))

    def randomly_place_all_ships(self, max_attempts_per_ship: int = 10_000) -> None:
        '''Randomly place all ships on the board.'''
        for length in self.ships:
            for _ in range(max_attempts_per_ship):
                direction = np.random.choice([Direction.HORIZONTAL, Direction.VERTICAL])

                if direction is Direction.HORIZONTAL:
                    y = np.random.randint(0, self.size)
                    x = np.random.randint(0, self.size - length + 1)
                else:
                    y = np.random.randint(0, self.size - length + 1)
                    x = np.random.randint(0, self.size)

                if self.is_valid_placement(y, x, length, direction):
                    self.place_ship(y, x, length, direction)
                    self.placing_index += 1
                    break
            else:
                raise RuntimeError(f"Failed to place ship of length {length} after {max_attempts_per_ship} attempts.")

        print("All ships placed randomly.")
        self.visualize(show_ships=True, show_attacks=False)

    def visualize(self, show_ships: bool = True, show_attacks: bool = True) -> None:
        BG_BLUE = "\033[44m"
        BG_GREY = "\033[100m"
        RESET = "\033[0m"

        def cell(y, x):
            is_ship = self.board[y, x] == 1
            is_hit = self.attacks[y, x] == 1
            is_miss = self.attacks[y, x] == -1

            if show_attacks and is_hit:
                return f"{BG_GREY}💥{RESET}"
            elif show_attacks and is_miss:
                return f"{BG_BLUE}💨{RESET}"
            elif show_ships and is_ship:
                return f"{BG_GREY}  {RESET}"
            else:
                return f"{BG_BLUE}  {RESET}"

        header = "   " + " ".join(f"{x:2d}" for x in range(self.size))
        print(header)

        for y in range(self.size):
            row = " ".join(cell(y, x) for x in range(self.size))
            print(f"{y:2d} {row}")