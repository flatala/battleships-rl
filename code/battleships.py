import numpy as np
from enum import Enum, auto
from typing import NamedTuple

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

    def get_next_length(self) -> int:
        ''' Get the length of the next ship to be placed. '''

        # check if all ships have been placed
        if self.placing_index >= len(self.ships):
            raise IndexError("All ships have been placed.")

        # get next ship length
        length = self.ships[self.placing_index]
        self.placing_index += 1
        return length

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
                    break
            else:
                raise RuntimeError(f"Failed to place ship of length {length} after {max_attempts_per_ship} attempts.")

        print("All ships placed randomly.")
        self.visualize(show_ships=True, show_attacks=False)

    def visualize(self, show_ships: bool = True, show_attacks: bool = True) -> None:
        """
        Print an ASCII view of the board.

        - Ships: 'S' (if show_ships)
        - Unknown: '.'
        - Miss: 'o'
        - Hit:  'X'
        """
        yx = np.full((self.size, self.size), ".", dtype="<U1")

        if show_ships:
            yx[self.board == 1] = "S"

        if show_attacks:
            yx[self.attacks == -1] = "o"
            yx[self.attacks == 1] = "X"

        header = "   " + " ".join(f"{x:2d}" for x in range(self.size))
        print(header)
        for y in range(self.size):
            row = " ".join(f"{c:2s}" for c in yx[y])
            print(f"{y:2d} {row}")