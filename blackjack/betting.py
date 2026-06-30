"""Bet-sizing layers.

These sit on top of the engine. The engine returns a round result *per unit
bet*; a bettor decides how many units to wager each round. None of these change
the house edge — they only reshape variance and risk of ruin. That is exactly
the point the "betting systems" analysis is meant to demonstrate.
"""

from typing import List


class Bettor:
    """Base class. ``next_bet`` returns units to wager; ``update`` takes the
    sign of the round result (+1 win, -1 loss, 0 push)."""

    name = "flat"

    def __init__(self, base: float = 1.0, table_max: float = 1e9):
        self.base = base
        self.table_max = table_max

    def _cap(self, bet: float) -> float:
        return min(bet, self.table_max)

    def next_bet(self) -> float:
        return self._cap(self.base)

    def update(self, sign: int) -> None:
        pass


class Flat(Bettor):
    name = "Flat betting"


class Martingale(Bettor):
    """Double after every loss, reset to base after a win. The table maximum is
    what makes this lose catastrophically rather than 'always win'."""

    name = "Martingale"

    def __init__(self, base=1.0, table_max=1e9):
        super().__init__(base, table_max)
        self.bet = base

    def next_bet(self):
        return self._cap(self.bet)

    def update(self, sign):
        if sign < 0:
            self.bet = min(self.bet * 2, self.table_max)
        elif sign > 0:
            self.bet = self.base
        # push: keep the same bet


class Fibonacci(Bettor):
    """Advance one step in the Fibonacci sequence on a loss, step back two on a
    win."""

    name = "Fibonacci"

    def __init__(self, base=1.0, table_max=1e9):
        super().__init__(base, table_max)
        self.seq: List[int] = [1, 1]
        self.idx = 0

    def _fib(self, i):
        while len(self.seq) <= i:
            self.seq.append(self.seq[-1] + self.seq[-2])
        return self.seq[i]

    def next_bet(self):
        return self._cap(self.base * self._fib(self.idx))

    def update(self, sign):
        if sign < 0:
            self.idx += 1
        elif sign > 0:
            self.idx = max(0, self.idx - 2)


class DAlembert(Bettor):
    """Raise the bet one unit after a loss, lower it one unit after a win."""

    name = "D'Alembert"

    def __init__(self, base=1.0, table_max=1e9):
        super().__init__(base, table_max)
        self.level = 0

    def next_bet(self):
        return self._cap(self.base * (1 + self.level))

    def update(self, sign):
        if sign < 0:
            self.level += 1
        elif sign > 0:
            self.level = max(0, self.level - 1)


class Paroli(Bettor):
    """Anti-Martingale: double on a win up to three in a row, then reset; reset
    on any loss."""

    name = "Paroli"

    def __init__(self, base=1.0, table_max=1e9):
        super().__init__(base, table_max)
        self.bet = base
        self.streak = 0

    def next_bet(self):
        return self._cap(self.bet)

    def update(self, sign):
        if sign > 0:
            self.streak += 1
            if self.streak >= 3:
                self.bet = self.base
                self.streak = 0
            else:
                self.bet = min(self.bet * 2, self.table_max)
        elif sign < 0:
            self.bet = self.base
            self.streak = 0


BETTORS = {
    "flat": Flat,
    "martingale": Martingale,
    "fibonacci": Fibonacci,
    "dalembert": DAlembert,
    "paroli": Paroli,
}


def make_bettor(name: str, base: float, table_max: float) -> Bettor:
    return BETTORS[name](base=base, table_max=table_max)


# --- Card-counting bet spread ----------------------------------------------

def count_spread(true_count: float, base: float = 1.0, max_units: float = 8.0) -> float:
    """A simple Hi-Lo bet ramp: flat at low/negative counts, ramping up with the
    true count to a capped maximum. This is what converts the counter's
    information edge into money."""
    if true_count < 1:
        units = 1.0
    elif true_count < 2:
        units = 1.0
    elif true_count < 3:
        units = 2.0
    elif true_count < 4:
        units = 4.0
    elif true_count < 5:
        units = 6.0
    else:
        units = max_units
    return base * min(units, max_units)
