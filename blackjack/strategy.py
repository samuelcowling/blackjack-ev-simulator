"""Player strategies and the basic-strategy chart.

Cards are represented by their blackjack value: 2-9 are themselves, every
ten-valued card (10/J/Q/K) is ``10``, and an ace is ``11`` (downgraded to 1 by
:func:`hand_total` when it would otherwise bust).

A strategy is a function ``decide(cards, dealer_up, can_double, can_split) -> action``
where ``action`` is one of:

    'H' hit   'S' stand   'D' double   'P' split

The decision dispatcher :func:`decide` selects the strategy by name.
"""

from typing import List


def hand_total(cards: List[int]):
    """Return ``(total, is_soft)`` for a hand, counting aces as 11 then 1.

    ``is_soft`` is True when an ace is still being counted as 11.
    """
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, aces > 0


def hilo_value(card: int) -> int:
    """Hi-Lo running-count contribution of a single card."""
    if card <= 6:        # 2,3,4,5,6
        return 1
    if card >= 10:       # 10,J,Q,K,A
        return -1
    return 0             # 7,8,9


# --- Basic strategy (6-deck, S17, DAS) -------------------------------------
# Codes: 'H' hit, 'S' stand, 'Dh' double-else-hit, 'Ds' double-else-stand, 'P' split.
# Dealer upcard ``up`` ranges 2..11 (11 = ace).

def _hard_code(total: int, up: int) -> str:
    if total >= 17:
        return "S"
    if 13 <= total <= 16:
        return "S" if up <= 6 else "H"
    if total == 12:
        return "S" if 4 <= up <= 6 else "H"
    if total == 11:
        return "Dh"
    if total == 10:
        return "Dh" if up <= 9 else "H"
    if total == 9:
        return "Dh" if 3 <= up <= 6 else "H"
    return "H"  # 5-8


def _soft_code(total: int, up: int) -> str:
    if total >= 20:
        return "S"                       # soft 20, 21
    if total == 19:
        return "Ds" if up == 6 else "S"  # double vs 6 only
    if total == 18:
        if 3 <= up <= 6:
            return "Ds"
        if up in (2, 7, 8):
            return "S"
        return "H"                       # vs 9, 10, A
    if total == 17:
        return "Dh" if 3 <= up <= 6 else "H"
    if total in (15, 16):
        return "Dh" if 4 <= up <= 6 else "H"
    if total in (13, 14):
        return "Dh" if 5 <= up <= 6 else "H"
    return "H"                           # soft 12 (e.g. unsplit A,A fallback)


# Dealer upcards on which each pair is split (DAS rules). 5s and 10s never split.
_PAIR_SPLIT = {
    2: {2, 3, 4, 5, 6, 7},
    3: {2, 3, 4, 5, 6, 7},
    4: {5, 6},
    6: {2, 3, 4, 5, 6},
    7: {2, 3, 4, 5, 6, 7},
    8: {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    9: {2, 3, 4, 5, 6, 8, 9},
    11: {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
}


def _resolve(code: str, can_double: bool) -> str:
    """Turn a chart code into a legal action given whether doubling is allowed."""
    if code == "Dh":
        return "D" if can_double else "H"
    if code == "Ds":
        return "D" if can_double else "S"
    return code  # 'H', 'S'


def basic_action(cards, dealer_up, can_double, can_split):
    """The mathematically optimal play under the default rules."""
    if can_split and len(cards) == 2 and cards[0] == cards[1]:
        pc = cards[0]
        if dealer_up in _PAIR_SPLIT.get(pc, set()):
            return "P"
        # otherwise fall through and treat as a normal hard/soft total
    total, soft = hand_total(cards)
    code = _soft_code(total, dealer_up) if soft else _hard_code(total, dealer_up)
    return _resolve(code, can_double)


def mimic_dealer_action(cards, dealer_up, can_double, can_split):
    """Play your hand exactly like the dealer: hit < 17, stand on 17+."""
    total, _ = hand_total(cards)
    return "H" if total < 17 else "S"


def never_bust_action(cards, dealer_up, can_double, can_split):
    """The 'intuitive' bad player who never risks busting (stands on hard 12+)."""
    total, soft = hand_total(cards)
    if soft and total < 18:
        return "H"
    if (not soft) and total <= 11:
        return "H"
    return "S"


# --- Card counting: the "Illustrious 18" index deviations ------------------
# A counter starts from basic strategy and deviates when the Hi-Lo true count
# crosses a known index. Each entry is keyed by
#     (player_total, is_soft, is_pair_of_tens, dealer_upcard)
# and maps to (index, action_at_or_above_index, action_below_index). The engine
# downgrades an illegal Double/Split to Hit/Stand. These 18 plays (Schlesinger's
# "Illustrious 18", Hi-Lo, S17) capture the large majority of the value that
# playing deviations add on top of bet spreading.
ILLUSTRIOUS_18 = {
    # play                       index  >=idx  <idx
    (16, False, False, 10):     (0,    "S",   "H"),
    (15, False, False, 10):     (4,    "S",   "H"),
    (20, False, True, 5):       (5,    "P",   "S"),   # 10,10 vs 5 -> split when rich
    (20, False, True, 6):       (4,    "P",   "S"),   # 10,10 vs 6
    (10, False, False, 10):     (4,    "D",   "H"),
    (10, False, False, 11):     (4,    "D",   "H"),   # 10 vs A
    (12, False, False, 3):      (2,    "S",   "H"),
    (12, False, False, 2):      (3,    "S",   "H"),
    (9,  False, False, 2):      (1,    "D",   "H"),
    (9,  False, False, 7):      (3,    "D",   "H"),
    (16, False, False, 9):      (5,    "S",   "H"),
    (13, False, False, 2):      (-1,   "S",   "H"),   # hit only when very negative
    (13, False, False, 3):      (-2,   "S",   "H"),
    (12, False, False, 4):      (0,    "S",   "H"),
    (12, False, False, 5):      (-2,   "S",   "H"),
    (12, False, False, 6):      (-1,   "S",   "H"),
}


def take_insurance(true_count: float, index: float = 3.0) -> bool:
    """Insurance becomes a +EV side bet once the shoe is ten-rich. This is the
    single most valuable index play in the Illustrious 18."""
    return true_count >= index


def counting_action(cards, dealer_up, can_double, can_split, true_count):
    """Basic strategy, overridden by any applicable Illustrious-18 deviation."""
    total, soft = hand_total(cards)
    is_pair_tens = len(cards) == 2 and cards[0] == 10 and cards[1] == 10
    dev = ILLUSTRIOUS_18.get((total, soft, is_pair_tens, dealer_up))
    if dev is not None:
        index, at_or_above, below = dev
        action = at_or_above if true_count >= index else below
        if action == "P" and not can_split:
            action = "S"
        if action == "D" and not can_double:
            action = "H"
        return action
    return basic_action(cards, dealer_up, can_double, can_split)


# Non-counting strategies ignore the true count.
_STRATEGIES = {
    "basic": basic_action,
    "mimic": mimic_dealer_action,
    "never_bust": never_bust_action,
}

STRATEGY_LABELS = {
    "basic": "Basic strategy",
    "mimic": "Mimic the dealer",
    "never_bust": "Never bust",
    "counting": "Card counting (Hi-Lo)",
}


def decide(strategy: str, cards, dealer_up, can_double, can_split, true_count=0.0) -> str:
    if strategy == "counting":
        return counting_action(cards, dealer_up, can_double, can_split, true_count)
    return _STRATEGIES[strategy](cards, dealer_up, can_double, can_split)
