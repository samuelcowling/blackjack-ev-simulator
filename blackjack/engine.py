"""The blackjack engine: a shoe with a running Hi-Lo count, plus the logic to
play a single round (player decisions with splits/doubles, dealer play, and
settlement).

:func:`play_round` returns the net result of a round in **units of the base
bet** (so a flat $1 bettor's bankroll change). Doubles and splits already scale
this result, so a separate bet-sizing layer can simply multiply by the wager.
"""

import numpy as np

from .rules import Rules
from .strategy import decide, hand_total, hilo_value, take_insurance


class Shoe:
    """A multi-deck shoe that deals sequentially and tracks the Hi-Lo count."""

    def __init__(self, rules: Rules, rng: np.random.Generator):
        self.rng = rng
        self.num_decks = rules.num_decks
        single = [v for v in range(2, 10) for _ in range(4)] + [10] * 16 + [11] * 4
        self.cards = np.array(single * rules.num_decks, dtype=np.int8)
        self.cut = int(len(self.cards) * rules.penetration)
        self.reshuffle()

    def reshuffle(self):
        self.rng.shuffle(self.cards)
        self.pos = 0
        self.running_count = 0

    def needs_shuffle(self) -> bool:
        return self.pos >= self.cut

    def draw(self) -> int:
        # Safety net: a round that runs deep into a shallow (e.g. single-deck)
        # shoe can need more cards than remain. Casinos finish the hand from a
        # fresh shuffle; we do the same. This is rare enough not to distort the
        # count meaningfully, and never fires for multi-deck shoes.
        if self.pos >= len(self.cards):
            self.reshuffle()
        c = int(self.cards[self.pos])
        self.pos += 1
        self.running_count += hilo_value(c)
        return c

    def decks_remaining(self) -> float:
        return (len(self.cards) - self.pos) / 52.0

    def true_count(self) -> float:
        """Running count normalised per remaining deck (Hi-Lo true count)."""
        dr = self.decks_remaining()
        return self.running_count / dr if dr >= 0.5 else self.running_count * 2.0


def _play_player_hands(shoe, cards, dealer_up, strategy, rules, true_count):
    """Play out the player's hand(s), resolving splits. Returns a list of
    ``(total, busted, wager)`` tuples — one per final hand."""
    finished = []
    stack = [{"cards": list(cards), "from_split": False, "ace_split": False}]
    n_splits = 0

    while stack:
        hand = stack.pop()
        c = hand["cards"]

        # Split aces receive exactly one card each and must stand.
        if hand["ace_split"]:
            t, _ = hand_total(c)
            finished.append((t, t > 21, 1.0))
            continue

        while True:
            total, _ = hand_total(c)
            if total > 21:
                finished.append((total, True, 1.0))
                break
            if total == 21:
                finished.append((total, False, 1.0))
                break

            can_double = len(c) == 2 and (not hand["from_split"] or rules.double_after_split)
            can_split = len(c) == 2 and c[0] == c[1] and n_splits < rules.max_splits
            action = decide(strategy, c, dealer_up, can_double, can_split, true_count)

            if action == "P":
                n_splits += 1
                is_ace = c[0] == 11
                for rank in (c[0], c[1]):
                    stack.append({
                        "cards": [rank, shoe.draw()],
                        "from_split": True,
                        "ace_split": is_ace,
                    })
                break
            elif action == "D":
                c.append(shoe.draw())
                t, _ = hand_total(c)
                finished.append((t, t > 21, 2.0))  # doubled wager
                break
            elif action == "H":
                c.append(shoe.draw())
            else:  # 'S'
                finished.append((total, False, 1.0))
                break

    return finished


def _play_dealer(shoe, cards, rules):
    c = list(cards)
    while True:
        total, soft = hand_total(c)
        if total >= 17:
            if total == 17 and soft and rules.hit_soft_17:
                c.append(shoe.draw())
                continue
            return total
        c.append(shoe.draw())


def play_round(shoe: Shoe, strategy: str, rules: Rules, true_count: float = 0.0) -> float:
    """Play one round and return the net result in base-bet units.

    ``true_count`` is only consulted by the counting strategy (for Illustrious-18
    deviations and insurance) and is otherwise ignored.
    """
    player = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]  # dealer[1] is the hole card
    dealer_up = dealer[0]

    player_bj = hand_total(player)[0] == 21
    dealer_bj = hand_total(dealer)[0] == 21

    # Insurance: a counting-only side bet of half the base bet when the dealer
    # shows an ace and the shoe is ten-rich. Pays 2:1 if the dealer has a natural.
    insurance = 0.0
    if strategy == "counting" and dealer_up == 11 and take_insurance(true_count, rules.insurance_index):
        insurance = 1.0 if dealer_bj else -0.5

    if player_bj or dealer_bj:
        if player_bj and dealer_bj:
            return insurance + 0.0
        return insurance + (rules.blackjack_payout if player_bj else -1.0)

    hands = _play_player_hands(shoe, player, dealer_up, strategy, rules, true_count)

    if any(not busted for _, busted, _ in hands):
        dealer_total = _play_dealer(shoe, dealer, rules)
    else:
        dealer_total = hand_total(dealer)[0]  # all hands busted; dealer stands pat

    result = insurance
    for total, busted, wager in hands:
        if busted:
            result -= wager
        elif dealer_total > 21 or total > dealer_total:
            result += wager
        elif total < dealer_total:
            result -= wager
        # equal totals push
    return result
