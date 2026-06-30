"""Table rules.

The default rule set is a realistic, widely-used Las Vegas Strip game:

    * 6-deck shoe
    * Dealer STANDS on soft 17 (S17)
    * Blackjack pays 3:2
    * Double on any first two cards
    * Double after split allowed (DAS)
    * Re-split to a maximum of 4 hands (3 splits); split aces get one card each
    * ~75% deck penetration before the shuffle

Under these rules the house edge against perfect basic strategy is roughly
0.4-0.5%, which is the number the simulator should reproduce as a sanity check.
"""

from dataclasses import dataclass


@dataclass
class Rules:
    num_decks: int = 6
    penetration: float = 0.75          # fraction of the shoe dealt before reshuffle
    hit_soft_17: bool = False          # False => dealer STANDS on soft 17 (S17)
    blackjack_payout: float = 1.5      # 3:2
    double_after_split: bool = True    # DAS
    max_splits: int = 3                # max additional hands from splitting (=> 4 hands)
    insurance_index: float = 3.0       # counter takes insurance at this true count or higher

    def payout_label(self) -> str:
        return {1.5: "3:2", 1.2: "6:5", 2.0: "2:1", 1.0: "1:1"}.get(
            self.blackjack_payout, f"{self.blackjack_payout:g}:1")

    def describe(self) -> str:
        s17 = "Stands on soft 17 (S17)" if not self.hit_soft_17 else "Hits soft 17 (H17)"
        return (
            f"{self.num_decks} decks  |  {s17}  |  "
            f"Blackjack pays {self.payout_label()}  |  "
            f"{'DAS' if self.double_after_split else 'no DAS'}  |  "
            f"{int(self.penetration * 100)}% penetration"
        )
