"""Monte Carlo runners that drive the engine.

These functions take primitive arguments (so they are easy to cache in
Streamlit) and return numpy arrays plus summary statistics.
"""

import numpy as np

from .rules import Rules
from .engine import Shoe, play_round
from .betting import make_bettor, count_spread
from . import stats


def _make_rules(num_decks, penetration, hit_soft_17, das, blackjack_payout=1.5):
    return Rules(
        num_decks=num_decks,
        penetration=penetration,
        hit_soft_17=hit_soft_17,
        double_after_split=das,
        blackjack_payout=blackjack_payout,
    )


def simulate_flat(strategy, n_rounds, *, num_decks=6, penetration=0.75,
                  hit_soft_17=False, das=True, blackjack_payout=1.5, seed=0):
    """Play ``n_rounds`` at a flat 1-unit bet. Returns the per-round result array.

    For the counting strategy the true count is tracked and fed into each round
    (so deviations/insurance apply) even though the bet stays flat.
    """
    rules = _make_rules(num_decks, penetration, hit_soft_17, das, blackjack_payout)
    rng = np.random.default_rng(seed)
    shoe = Shoe(rules, rng)
    results = np.empty(n_rounds, dtype=np.float64)
    for i in range(n_rounds):
        if shoe.needs_shuffle():
            shoe.reshuffle()
        tc = shoe.true_count() if strategy == "counting" else 0.0
        results[i] = play_round(shoe, strategy, rules, true_count=tc)
    return results


def edge_for_strategy(strategy, n_rounds, **kw):
    """Convenience: run a flat sim and return :class:`stats.EdgeStats`."""
    return stats.summarize(simulate_flat(strategy, n_rounds, **kw))


def simulate_counting(n_rounds, *, base=1.0, max_units=8.0, num_decks=6,
                      penetration=0.75, hit_soft_17=False, das=True,
                      blackjack_payout=1.5, seed=0):
    """Card counting: basic strategy + Illustrious-18 deviations + insurance,
    with a true-count bet spread.

    Returns a dict with per-round pnl, per-round bets, and the true count seen
    at the start of each round.
    """
    rules = _make_rules(num_decks, penetration, hit_soft_17, das, blackjack_payout)
    rng = np.random.default_rng(seed)
    shoe = Shoe(rules, rng)

    pnl = np.empty(n_rounds, dtype=np.float64)
    bets = np.empty(n_rounds, dtype=np.float64)
    tcs = np.empty(n_rounds, dtype=np.float64)

    for i in range(n_rounds):
        if shoe.needs_shuffle():
            shoe.reshuffle()
        tc = shoe.true_count()
        bet = count_spread(tc, base=base, max_units=max_units)
        result = play_round(shoe, "counting", rules, true_count=tc)
        tcs[i] = tc
        bets[i] = bet
        pnl[i] = result * bet

    total = pnl.sum()
    wagered = bets.sum()
    return {
        "pnl": pnl,
        "bets": bets,
        "true_counts": tcs,
        "total_pnl": float(total),
        "total_wagered": float(wagered),
        "advantage_pct": float(total / wagered * 100.0) if wagered else 0.0,
    }


def simulate_betting_system(system, n_rounds, *, base=1.0, start_bankroll=200.0,
                            table_max=500.0, strategy="basic", num_decks=6,
                            penetration=0.75, hit_soft_17=False, das=True,
                            blackjack_payout=1.5, seed=0):
    """Track one bankroll trajectory for a betting system. Returns a dict with
    the bankroll path and whether/when it went bust."""
    rules = _make_rules(num_decks, penetration, hit_soft_17, das, blackjack_payout)
    rng = np.random.default_rng(seed)
    shoe = Shoe(rules, rng)
    bettor = make_bettor(system, base=base, table_max=table_max)

    path = np.empty(n_rounds + 1, dtype=np.float64)
    path[0] = start_bankroll
    bankroll = start_bankroll
    ruined_at = -1

    for i in range(n_rounds):
        if shoe.needs_shuffle():
            shoe.reshuffle()
        bet = min(bettor.next_bet(), bankroll)
        if bet <= 0:
            path[i + 1:] = bankroll
            ruined_at = i
            break
        result = play_round(shoe, strategy, rules)
        bankroll += result * bet
        bettor.update(1 if result > 0 else (-1 if result < 0 else 0))
        path[i + 1] = bankroll
        if bankroll <= 0:
            path[i + 1:] = 0.0
            ruined_at = i
            break

    return {"path": path, "final": float(path[-1]), "ruined_at": ruined_at}


def simulate_betting_system_paths(system, n_rounds, n_paths, *, seed=0, **kw):
    """Run many independent bankroll trajectories for one betting system."""
    paths = np.empty((n_paths, n_rounds + 1), dtype=np.float64)
    finals = np.empty(n_paths, dtype=np.float64)
    ruined = np.zeros(n_paths, dtype=bool)
    for k in range(n_paths):
        out = simulate_betting_system(system, n_rounds, seed=seed + k, **kw)
        paths[k] = out["path"]
        finals[k] = out["final"]
        ruined[k] = out["ruined_at"] >= 0
    return {
        "paths": paths,
        "finals": finals,
        "risk_of_ruin": float(ruined.mean()),
        "mean_final": float(finals.mean()),
        "median_final": float(np.median(finals)),
    }
