"""Render static PNGs of the key charts from real simulation data.

Outputs to ``assets/``. These are used in the README and as quick previews.
Run:  python scripts/make_charts.py
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack import stats
from blackjack.strategy import STRATEGY_LABELS
from blackjack.simulate import (
    simulate_flat, simulate_counting, simulate_betting_system_paths,
)

N = 150_000
SEED = 7
PALETTE = {"basic": "#2563eb", "mimic": "#f59e0b", "never_bust": "#dc2626", "counting": "#16a34a"}
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)

BASE = dict(paper_bgcolor="white", plot_bgcolor="white", font=dict(size=15),
            margin=dict(l=60, r=30, t=60, b=55))
GRID = "rgba(148,163,184,0.25)"


def save(fig, name):
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID)
    path = os.path.join(ASSETS, name)
    fig.write_image(path, width=1000, height=560, scale=2)
    print("wrote", path)


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# 1. Strategy comparison ----------------------------------------------------
labels, edges, errs, colors = [], [], [], []
for strat in ["basic", "mimic", "never_bust", "counting"]:
    if strat == "counting":
        c = simulate_counting(N, seed=SEED)
        s = stats.summarize(c["pnl"] / c["bets"].mean())
    else:
        s = stats.summarize(simulate_flat(strat, N, seed=SEED))
    lo, hi = s.ci_pct
    labels.append(STRATEGY_LABELS[strat]); edges.append(s.house_edge_pct)
    errs.append((hi - lo) / 2); colors.append(PALETTE[strat])

fig = go.Figure(go.Bar(x=labels, y=edges, marker_color=colors,
                       error_y=dict(type="data", array=errs, thickness=1.5),
                       text=[f"{e:+.2f}%" for e in edges], textposition="outside"))
fig.add_hline(y=0, line_color="rgba(100,116,139,0.6)")
fig.update_layout(title="House edge by strategy (95% CI)", yaxis_title="House edge (%)", **BASE)
save(fig, "strategy_comparison.png")

# 2. Convergence ------------------------------------------------------------
fig = go.Figure()
for strat in ["basic", "mimic", "counting"]:
    if strat == "counting":
        c = simulate_counting(N, seed=SEED)
        res = c["pnl"] / c["bets"].mean()
    else:
        res = simulate_flat(strat, N, seed=SEED)
    rm = stats.running_mean(res) * -100
    sem = stats.running_sem(res) * 100
    idx = np.unique(np.geomspace(1, len(rm), 1500).astype(int) - 1)
    x = idx + 1
    band = stats.Z95 * sem[idx]
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([rm[idx] + band, (rm[idx] - band)[::-1]]),
                             fill="toself", fillcolor=_rgba(PALETTE[strat], 0.12),
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=rm[idx], name=STRATEGY_LABELS[strat],
                             line=dict(color=PALETTE[strat], width=2.4)))
fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,116,139,0.5)")
fig.update_xaxes(type="log")
fig.update_layout(title="House edge converges with volume (95% band)",
                  xaxis_title="Hands played (log)", yaxis_title="Running house edge (%)",
                  legend=dict(orientation="h", y=1.04), **BASE)
save(fig, "convergence.png")

# 3. Betting systems --------------------------------------------------------
fig = go.Figure()
sys_colors = {"flat": "#2563eb", "martingale": "#dc2626", "dalembert": "#f59e0b"}
for system, color in sys_colors.items():
    res = simulate_betting_system_paths(system, 1000, 150, base=1.0,
                                        start_bankroll=200, table_max=500, seed=SEED)
    for p in res["paths"][:14]:
        fig.add_trace(go.Scatter(y=p, line=dict(color=color, width=1), opacity=0.25,
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=np.median(res["paths"], axis=0), name=system.title(),
                             line=dict(color=color, width=2.8)))
fig.add_hline(y=200, line_dash="dot", line_color="rgba(100,116,139,0.7)")
fig.add_hline(y=0, line_color="#dc2626", line_width=1)
fig.update_layout(title="Betting systems reshape variance, not expectation",
                  xaxis_title="Hands played", yaxis_title="Bankroll (units)",
                  legend=dict(orientation="h", y=1.04), **BASE)
save(fig, "betting_systems.png")

# 4. Counting vs flat -------------------------------------------------------
c = simulate_counting(N, seed=SEED)
flat = simulate_flat("basic", N, seed=SEED)
idx = np.unique(np.geomspace(1, N, 1500).astype(int) - 1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx + 1, y=np.cumsum(c["pnl"])[idx], name="Card counting",
                         line=dict(color="#16a34a", width=2.6)))
fig.add_trace(go.Scatter(x=idx + 1, y=np.cumsum(flat)[idx], name="Flat basic",
                         line=dict(color="#2563eb", width=2.2)))
fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,116,139,0.5)")
fig.update_layout(title="Cumulative profit: counting vs flat basic strategy",
                  xaxis_title="Hands played", yaxis_title="Cumulative units",
                  legend=dict(orientation="h", y=1.04), **BASE)
save(fig, "counting.png")

print("done")
