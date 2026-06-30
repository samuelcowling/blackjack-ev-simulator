"""Blackjack Monte Carlo EV Simulator — Streamlit dashboard.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blackjack.rules import Rules
from blackjack import stats
from blackjack.strategy import STRATEGY_LABELS
from blackjack.betting import BETTORS, count_spread
from blackjack.simulate import (
    simulate_flat,
    simulate_counting,
    simulate_betting_system_paths,
)

# --- Page setup & theme -----------------------------------------------------
st.set_page_config(page_title="Blackjack EV Simulator", page_icon="🃏", layout="wide")

ACCENT = "#2563eb"
GOOD = "#16a34a"
BAD = "#dc2626"
GRID = "rgba(148,163,184,0.18)"
PALETTE = {
    "basic": "#2563eb",
    "mimic": "#f59e0b",
    "never_bust": "#dc2626",
    "counting": "#16a34a",
}


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing:-0.01em;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _layout(fig, height=420, ytitle="", xtitle=""):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title=xtitle)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID, title=ytitle)
    return fig


# --- Cached simulation wrappers --------------------------------------------
# Rule settings are bundled into a hashable tuple ``rp`` so changing any rule in
# the sidebar invalidates the cache and re-runs the simulations.
def _rules_from_rp(rp):
    nd, pen, h17, das, bj = rp
    return dict(num_decks=nd, penetration=pen, hit_soft_17=h17, das=das,
                blackjack_payout=bj)


@st.cache_data(show_spinner=False)
def run_flat(strategy, n_rounds, seed, rp):
    return simulate_flat(strategy, n_rounds, seed=seed, **_rules_from_rp(rp))


@st.cache_data(show_spinner=False)
def run_counting(n_rounds, max_units, seed, rp):
    return simulate_counting(n_rounds, max_units=max_units, seed=seed,
                             **_rules_from_rp(rp))


def counting_edge_stats(c):
    """Bet-weighted edge for the counter: each round's P&L is normalised by the
    *average* bet so the mean equals total-won / total-wagered (the real
    advantage). Dividing by each round's own bet would erase the edge that bet
    spreading creates."""
    return stats.summarize(c["pnl"] / c["bets"].mean())


@st.cache_data(show_spinner=False)
def run_systems(system, n_rounds, n_paths, base, start_bankroll, table_max, seed, rp):
    return simulate_betting_system_paths(
        system, n_rounds, n_paths, base=base,
        start_bankroll=start_bankroll, table_max=table_max, seed=seed,
        **_rules_from_rp(rp),
    )


def _logspace_idx(n, k=2000):
    """Indices for downsampling a length-n array to ~k log-spaced points."""
    if n <= k:
        return np.arange(n)
    idx = np.unique(np.geomspace(1, n, k).astype(int) - 1)
    return idx[idx < n]


# --- Sidebar ----------------------------------------------------------------
st.sidebar.header("Simulation controls")

n_rounds = st.sidebar.select_slider(
    "Hands per simulation",
    options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
    value=200_000,
    help="More hands → tighter confidence intervals, slower runs (cached after first run).",
)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Table rules")
num_decks = st.sidebar.selectbox("Decks", [1, 2, 4, 6, 8], index=3)
h17 = st.sidebar.checkbox("Dealer hits soft 17 (H17)", value=False,
                          help="H17 adds roughly +0.2% to the house edge.")
payout_choice = st.sidebar.radio("Blackjack pays", ["3:2", "6:5"], horizontal=True,
                                 help="6:5 is a notoriously bad rule — it adds ~1.4%.")
bj_payout = 1.5 if payout_choice == "3:2" else 1.2
das = st.sidebar.checkbox("Double after split (DAS)", value=True)
penetration = st.sidebar.slider("Deck penetration", 0.50, 0.90, 0.75, 0.05,
                                help="How deep the shoe is dealt before reshuffling. "
                                     "Deeper penetration helps a card counter.")

# Bundle rules for cache keys, and a Rules object for display.
rp = (num_decks, penetration, h17, das, bj_payout)
RULES = Rules(num_decks=num_decks, penetration=penetration, hit_soft_17=h17,
              double_after_split=das, blackjack_payout=bj_payout)
st.sidebar.info(RULES.describe())

st.sidebar.markdown("---")
st.sidebar.caption("These two figures only affect the **Betting systems** tab.")
start_bankroll = st.sidebar.number_input("Starting bankroll (units)", 50, 5000, 200, 50)
table_max = st.sidebar.number_input("Table maximum bet (units)", 10, 5000, 500, 10)


# --- Header -----------------------------------------------------------------
st.title("🃏 Blackjack Monte Carlo EV Simulator")
st.markdown(
    "A Monte Carlo engine that deals millions of hands to measure the **house edge**, "
    "compare **playing strategies**, stress-test **betting systems**, and show how "
    "**card counting** flips the advantage to the player. "
    f"<span style='color:#64748b'>Rules: {RULES.describe()}.</span>",
    unsafe_allow_html=True,
)

tab_cmp, tab_conv, tab_bet, tab_count = st.tabs(
    ["📊 Strategy comparison", "📉 Edge convergence", "🎲 Betting systems", "🧮 Card counting"]
)


# ===========================================================================
# TAB 1 — Strategy comparison
# ===========================================================================
with tab_cmp:
    st.subheader("How much does your strategy cost you?")
    st.caption(
        "House edge = expected loss per hand as a % of your bet, with 95% confidence "
        "intervals. Lower is better for the player."
    )
    strat_order = ["basic", "mimic", "never_bust", "counting"]
    rows, edges, errs, colors, labels = [], [], [], [], []

    prog = st.progress(0.0, text="Dealing hands…")
    for i, strat in enumerate(strat_order):
        if strat == "counting":
            c = run_counting(n_rounds, 8.0, seed, rp)
            s = counting_edge_stats(c)  # bet-weighted edge per unit wagered
        else:
            s = stats.summarize(run_flat(strat, n_rounds, seed, rp))
        lo, hi = s.ci_pct
        edges.append(s.house_edge_pct)
        errs.append((hi - lo) / 2.0)
        colors.append(PALETTE[strat])
        labels.append(STRATEGY_LABELS[strat])
        rows.append({
            "Strategy": STRATEGY_LABELS[strat],
            "House edge": f"{s.house_edge_pct:+.3f}%",
            "95% CI": f"[{lo:+.3f}%, {hi:+.3f}%]",
            "Std dev (units)": f"{s.std:.3f}",
            "Hands": f"{s.n:,}",
        })
        prog.progress((i + 1) / len(strat_order), text=f"Simulated {STRATEGY_LABELS[strat]}")
    prog.empty()

    c1, c2, c3, c4 = st.columns(4)
    for col, strat, edge in zip([c1, c2, c3, c4], strat_order, edges):
        verdict = "player edge" if edge < 0 else "house edge"
        col.metric(STRATEGY_LABELS[strat], f"{edge:+.2f}%", verdict,
                   delta_color="inverse")

    fig = go.Figure()
    fig.add_bar(
        x=labels, y=edges,
        marker_color=colors,
        error_y=dict(type="data", array=errs, color="rgba(100,116,139,0.8)", thickness=1.5),
        text=[f"{e:+.2f}%" for e in edges], textposition="outside",
    )
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.6)")
    _layout(fig, ytitle="House edge (%)")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "📌 Basic strategy keeps the house edge near its theoretical floor (~0.4%). "
        "Intuitive-but-wrong play (mimic / never-bust) costs **10×+** more. Card counting "
        "with a bet spread is the only approach that turns the edge negative — i.e. "
        "profitable for the player."
    )


# ===========================================================================
# TAB 2 — Edge convergence
# ===========================================================================
with tab_conv:
    st.subheader("The law of large numbers in action")
    st.caption(
        "Each curve is the running house edge as hands accumulate. Early on, luck "
        "dominates and the estimate swings wildly; with volume it converges to the "
        "true edge and the 95% confidence band collapses."
    )
    chosen = st.multiselect(
        "Strategies to plot",
        options=strat_order, default=["basic", "mimic"],
        format_func=lambda s: STRATEGY_LABELS[s],
    )

    fig = go.Figure()
    for strat in chosen:
        if strat == "counting":
            c = run_counting(n_rounds, 8.0, seed, rp)
            results = c["pnl"] / c["bets"].mean()  # bet-weighted, preserves the edge
        else:
            results = run_flat(strat, n_rounds, seed, rp)
        rm = stats.running_mean(results) * -100.0   # convert to house-edge %
        sem = stats.running_sem(results) * 100.0
        idx = _logspace_idx(len(rm))
        x = idx + 1
        color = PALETTE[strat]
        band = stats.Z95 * sem[idx]
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([rm[idx] + band, (rm[idx] - band)[::-1]]),
            fill="toself", fillcolor=_rgba(color, 0.10),
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=rm[idx], name=STRATEGY_LABELS[strat],
            line=dict(color=color, width=2.2),
        ))
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.5)", line_dash="dot")
    fig.update_xaxes(type="log")
    _layout(fig, height=460, ytitle="Running house edge (%)", xtitle="Hands played (log scale)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "📌 Notice how the band narrows roughly with 1/√n — quadrupling the hands "
        "halves the uncertainty. This is why a casino with millions of hands has "
        "essentially zero risk, while a single player's night is mostly variance."
    )


# ===========================================================================
# TAB 3 — Betting systems
# ===========================================================================
with tab_bet:
    st.subheader("Do betting systems beat the house? (No.)")
    st.caption(
        "Martingale, Fibonacci, D'Alembert and Paroli all reshape *variance* — but "
        "none change the expected value. Played on basic strategy, every one of them "
        "still bleeds money at the same rate; some just go bust faster."
    )

    sys_choice = st.multiselect(
        "Betting systems to test",
        options=["flat", "martingale", "fibonacci", "dalembert", "paroli"],
        default=["flat", "martingale", "dalembert"],
        format_func=lambda s: BETTORS[s].name,
    )
    cset = st.columns(3)
    rounds_per_path = cset[0].select_slider(
        "Hands per session", [200, 500, 1000, 2000, 5000], value=1000)
    n_paths = cset[1].select_slider("Simulated sessions", [50, 100, 200, 500], value=200)
    base_unit = cset[2].select_slider("Base bet (units)", [1, 2, 5, 10], value=1)

    if not sys_choice:
        st.info("Pick at least one betting system above.")
    else:
        summary_rows = []
        traj_fig = go.Figure()
        hist_fig = go.Figure()
        sys_colors = ["#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#7c3aed"]

        for ci, system in enumerate(sys_choice):
            res = run_systems(system, rounds_per_path, n_paths, base_unit,
                              start_bankroll, table_max, seed, rp)
            color = sys_colors[ci % len(sys_colors)]
            paths = res["paths"]

            # a handful of sample trajectories + the median path
            sample = paths[: min(12, len(paths))]
            for j, p in enumerate(sample):
                traj_fig.add_trace(go.Scatter(
                    y=p, line=dict(color=color, width=1), opacity=0.28,
                    showlegend=False, hoverinfo="skip"))
            median_path = np.median(paths, axis=0)
            traj_fig.add_trace(go.Scatter(
                y=median_path, name=BETTORS[system].name,
                line=dict(color=color, width=2.6)))

            hist_fig.add_trace(go.Histogram(
                x=res["finals"], name=BETTORS[system].name,
                marker_color=color, opacity=0.6, nbinsx=40))

            summary_rows.append({
                "System": BETTORS[system].name,
                "Mean final bankroll": f"{res['mean_final']:.1f}",
                "Median final": f"{res['median_final']:.1f}",
                "Net P&L": f"{res['mean_final'] - start_bankroll:+.1f}",
                "Risk of ruin": f"{res['risk_of_ruin'] * 100:.1f}%",
            })

        traj_fig.add_hline(y=start_bankroll, line_dash="dot",
                           line_color="rgba(100,116,139,0.7)",
                           annotation_text="starting bankroll")
        traj_fig.add_hline(y=0, line_color=BAD, line_width=1)
        _layout(traj_fig, height=420, ytitle="Bankroll (units)",
                xtitle="Hands played")
        traj_fig.update_layout(title="Bankroll trajectories (faint = sample sessions, bold = median)")

        hist_fig.update_layout(barmode="overlay")
        _layout(hist_fig, height=340, ytitle="Sessions", xtitle="Final bankroll (units)")
        hist_fig.add_vline(x=start_bankroll, line_dash="dot",
                           line_color="rgba(100,116,139,0.7)")

        st.plotly_chart(traj_fig, width="stretch")
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.plotly_chart(hist_fig, width="stretch")
        with cc2:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch",
                         hide_index=True)
        st.caption(
            "📌 Martingale produces lots of small wins and rare catastrophic losses "
            "(a fat left tail + high risk of ruin once the table maximum blocks the "
            "next double). Flat betting has the lowest variance. Across every system "
            "the **mean** bankroll declines at the same house-edge rate — variance "
            "shuffles *when* you lose, never *whether* you do."
        )


# ===========================================================================
# TAB 4 — Card counting
# ===========================================================================
with tab_count:
    st.subheader("Card counting: turning information into an edge")
    st.caption(
        "Hi-Lo assigns +1 to low cards (2-6), 0 to 7-9, and −1 to tens and aces. "
        "When the running count (÷ decks remaining = *true count*) is high, the "
        "remaining shoe is rich in tens and aces, which favours the player — so the "
        "counter **bets more**, **takes insurance** at a true count of 3+, and makes "
        "the **Illustrious-18 deviations** from basic strategy."
    )

    max_units = st.select_slider(
        "Max bet spread (units at the highest counts)",
        options=[4, 6, 8, 12, 16], value=8)

    c = run_counting(n_rounds, float(max_units), seed, rp)
    flat = stats.summarize(run_flat("basic", n_rounds, seed, rp))
    s_count = counting_edge_stats(c)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flat basic strategy", f"{flat.house_edge_pct:+.2f}%", "house edge",
              delta_color="inverse")
    m2.metric("Counting (Hi-Lo)", f"{s_count.house_edge_pct:+.2f}%",
              "player edge" if s_count.house_edge_pct < 0 else "house edge",
              delta_color="inverse")
    m3.metric("Net result", f"{c['total_pnl']:+,.0f} units")
    m4.metric("Total wagered", f"{c['total_wagered']:,.0f} units")

    # Bet-spread schedule
    tc_grid = np.arange(-2, 7)
    spread = [count_spread(t, base=1.0, max_units=float(max_units)) for t in tc_grid]
    csp = pd.DataFrame({"True count": tc_grid, "Units bet": spread})

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown("**Bet ramp by true count**")
        bar = go.Figure(go.Bar(x=tc_grid, y=spread, marker_color=ACCENT,
                               text=spread, textposition="outside"))
        _layout(bar, height=300, ytitle="Units bet", xtitle="True count")
        st.plotly_chart(bar, width="stretch")
    with cg2:
        st.markdown("**Cumulative profit: counting vs flat**")
        idx = _logspace_idx(n_rounds)
        cum_count = np.cumsum(c["pnl"])[idx]
        cum_flat = np.cumsum(run_flat("basic", n_rounds, seed, rp))[idx]
        line = go.Figure()
        line.add_trace(go.Scatter(x=idx + 1, y=cum_count, name="Counting",
                                  line=dict(color=GOOD, width=2.4)))
        line.add_trace(go.Scatter(x=idx + 1, y=cum_flat, name="Flat basic",
                                  line=dict(color=ACCENT, width=2.0)))
        line.add_hline(y=0, line_color="rgba(100,116,139,0.5)", line_dash="dot")
        _layout(line, height=300, ytitle="Cumulative units", xtitle="Hands played")
        st.plotly_chart(line, width="stretch")

    st.caption(
        "📌 The flat basic-strategy line drifts downward (the house edge), while the "
        "counter's cumulative profit trends upward — but notice the swings: counting "
        "wins in the long run yet rides large short-term variance. This model includes "
        "the Illustrious-18 deviations and insurance, but still idealises away real-world "
        "frictions (casino 'heat', bet-sizing errors, and the full set of rarer index plays)."
    )

st.markdown("---")
st.caption(
    "Built with NumPy · pandas · Streamlit · Plotly. Engine: a full 6-deck blackjack "
    "simulator with splits, doubles, naturals and a live Hi-Lo count. All figures are "
    "Monte Carlo estimates with 95% confidence intervals."
)
