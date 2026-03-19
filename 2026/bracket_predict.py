import jax 
from jax import numpy as jnp
import pandas as pd 

def build_seeds_df(seeds_df, teams_df, season=2026):
    return (
        seeds_df[seeds_df["Season"] == season]
        .merge(teams_df[["TeamID", "TeamNameSpelling"]], on="TeamID", how="left")
        .reset_index(drop=True)
    )

def potential_bracket(samples, seeds_df, team_to_idx, n_draws=1000, rng_key=None):
    """
    DataFrame with one row per possible matchup (TeamID_a < TeamID_b)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Subsample posterior draws
    total_draws = samples["off"].shape[0]
    idx = jax.random.choice(rng_key, total_draws, shape=(n_draws,), replace=False)
    off_draws  = samples["off"][idx, :]    # (n_draws, T)
    deff_draws = samples["def"][idx, :]   # (n_draws, T)

    # Self join — mirrors R's inner_join with join=1
    # filter(TeamID_a != TeamID_b) & filter(TeamID_a < TeamID_b)
    bracket = (
        seeds_df
        .merge(seeds_df, how="cross", suffixes=("_a", "_b"))
        .query("Seed_a != Seed_b")          # different teams
        .query("Seed_a < Seed_b")           # avoid duplicates
        .reset_index(drop=True)
    )

    results = []
    for _, row in bracket.iterrows():
        team_a = row["TeamName_a"].upper()
        team_b = row["TeamName_b"].upper()

        if team_a not in team_to_idx or team_b not in team_to_idx:
            continue

        idx_a = team_to_idx[team_a]
        idx_b = team_to_idx[team_b]

        # mirrors R's: team_spread = value_a - value_b
        margin_draws = (
            (off_draws[:, idx_a] - deff_draws[:, idx_b]) -
            (off_draws[:, idx_b] - deff_draws[:, idx_a])
        )

        # mirrors R's: pred_prob = pred_prob_fn(total_spread, fit_win_prob)
        p_a_win_draws = jax.nn.sigmoid(0.15 * margin_draws)

        results.append({
            "seed_a":       row["Seed_a"],
            "team_a":       row["TeamName_a"],
            "seed_b":       row["Seed_b"],
            "team_b":       row["TeamName_b"],
            "margin_mean":  float(jnp.mean(margin_draws)),
            "margin_sd":    float(jnp.std(margin_draws)),
            "p_a_win_mean": float(jnp.mean(p_a_win_draws)),
            "p_a_win_sd":   float(jnp.std(p_a_win_draws)),
            "p_a_win_p5":   float(jnp.percentile(p_a_win_draws, 5)),
            "p_a_win_p95":  float(jnp.percentile(p_a_win_draws, 95)),
        })

    return pd.DataFrame(results).sort_values("margin_mean", ascending=False).reset_index(drop=True)