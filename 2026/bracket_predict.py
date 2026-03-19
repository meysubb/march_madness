import jax 
import numpy as np
from scipy.stats import norm
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
    off_draws   = samples["off"][idx, :]    # (n_draws, T)
    deff_draws  = samples["def"][idx, :]    # (n_draws, T)
    sigma_draws = samples["sigma"][idx]     # (n_draws,)

    bracket = (
        seeds_df
        .merge(seeds_df, how="cross", suffixes=("_a", "_b"))
        .query("Seed_a != Seed_b")
        .query("Seed_a < Seed_b")
        .reset_index(drop=True)
        .rename(columns={"TeamNameSpelling_a": "Team_a", "TeamNameSpelling_b": "Team_b"})
    )

    results = []
    for _, row in bracket.iterrows():
        team_a = row["Team_a"].upper()
        team_b = row["Team_b"].upper()

        if team_a not in team_to_idx or team_b not in team_to_idx:
            continue

        idx_a = team_to_idx[team_a]
        idx_b = team_to_idx[team_b]

        # Expected score differential: (off_a - def_b) - (off_b - def_a)
        # shape: (n_draws,)
        margin_draws = (
            (off_draws[:, idx_a] - deff_draws[:, idx_b]) -
            (off_draws[:, idx_b] - deff_draws[:, idx_a])
        )

        # P(team_a wins) per draw: margin ~ Normal(diff, sigma*sqrt(2))
        # => P(score_a > score_b) = norm.cdf(diff / (sigma * sqrt(2)))
        denom = np.array(sigma_draws) * np.sqrt(2)
        p_a_win_draws = norm.cdf(np.array(margin_draws) / denom)  # (n_draws,)

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