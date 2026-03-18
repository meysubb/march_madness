"""
The original R model has two stages:
  1. Weighted OLS: line ~ 1 + game_matrix + neutral
     Estimates team strength ratings from point spreads.
  2. Logistic regression: home_win ~ 0 + line
     Converts any spread to a win probability.

The day-based weights from R become likelihood weights in the Bayesian
model -- same idea, expressed naturally as each game's contribution
to the log-likelihood.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import pandas as pd


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def basic_spread_model(home_idx, away_idx, neutral, weights, T, line, home_win):
    # hack the model and do intercept * (1-neutral) instead of separate param
    intercept = numpyro.sample("intercept", dist.Normal(3.0, 1.0))

    # team partial pooling
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(5.0)) 
    with numpyro.plate("teams", T):
        beta = numpyro.sample("beta", dist.Normal(0.0, sigma_team))

    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))

    mu = intercept * (1 - neutral) + beta[home_idx] - beta[away_idx]
    numpyro.deterministic("mu", mu)

    # add some weighting?
    weighted_sigma = sigma / jnp.sqrt(weights)
    numpyro.sample("line", dist.Normal(mu, weighted_sigma), obs=line)
    # numpyro.sample("line", dist.Normal(mu, sigma), obs=line)

def fit(data, num_warmup=1000, num_samples=1000, num_chains=4, seed=42):
    """
    Fit the model with NUTS.

    Parameters
    ----------
    data : dict from prepare_data()

    Returns
    -------
    mcmc    : fitted MCMC object
    samples : dict of posterior samples
    """
    kernel = NUTS(basic_spread_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains)

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        home_idx  = data["home_idx"],
        away_idx  = data["away_idx"],
        neutral   = data["neutral"],
        weights   = data["weights"],
        T         = data["T"],
        line      = data["line"],
        home_win  = data["home_win"],
    )

    return mcmc, mcmc.get_samples()


def prepare_data(df):
    """
    Prepare a DataFrame for the model.

    Expected columns: home, road, line, home_win, date, neutral
    """
    df['home_win'] = (df['hscore'] > df['rscore']).astype(int)
    dates = pd.to_datetime(df["date"])
    days = (dates - dates.min()).dt.days + 1
    weights = (days ** 4) / (days ** 4).max()

    # Can use the sklearn LabelEncoder.
    unique_teams = sorted(set(df["home"].str.upper()) | set(df["road"].str.upper()))
    team_to_idx = {t: i for i, t in enumerate(unique_teams)}

    home_idx = df["home"].str.upper().map(team_to_idx).values
    away_idx = df["road"].str.upper().map(team_to_idx).values

    return {
        "home_idx":  jnp.array(home_idx, dtype=jnp.int32),
        "away_idx":  jnp.array(away_idx, dtype=jnp.int32),
        "neutral":   jnp.array(df["neutral"].values, dtype=float),
        "weights":   jnp.array(weights.values, dtype=float),
        "line":      jnp.array(df["line"].values, dtype=float),
        "home_win":  jnp.array(df["home_win"].values, dtype=jnp.int32),
        "T":         len(unique_teams),
        "teams":     unique_teams,
        "team_to_idx": team_to_idx,
    }, unique_teams

def extract_team_ratings(samples, teams):
    beta = samples["beta"]   # (draws, T)
    
    return pd.DataFrame({
        "team": teams,
        "mean": np.array(jnp.mean(beta, axis=0)),
        "lower_95": np.array(jnp.percentile(beta, 2.5, axis=0)),
        "upper_95": np.array(jnp.percentile(beta, 97.5, axis=0)),
    }).sort_values("mean", ascending=False).reset_index(drop=True)

