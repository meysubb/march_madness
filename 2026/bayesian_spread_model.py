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

from random import seed

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

#' To go Off/Def, we can't use spread anymore, have to use the actual score. 
def off_def_model(home_idx, away_idx, conf_idx, neutral, weights, T, C, 
                  score_home=None, 
                  score_away=None,):
    
    mu_intercept = numpyro.sample("mu_intercept", dist.Normal(70.0, 10.0))
    alpha        = numpyro.sample("alpha", dist.Normal(3.0, 1.0))

    # Conference Priors
    sigma_off_conf = numpyro.sample("sigma_off_conf", dist.HalfNormal(5.0))
    sigma_def_conf = numpyro.sample("sigma_def_conf", dist.HalfNormal(5.0))

    with numpyro.plate("conferences", C):
        mu_off_conf = numpyro.sample("mu_off_conf", dist.Normal(0.0, sigma_off_conf))
        mu_def_conf = numpyro.sample("mu_def_conf", dist.Normal(0.0, sigma_def_conf))

    sigma_off = numpyro.sample("sigma_off", dist.HalfNormal(5.0))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(5.0))

    # LKJ prior on within-conference off/def correlation 
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(dimension=2, concentration=2.0))
    D = jnp.diag(jnp.array([sigma_off, sigma_def]))
    L_cov = D @ L_corr   # (2, 2)

    with numpyro.plate("teams", T):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([2]).to_event(1))

    conf_means = jnp.stack([mu_off_conf[conf_idx],
                            mu_def_conf[conf_idx]], axis=1)  # (T, 2)

    team_effects = conf_means + z @ L_cov.T  # (T, 2)

    off = numpyro.deterministic("off", team_effects[:, 0])  # (T,)
    defense = numpyro.deterministic("def", team_effects[:, 1])  # (T,)

    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))

    home_court = alpha * (1 - neutral)
    mu_home = mu_intercept + off[home_idx] - defense[away_idx] + home_court
    mu_away = mu_intercept + off[away_idx] - defense[home_idx] - home_court

    weighted_sigma = sigma / jnp.sqrt(weights)
    numpyro.sample("score_home", dist.Normal(mu_home, weighted_sigma), obs=score_home)
    numpyro.sample("score_away", dist.Normal(mu_away, weighted_sigma), obs=score_away)

def fit(data, model_func, num_warmup=1000, num_samples=1000, num_chains=4, seed=42):
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
    kernel = NUTS(model_func)
    mcmc = MCMC(kernel, num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains)

    rng_key = jax.random.PRNGKey(seed) 
    mcmc.run(jax.random.PRNGKey(seed), **data)
    
    return mcmc, mcmc.get_samples()


def prepare_data(df):
    """
    Prepare a DataFrame for the model.

    Expected columns: home, road, line, home_win, date, neutral
    """
    df = df.copy()
    df = df.dropna(subset=["line"])
    df = df.dropna(subset=["hscore", "rscore"])
    df['home_win'] = (df['hscore'] > df['rscore']).astype(int)
    dates = pd.to_datetime(df["date"])
    days = (dates - dates.min()).dt.days + 1

    weights = (days ** 4) / (days ** 4).max()
    weights = weights.clip(lower=0.01)

    # Can use the sklearn LabelEncoder.
    unique_teams = sorted(set(df["home"].str.upper()) | set(df["road"].str.upper()))
    team_to_idx = {t: i for i, t in enumerate(unique_teams)}

    unique_confs = sorted(set(df["home_conf"].str.upper()) | set(df["road_conf"].str.upper()))
    conf_to_idx = {c: i for i, c in enumerate(unique_confs)}

    team_conf_map = {}
    for _, row in df.iterrows():
        team_conf_map[row["home"].upper()] = row["home_conf"].upper()
        team_conf_map[row["road"].upper()] = row["road_conf"].upper()

    conf_idx = jnp.array(
        [conf_to_idx[team_conf_map[t]] for t in unique_teams],
        dtype=jnp.int32
    )

    home_idx = df["home"].str.upper().map(team_to_idx).values
    away_idx = df["road"].str.upper().map(team_to_idx).values

    return {
        "home_idx":  jnp.array(home_idx, dtype=jnp.int32),
        "away_idx":  jnp.array(away_idx, dtype=jnp.int32),
        "neutral":   jnp.array(df["neutral"].values, dtype=float),
        "weights":   jnp.array(weights.values, dtype=float),
        "line":      jnp.array(df["line"].values, dtype=float),
        "score_home": jnp.array(pd.to_numeric(df["hscore"], errors="coerce").values, dtype=float),
        "score_away": jnp.array(pd.to_numeric(df["rscore"], errors="coerce").values, dtype=float),
        "home_win":  jnp.array(df["home_win"].values, dtype=jnp.int32),
        "conf_idx": conf_idx,        
        "T":         len(unique_teams),        
        "C":        len(unique_confs),
        #"teams":     unique_teams,
        #"team_to_idx": team_to_idx,
    }, unique_teams, unique_confs, team_to_idx, conf_to_idx

def extract_team_ratings(samples, teams):
    beta = samples["beta"]   # (draws, T)
    
    return pd.DataFrame({
        "team": teams,
        "mean": np.array(jnp.mean(beta, axis=0)),
        "lower_95": np.array(jnp.percentile(beta, 2.5, axis=0)),
        "upper_95": np.array(jnp.percentile(beta, 97.5, axis=0)),
    }).sort_values("mean", ascending=False).reset_index(drop=True)

