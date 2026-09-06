"""V2: explicit game markets, causal tendencies, and a tested script-mixture model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES
from .game_features import MARKET_FEATURES, TENDENCY_FEATURES
from .model import probabilities, point_metrics, crps

VERSION = 'attempts-v2.0'
CANDIDATES = ('v1_ridge', 'market_ridge', 'tendency_ridge', 'tendency_boosting', 'script_opportunity')
STATE_FEATURES = ['expected_margin', 'game_total', 'home']
STATES = ('leading', 'close', 'trailing')

def ridge(alpha=150.):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

class ScriptExposure:
    """Forecast state among counted plays, not the final winner.

    Close-game plays in the final 120 seconds are absent from the neutral
    tendency denominator, whereas substantial leads/deficits include that
    period. Shares therefore describe this counted subset, not every play or
    time spent leading. The opportunity candidate extrapolates these shares
    across projected plays; this is a documented approximation, not a complete
    game-script simulation.
    """
    def fit(self, df):
        counts = df[['lead_plays', 'neutral_plays', 'trail_plays']].fillna(0).to_numpy()
        shares = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
        usable = counts.sum(axis=1) > 0
        x = np.repeat(df.loc[usable, STATE_FEATURES].to_numpy(), 3, axis=0)
        y = np.tile(np.arange(3), usable.sum())
        weights = shares[usable].ravel()
        good = weights > 0
        self.estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=1000, random_state=730))
        self.estimator.fit(x[good], y[good], logisticregression__sample_weight=weights[good])
        return self
    def predict(self, df):
        return self.estimator.predict_proba(df[STATE_FEATURES].to_numpy())

@dataclass
class Candidate:
    name: str
    estimator: object
    columns: list[str]
    exposure: object = None
    plays_model: object = None
    def physical(self, df):
        shares = self.exposure.predict(df)
        rates = df[['lead_pass12', 'neutral_pass12', 'trail_pass12']].to_numpy()
        plays = self.plays_model.predict(df[MARKET_FEATURES]).clip(40, 85)
        return plays * np.sum(shares * rates, axis=1) * df.attempts_per_dropback12.to_numpy().clip(.70, .99)
    def predict(self, df):
        if self.name == 'script_opportunity':
            x = df[['qb_mean12', 'qb_carries5', 'home', 'rest']].copy()
            x['physical_attempts'] = self.physical(df)
            return self.estimator.predict(x).clip(8, 60)
        return self.estimator.predict(df[self.columns]).clip(12, 55)

def fit_candidate(name, df):
    if name == 'script_opportunity':
        exposure = ScriptExposure().fit(df)
        plays = ridge().fit(df[MARKET_FEATURES], df.plays)
        candidate = Candidate(name, None, [], exposure, plays)
        x = df[['qb_mean12', 'qb_carries5', 'home', 'rest']].copy()
        x['physical_attempts'] = candidate.physical(df)
        candidate.estimator = ridge(50.).fit(x, df.attempts)
        return candidate
    columns = FEATURES if name == 'v1_ridge' else MARKET_FEATURES if name == 'market_ridge' else TENDENCY_FEATURES
    obj = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=7, learning_rate=.035, l2_regularization=20., min_samples_leaf=60, random_state=730, early_stopping=False) if name == 'tendency_boosting' else ridge()
    obj.fit(df[columns], df.attempts)
    return Candidate(name, obj, columns)

def distribution_scores(df, residuals):
    y, pred = df.attempts.to_numpy(), df.prediction.to_numpy()
    metrics = point_metrics(y, pred)
    # Chunked vectorized probability distribution avoids one slow Python call per game.
    scores, hits, probabilities_list, truths = [], [], [], []
    thresholds = np.arange(0, 90)
    for start in range(0, len(df), 24):
        mu = pred[start:start+24]
        actual = y[start:start+24]
        cdf = norm.cdf(thresholds[None, :, None] + .5 - mu[:, None, None] - residuals[None, None, :]).mean(axis=2)
        scores.extend(np.sum((cdf - (actual[:, None] <= thresholds[None, :])) ** 2, axis=1))
        low, high = mu + np.quantile(residuals, .1), mu + np.quantile(residuals, .9)
        hits.extend((actual >= low) & (actual <= high))
        p = 1 - cdf[:, [24, 29, 34, 39, 44]]
        probabilities_list.extend(p.ravel())
        truths.extend((actual[:, None] > np.array([24.5, 29.5, 34.5, 39.5, 44.5])).ravel())
    p = np.clip(probabilities_list, 1e-7, 1-1e-7)
    target = np.array(truths)
    metrics.update(crps=float(np.mean(scores)), coverage_80=float(np.mean(hits)), fixed_threshold_brier=float(np.mean((p-target)**2)), fixed_threshold_log_loss=float(-np.mean(target*np.log(p)+(1-target)*np.log(1-p))))
    return metrics, np.array(scores)

def choose_candidate(development):
    """Additional complexity must improve CRPS by at least 0.5% over market ridge."""
    simple = 'market_ridge'
    richer = min(('tendency_ridge', 'tendency_boosting', 'script_opportunity'), key=lambda k: development[k]['crps'])
    return richer if development[richer]['crps'] < development[simple]['crps'] * .995 else simple

def train_validate_v2(df, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    eligible = df[np.isfinite(df.expected_margin) & np.isfinite(df.game_total) & (df.tendency_games >= 1)].copy()
    excluded = len(df)-len(eligible)
    all_oof, summaries, year_metrics = [], {}, {}
    for name in CANDIDATES:
        parts = []
        for year in range(2019, 2026):
            tr = df[df.season < year] if name == 'v1_ridge' else eligible[eligible.season < year]
            te = eligible[eligible.season == year].copy()
            if te.empty:
                continue
            obj = fit_candidate(name, tr)
            te['prediction'] = obj.predict(te)
            te['residual'] = te.attempts-te.prediction
            te['candidate'] = name
            parts.append(te)
        oof = pd.concat(parts, ignore_index=True)
        year_metrics[name] = {}
        metrics_list = []
        for year in range(2022, 2025):
            te = oof[oof.season == year]
            residuals = oof[oof.season < year].residual.to_numpy()
            m, _ = distribution_scores(te, residuals)
            year_metrics[name][str(year)] = m
            metrics_list.append(m)
        total = sum(m['n'] for m in metrics_list)
        summaries[name] = {k:float(sum(m[k]*m['n'] for m in metrics_list)/total) for k in ('mae','rmse','crps','fixed_threshold_brier','fixed_threshold_log_loss','coverage_80')}
        summaries[name]['rmse'] = float(np.sqrt(sum(m['rmse']**2*m['n'] for m in metrics_list)/total))
        summaries[name]['n'] = total
        all_oof.append(oof)
        print(f"{name}: development CRPS={summaries[name]['crps']:.4f}, MAE={summaries[name]['mae']:.4f}", flush=True)
    chosen = choose_candidate(summaries)
    oof = pd.concat(all_oof, ignore_index=True)
    benchmark, individual = {}, {}
    for name in CANDIDATES:
        one = oof[oof.candidate == name]
        test = one[one.season == 2025]
        m, per_game = distribution_scores(test, one[one.season < 2025].residual.to_numpy())
        benchmark[name] = m
        individual[name] = test.assign(crps=per_game)
    selected = oof[oof.candidate == chosen]
    current, baseline = individual[chosen], individual['v1_ridge']
    paired = current[['game_id','player_id','attempts','prediction','crps']].merge(baseline[['game_id','player_id','prediction','crps']], on=['game_id','player_id'], suffixes=('_v2','_v1'), validate='one_to_one')
    paired['mae_difference'] = abs(paired.attempts-paired.prediction_v2)-abs(paired.attempts-paired.prediction_v1)
    paired['crps_difference'] = paired.crps_v2-paired.crps_v1
    grouped = paired.groupby('game_id')[['mae_difference','crps_difference']].mean().to_numpy()
    rng = np.random.default_rng(730)
    boot = grouped[rng.integers(0,len(grouped),(2000,len(grouped)))].mean(axis=1)
    cis = {k:np.quantile(boot[:,i],[.025,.975]).tolist() for i,k in enumerate(('mae_difference','crps_difference'))}
    residuals = selected[selected.season < 2025].residual.to_numpy()
    segments = {}
    for label, mask in [('favored_7_plus',current.expected_margin >= 7),('close_spread',current.expected_margin.abs()<7),('underdog_7_plus',current.expected_margin<=-7),('week1',current.week==1)]:
        if mask.any():segments[label] = distribution_scores(current[mask], residuals)[0]
    model = fit_candidate(chosen, eligible)
    exposure = ScriptExposure().fit(eligible)
    results = {'version':VERSION,'chosen':chosen,'selection_rule':'2022–2024 expanding-season CRPS; complex models require 0.5% improvement over explicit-market ridge. 2025 is an exposed benchmark, not a new untouched test.', 'historical_odds_timing':'nflverse closing-reference game spread/total; morning archive unavailable','candidates':{k:{'development_2022_2024':v} for k,v in summaries.items()},'development_by_year':year_metrics,'benchmark_2025':benchmark,'holdout_2025':benchmark[chosen],'baseline_2025':benchmark['v1_ridge'],'baseline_label':'Frozen V1 ridge on matched rows','paired_game_bootstrap_95ci':cis,'segments_2025':segments,'training_rows':len(eligible),'excluded_missing_history_or_game_market':excluded,'historical_market_roi':None,'market_edge_validated':False,'blend_weight_learned':False,'feature_names':model.columns if model.columns else TENDENCY_FEATURES,'xpass_used_as_predictor':False,'limitations':['2025 benchmark already exposed','historical spreads are closing references, not archived morning quotes','state probabilities describe representative eligible plays, not exact time-leading probabilities','market blending policy is a prospective hypothesis, not historical prop-odds optimized']}
    artifact = {'version':VERSION,'name':chosen,'estimator':model,'exposure':exposure,'residuals':selected.residual.to_numpy(),'metrics':results}
    with (output/'model.pkl').open('wb') as f:
        pickle.dump(artifact, f)
    (output/'validation.json').write_text(json.dumps(results,indent=2))
    oof[['game_id','season','week','player_id','player','attempts','candidate','prediction','residual','expected_margin','game_total']].to_csv(output/'oof_predictions.csv',index=False)
    paired.to_csv(output/'paired_v1_v2_2025.csv',index=False)
    return artifact


def scenario_analysis(artifact, frame):
    results = []
    for delta in (-7., 0., 7.):
        f = frame.copy()
        f['expected_margin'] = delta
        f['margin_abs'] = abs(delta)
        f['margin_x_team_rate'] = delta*(f.team_rate5-.5)
        f['margin_x_qb_carries'] = delta*f.qb_carries5
        f['state_pass_spread_interaction'] = delta*(f.trail_pass12-f.lead_pass12)
        results.append(artifact['estimator'].predict(f).tolist())
    return results
