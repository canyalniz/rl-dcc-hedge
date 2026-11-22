from typing import Optional, Tuple, Type
from rl_dcc_hedge.environments import HedgeEnv
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd

class BaselineStrategy(ABC):
	"""Base class for baseline strategies."""

	@abstractmethod
	def run(self, env: Type[HedgeEnv]) -> pd.DataFrame:
		"""Simulate the strategy on a HedgeEnv implementation"""
		pass

class TheoreticalMinVar(BaselineStrategy):
	"""Theoretical minimum-variance hedge ratio baseline.

	At every step the strategy observes the conditional standard deviations of
	the anchor and secondary log returns (``_condstd``) and their conditional
	correlation (``_condcorr``) from the environment's feature block. The
	environment's equity evolves with simple returns R = exp(r) - 1, so the
	log-return moments are first converted to the simple-return mean
	``_simple_mean`` and covariance ``_simple_cov`` (lognormal moment
	matching, see _observe()), and the strategy requests the theoretical
	minimum-variance hedge ratio

		alpha = Cov(R_anchor, R_secondary) / Var(R_secondary)

	as its action. Under the environment's alpha = -E_s / E_a convention this
	alpha minimizes the conditional variance of the hedged portfolio equity.

	Parameters
	----------
	mu : sequence of float
		Length-2 ``(anchor, secondary)`` constant per-step expected log
		returns used by the log -> simple moment conversion. The
		minimum-variance alpha depends on them only through the factor
		exp(mu_anchor - mu_secondary), so the zero-drift default is accurate
		to O(mu_anchor - mu_secondary).
	"""

	# canonical (DataHandler) feature layout: per-tic tags tic-major/tag-minor
	# (anchor = tic 0, secondary = tic 1), followed by the standalone tags
	DEFAULT_PER_TIC_FEATURE_TAGS = ["innov", "condstd"]
	DEFAULT_STANDALONE_FEATURE_TAGS = ["condcorr"]

	def __init__(self, mu=[0.001440, 0.001407]):
		"""Validates and stores the constant log-return means."""
		mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
		if mu_vec.shape != (2,):
			raise ValueError(
				f"{type(self).__name__} mu must have length 2 [anchor, secondary], got {mu!r}"
			)
		if not np.all(np.isfinite(mu_vec)):
			raise ValueError(f"{type(self).__name__} mu must be finite, got {mu!r}")
		self._mu = mu_vec
		# conditional log-return moments observed at the current step, refreshed by _observe()
		self._condstd: Optional[np.ndarray] = None # (anchor, secondary)
		self._condcorr: Optional[float] = None
		# the same moments converted to simple returns R = exp(r) - 1
		self._simple_mean: Optional[np.ndarray] = None # E[R], (anchor, secondary)
		self._simple_cov: Optional[np.ndarray] = None # Cov(R), 2x2
		# pre-trade hedge ratio alpha = -E_s / E_a of the book at the current step
		self._current_alpha: Optional[float] = None

	def __str__(self):
		return "MV"

	def _moment_indices(self, env: HedgeEnv) -> Tuple[int, int, int]:
		"""Locate the (condstd_anchor, condstd_secondary, condcorr) feature columns.

		MT_HedgeEnv advertises its column tags through
		``_per_tic_features_column_tags`` / ``_standalone_features_column_tags``;
		other environments are assumed to follow the canonical DataHandler layout.
		"""
		per_tic_tags = getattr(env, "_per_tic_features_column_tags", self.DEFAULT_PER_TIC_FEATURE_TAGS)
		standalone_tags = getattr(env, "_standalone_features_column_tags", self.DEFAULT_STANDALONE_FEATURE_TAGS)
		try:
			condstd_offset = per_tic_tags.index("condstd")
			condcorr_offset = standalone_tags.index("condcorr")
		except ValueError as err:
			raise ValueError(
				f"{type(self).__name__} needs a 'condstd' per-tic feature and a 'condcorr' "
				f"standalone feature; got per-tic tags {per_tic_tags!r} and standalone tags {standalone_tags!r}"
			) from err
		indices = (
			condstd_offset, # anchor condstd (tic 0)
			len(per_tic_tags) + condstd_offset, # secondary condstd (tic 1)
			env.num_tics * len(per_tic_tags) + condcorr_offset, # standalone condcorr
		)
		for idx, name in zip(indices, ("anchor condstd", "secondary condstd", "condcorr")):
			if not 0 <= idx < env.num_features:
				raise ValueError(
					f"{type(self).__name__}: the {name} column index {idx} falls outside the "
					f"environment's feature block of {env.num_features} feature(s); "
					f"check the feature column layout"
				)
		return indices

	def _observe(self, moments: np.ndarray) -> None:
		"""Records the conditional moments observed at the current step.

		``moments`` is the (condstd_anchor, condstd_secondary, condcorr) triple
		read from the current feature row (log-return moments).

		The log returns r ~ (mu, Sigma) are converted to the moments of the
		simple returns R = exp(r) - 1 that drive the environment's equity, via
		the lognormal identities with m = mu + diag(Sigma) / 2:

			E[R_i]         = exp(m_i) - 1
			Cov(R_i, R_j)  = exp(m_i + m_j) * (exp(Sigma_ij) - 1)

		(exact for Gaussian log returns; a moment-matching approximation for
		the fitted t innovations, whose exponential moments are undefined).
		"""
		self._condstd = np.asarray(moments[:2], dtype=np.float64)
		self._condcorr = float(moments[2])
		if not (np.all(np.isfinite(moments)) and np.all(self._condstd > 0)):
			raise ValueError(
				f"{type(self).__name__} observed invalid conditional moments: "
				f"condstd={self._condstd!r}, condcorr={self._condcorr!r}"
			)
		log_cov = np.outer(self._condstd, self._condstd)
		log_cov[0, 1] = log_cov[1, 0] = self._condcorr * self._condstd[0] * self._condstd[1]
		m = self._mu + 0.5 * np.diag(log_cov)
		self._simple_mean = np.expm1(m)
		self._simple_cov = np.exp(m[:, None] + m[None, :]) * np.expm1(log_cov)

	def _hedge_ratio(self) -> float:
		"""Theoretical minimum-variance hedge ratio for the observed moments."""
		return self._simple_cov[0, 1] / self._simple_cov[1, 1]

	def run(self, env: Type[HedgeEnv]) -> pd.DataFrame:
		"""Simulates a full test episode on ``env`` and returns its trajectory df."""
		if not isinstance(env, HedgeEnv):
			raise ValueError(f"{type(self).__name__} expects a single HedgeEnv instance, got {type(env)}")
		anchor_condstd_idx, secondary_condstd_idx, condcorr_idx = self._moment_indices(env)

		obs, _ = env.reset()
		terminated = truncated = False
		while not (terminated or truncated):
			# observations are [portfolio state (NUM_OBS_VAR vars) | feature block];
			# _moment_indices() addresses the feature block only, so slice it off first
			feature_block = obs[len(obs) - env.num_features:]
			self._observe(feature_block[[anchor_condstd_idx, secondary_condstd_idx, condcorr_idx]])
			# exact pre-trade alpha (the observation only carries it rescaled, in float_dtype)
			self._current_alpha = float(env._alpha)
			action = np.clip(self._hedge_ratio(), env.action_space.low[0], env.action_space.high[0])
			obs, _, terminated, truncated, _ = env.step(np.asarray([action], dtype=env.action_space.dtype))

		return env.get_trajectory_df()

class CostAwareMeanVarHedge(TheoreticalMinVar):
	"""Cost-aware mean-variance hedge ratio baseline.

	Implements the maximum risk-adjusted-return hedge ratio under proportional
	transaction costs on the rebalancing turnover derived in
	``docs/optimal_hedge_ratio.md`` (section 3.1): with constant per-step
	expected log returns ``mu = (mu_anchor, mu_secondary)``, risk-aversion
	parameter ``lam`` (lambda), the environment's proportional
	transaction-cost rate ``kappa = env.transaction_cost_rate`` and the
	book's pre-trade hedge ratio ``alpha_0``, it requests the alpha maximizing

		mu_p(alpha) - (lam / 2) * Var_p(alpha) - kappa * |alpha - alpha_0|

	under the environment's alpha = -E_s / E_a convention (the environment
	charges kappa * |Delta E_s| = kappa * E_a * |alpha - alpha_0|), where (per
	step, with the simple-return moments derived from the observed
	log-return moments, see TheoreticalMinVar._observe())

		mu_p(alpha)  = E[R_a] - alpha * E[R_s]
		Var_p(alpha) = Var(R_a) + alpha^2 * Var(R_s) - 2 * alpha * Cov(R_a, R_s)

	The closed form (docs/optimal_hedge_ratio.md, section 3.1) is a no-trade
	band around the frictionless target (section 2 of the doc)

		target     = (lam * Cov(R_a, R_s) - E[R_s]) / (lam * Var(R_s))
		half_width = kappa / (lam * Var(R_s))
		alpha*     = clip(alpha_0, target - half_width, target + half_width)

	i.e. the book is left untouched while it lies within ``half_width`` of the
	target, and is otherwise rebalanced only to the nearest band edge.
	``lam -> infinity`` recovers TheoreticalMinVar's minimum-variance alpha
	(section 1); small ``lam`` strengthens the return-seeking tilt
	``-E[R_s] / (lam * Var(R_s))`` and widens the band.

	``mu_anchor`` cancels from the optimizer apart from the O(mu) scaling of
	the simple-return moments: the environment never trades the anchor leg,
	so its expected return only shifts the utility level, not the argmax over
	alpha.

	Deliberately NOT registered in HEDGE_BASELINE_STRATEGIES: registry
	consumers re-construct strategies with no arguments
	(``type(HEDGE_BASELINE_STRATEGIES[name])()`` in run_experiments.py), while
	this baseline is parameterized at construction; instantiate it directly
	(the same deliberate pattern as UnhedgedAnchor).

	Parameters
	----------
	mu : sequence of float
		Length-2 ``(anchor, secondary)`` constant per-step expected log
		returns, in the same per-step units as the environment's conditional
		std / correlation features (e.g. ``dist_params['mu']`` of the fitted
		model).
	lam : float
		Finite positive risk-aversion parameter lambda: how much to minimize
		risk (large ``lam``) vs. maximize expected return (small ``lam``).
	"""

	def __init__(self, lam, mu=[0.001440, 0.001407]): # updated to SPY_XLU
		"""Validates and stores the constant means and the risk aversion."""
		super().__init__(mu=mu)
		lam = float(lam)
		if not (math.isfinite(lam) and lam > 0.0):
			raise ValueError(
				f"{type(self).__name__} lam must be a finite positive risk-aversion "
				f"parameter, got {lam!r}"
			)
		self._lam = lam
		# proportional transaction-cost rate kappa, read from the environment
		# at the start of run()
		self._transaction_cost_rate: Optional[float] = None

	def __str__(self):
		return f"MRR-C_{self._lam}"

	def _hedge_ratio(self) -> float:
		"""Cost-aware risk-adjusted-return hedge ratio for the observed moments."""
		if self._transaction_cost_rate is None:
			raise RuntimeError(
				f"{type(self).__name__}._hedge_ratio() needs the transaction-cost rate; "
				"call run(env) first (kappa is read from the environment)"
			)
		var_s = self._simple_cov[1, 1]
		# frictionless mean-variance-utility target (lam * Cov - E[R_s]) / (lam * Var)
		target = (self._lam * self._simple_cov[0, 1] - self._simple_mean[1]) / (self._lam * var_s)
		half_width = self._transaction_cost_rate / (self._lam * var_s)
		# no-trade band: costs are charged on |alpha - alpha_0| only
		return min(max(self._current_alpha, target - half_width), target + half_width)

	def run(self, env: Type[HedgeEnv]) -> pd.DataFrame:
		"""Reads kappa from ``env`` and simulates the episode on it."""
		if not isinstance(env, HedgeEnv):
			raise ValueError(f"{type(self).__name__} expects a single HedgeEnv instance, got {type(env)}")
		kappa = getattr(env, "transaction_cost_rate", None)
		if kappa is None:
			raise ValueError(
				f"{type(self).__name__} needs env.transaction_cost_rate (the proportional "
				f"transaction-cost rate), got {type(env).__name__} without the attribute"
			)
		kappa = float(kappa)
		if not (math.isfinite(kappa) and 0.0 <= kappa < 1.0):
			raise ValueError(
				f"{type(self).__name__} needs a transaction-cost rate in [0, 1), got {kappa!r}"
			)
		self._transaction_cost_rate = kappa
		return super().run(env)

class UnhedgedAnchor(BaselineStrategy):
	"""The unhedged no-trade book: floor(E0/P0_anchor) anchor units + residual cash.

	Requests ``action = [0.0]`` at every step, so the environment never trades
	(target secondary equity 0, zero transaction cost) and the portfolio is
	exactly the environment's initial all-anchor book with the residual cash
	accruing ``annual_financing_rate`` over the episode. Realized through the
	same environment mechanics/frictions/trajectory format as the agents and
	the other baselines, it serves as the unhedged reference trajectory of the
	hedge-effectiveness analysis in ``run_experiments.py``.

	Deliberately NOT registered in HEDGE_BASELINE_STRATEGIES so that
	``run_rl.py --run_baselines`` behavior stays unchanged; ``run_experiments.py``
	imports and instantiates it directly.
	"""

	def __str__(self):
		return "unhedged_anchor"

	def run(self, env: Type[HedgeEnv]) -> pd.DataFrame:
		"""Simulates a full episode without ever trading and returns the trajectory df."""
		if not isinstance(env, HedgeEnv):
			raise ValueError(f"{type(self).__name__} expects a single HedgeEnv instance, got {type(env)}")
		obs, _ = env.reset()
		terminated = truncated = False
		while not (terminated or truncated):
			obs, _, terminated, truncated, _ = env.step(np.asarray([0.0], dtype=env.action_space.dtype))

		return env.get_trajectory_df()

class StaticOLSHedge(BaselineStrategy):
	"""Static OLS minimum-variance hedge ratio baseline.

	The hedge ratio is the OLS slope of the anchor's simple return on the
	secondary's simple return over a fitting sample (log returns are first
	converted, y_t = exp(r^A_t) - 1, x_t = exp(r^B_t) - 1):

		y_t = a + h * x_t + e_t,   h_hat = Cov(y, x) / Var(x)

	The intercept is kept, so h_hat minimizes the in-sample variance of the
	hedged return y_t - h * x_t. Under the environment's alpha = -E_s / E_a
	convention h_hat is requested directly as the action.

	The position is set to h_hat at the first step and never adjusted
	afterwards: every later step requests the book's current (drifted)
	alpha, so the environment does not trade except where its margin
	constraints force it to.

	Parameters
	----------
	fitting_log_returns : array-like
		``(T, 2)`` per-step log returns ``(anchor, secondary)`` of the fitting
		sample, in the same per-step units as the test environment.
	"""

	# relative nudge applied to the hold action so the environment's
	# int (toward zero) position rounding, after the float32 action cast,
	# reproduces the current secondary position instead of trading one unit
	HOLD_NUDGE = 1e-6

	def __init__(self, fitting_log_returns):
		"""Validates the fitting sample and estimates the OLS hedge ratio."""
		self._hedge_ratio = self.ols_hedge_ratio(fitting_log_returns)

	def __str__(self):
		return "OLS-static"

	@staticmethod
	def ols_hedge_ratio(log_returns) -> float:
		"""OLS slope (with intercept) of anchor on secondary simple returns."""
		log_returns = np.asarray(log_returns, dtype=np.float64)
		if log_returns.ndim != 2 or log_returns.shape[1] != 2:
			raise ValueError(
				f"fitting log returns must have shape (T, 2) [anchor, secondary], got {log_returns.shape}"
			)
		if log_returns.shape[0] < 3:
			raise ValueError(f"fitting sample needs at least 3 observations, got {log_returns.shape[0]}")
		if not np.all(np.isfinite(log_returns)):
			raise ValueError("fitting log returns must be finite")
		simple_returns = np.expm1(log_returns)
		y, x = simple_returns[:, 0], simple_returns[:, 1]
		x_dev = x - x.mean()
		sxx = x_dev @ x_dev
		if not sxx > 0.0:
			raise ValueError("secondary simple returns have zero variance over the fitting sample")
		return float(x_dev @ (y - y.mean()) / sxx)

	def _rebalance(self, step_idx: int) -> bool:
		"""Whether the hedge ratio is enforced at ``step_idx``."""
		return step_idx == 0

	def _target_ratio(self, env: HedgeEnv, step_idx: int) -> float:
		"""Hedge ratio requested at a rebalance step ``step_idx`` of ``env``'s episode."""
		return self._hedge_ratio

	def run(self, env: Type[HedgeEnv]) -> pd.DataFrame:
		"""Simulates a full test episode on ``env`` and returns its trajectory df."""
		if not isinstance(env, HedgeEnv):
			raise ValueError(f"{type(self).__name__} expects a single HedgeEnv instance, got {type(env)}")
		obs, _ = env.reset()
		terminated = truncated = False
		step_idx = 0
		while not (terminated or truncated):
			if self._rebalance(step_idx):
				action = self._target_ratio(env, step_idx)
			else:
				# hold: request the pre-trade alpha (exact, the observation only carries it rescaled)
				action = float(env._alpha) * (1.0 + self.HOLD_NUDGE)
			action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
			obs, _, terminated, truncated, _ = env.step(np.asarray([action], dtype=env.action_space.dtype))
			step_idx += 1

		return env.get_trajectory_df()

class PeriodicOLSHedge(StaticOLSHedge):
	"""Periodically rebalanced OLS minimum-variance hedge ratio baseline.

	Uses the same fixed OLS hedge ratio as StaticOLSHedge (estimated once on
	the fitting sample), but enforces it every ``rebalance_period`` steps
	(steps 0, P, 2P, ...); in between, the book is held and its alpha drifts
	with prices.

	Parameters
	----------
	fitting_log_returns : array-like
		``(T, 2)`` per-step log returns ``(anchor, secondary)`` of the fitting
		sample.
	rebalance_period : int
		Positive number of environment steps between rebalances; 1 enforces
		the hedge ratio at every step.
	"""

	def __init__(self, fitting_log_returns, rebalance_period: int):
		"""Validates the rebalance period and estimates the OLS hedge ratio."""
		super().__init__(fitting_log_returns)
		if isinstance(rebalance_period, bool) or int(rebalance_period) != rebalance_period or rebalance_period < 1:
			raise ValueError(
				f"{type(self).__name__} rebalance_period must be a positive integer, got {rebalance_period!r}"
			)
		self._rebalance_period = int(rebalance_period)

	def __str__(self):
		return f"OLS-P{self._rebalance_period}"

	def _rebalance(self, step_idx: int) -> bool:
		return step_idx % self._rebalance_period == 0

class RollingOLSHedge(PeriodicOLSHedge):
	"""Rolling-window OLS minimum-variance hedge ratio baseline.

	At every rebalance step t (steps 0, P, 2P, ...) the hedge ratio is
	re-estimated by OLS (StaticOLSHedge.ols_hedge_ratio) on the most recent
	``window`` per-step log returns available before trading at prices[t]:
	the t returns realized so far in the episode, log(prices[1:t+1] / prices[:t]),
	preceded by the tail of the fitting sample while t < window. The return
	of step t itself is never used. Both environment types start each
	trajectory at the last modeling-period price, so the fitting sample and
	the episode's realized returns form one continuous series.

	The window is rebuilt from ``env.prices`` at each rebalance, so the
	instance keeps no per-episode state and can be reused across trajectories.

	Parameters
	----------
	fitting_log_returns : array-like
		``(T, 2)`` per-step log returns ``(anchor, secondary)`` of the fitting
		sample; its last ``window`` rows warm up the window.
	window : int
		Number of per-step returns in the OLS window, in [3, T].
	rebalance_period : int
		Positive number of environment steps between re-estimations and
		rebalances; 1 re-estimates at every step.
	"""

	def __init__(self, fitting_log_returns, window: int, rebalance_period: int = 1):
		"""Validates the window and stores the warm-up tail of the fitting sample."""
		super().__init__(fitting_log_returns, rebalance_period)
		fitting_log_returns = np.asarray(fitting_log_returns, dtype=np.float64)
		if isinstance(window, bool) or int(window) != window or not 3 <= window <= fitting_log_returns.shape[0]:
			raise ValueError(
				f"{type(self).__name__} window must be an integer in [3, {fitting_log_returns.shape[0]}] "
				f"(the fitting sample length), got {window!r}"
			)
		self._window = int(window)
		self._warmup = fitting_log_returns[-self._window:]

	def __str__(self):
		return f"OLS-R{self._window}-P{self._rebalance_period}"

	def _target_ratio(self, env: HedgeEnv, step_idx: int) -> float:
		"""OLS hedge ratio on the ``window`` returns realized before trading at step ``step_idx``."""
		prices = np.asarray(env.prices[:step_idx + 1], dtype=np.float64)
		realized = np.log(prices[1:] / prices[:-1])[-self._window:] # (min(t, window), 2)
		window_returns = np.concatenate([self._warmup[len(realized):], realized])
		return self.ols_hedge_ratio(window_returns)

HEDGE_BASELINE_STRATEGIES = {
	"theoretical_minimum_variance": TheoreticalMinVar(),
	# "unhedged_anchor": UnhedgedAnchor(),
	"MRR-C_1": CostAwareMeanVarHedge(lam=1),
	"MRR-C_2": CostAwareMeanVarHedge(lam=2),
	"MRR-C_10": CostAwareMeanVarHedge(lam=10),
}
