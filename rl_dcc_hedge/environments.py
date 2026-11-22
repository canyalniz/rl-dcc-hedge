from __future__ import annotations

import datetime
import math

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Iterable, Sequence
import itertools

from rl_dcc_hedge.utils.env_tools import alpha_bounds, repair_closed_form

from scipy import linalg
from scipy.stats import t as student_t


class HedgeEnv(gym.Env):

    # metadata = {"render_modes": ["human"]}
    KNOWN_RC = ["log_return", "downside_risk_penalty", "alpha_penalty", "cvar_95", "vol", "var"]
    # reward components that read the step's conditional std / correlation features
    CONDITIONAL_RC = ("cvar_95", "vol", "var")
    DOWNSIDE_CAP = 10.0

    def __init__(
        self,
        sampling_times: ArrayLike,
        num_features: int,
        tics: List[str],
        starting_prices: Dict[str,float],
        initial_amount: float = 1e4,
        reward_scaling: float =  1,
        transaction_cost_rate: float = 0.0,
        annual_financing_rate: float = 0.0,
        reward_components: str = "log_return:1.0,downside_risk_penalty:0.2,alpha_penalty:0.01",
        float_dtype: type = np.float32,
        abs_ratio_bound: float = 2.5,
        im: float = 0.5,
        mm_long: float = 0.25,
        mm_short: float = 0.30,
        bankruptcy_penalty: float = -2.5,
        infeasible_state_penalty: float = -0.01,
        record_trajectory: bool = True,
        dist_params: Optional[Dict[str, Any]] = None,
        condstd_feature_indices: Optional[Sequence[int]] = None,
        condcorr_feature_index: Optional[int] = None,
        **kwargs,
    ):
        self.float_dtype = float_dtype
        self.num_features = num_features
        self.sampling_times = sampling_times # (timesteps+1,)
        self.timesteps = len(self.sampling_times) - 1
        self.tic_list = tics
        self.num_tics = len(tics)
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.record_trajectory = record_trajectory
        
        self.abs_ratio_bound = abs_ratio_bound
        if not (0 <= transaction_cost_rate < min(mm_long, mm_short)):
            raise ValueError(f"transaction_cost_rate ({transaction_cost_rate}) "
                            f"must be in [0, min(mm_long, mm_short))")
        self.transaction_cost_rate = transaction_cost_rate
        # single annual cash rate serving as both the risk-free interest rate
        # (earned on positive cash) and the financing rate (paid on negative,
        # i.e. borrowed cash): the equal-rates idealization
        self.annual_financing_rate = float(annual_financing_rate)
        if self.annual_financing_rate <= -1.0:
            raise ValueError(
                f"annual_financing_rate ({annual_financing_rate}) must be > -1.0")
        # multiplicative cash growth factor of each evolution step
        # (all 1.0 when the rate is 0: no accrual, previous behavior)
        self._step_financing_factors = self._compute_step_financing_factors()
        self.bankruptcy_penalty = bankruptcy_penalty
        self.infeasible_state_penalty = infeasible_state_penalty
        self.starting_prices = starting_prices        

        self.reward_component_weights = {}
        for component in reward_components.split(","):
            name, weight = component.split(":")
            weight = float(weight)  # Ensure weight is a float
            name = name.lower().strip()
            self.reward_component_weights[name] = weight

        self.reward_component_names = self.reward_component_weights.keys()

        self.im = im
        self.mm_long = mm_long # long margin rate
        self.mm_short = mm_short # short margin rate

        ## PORTFOLIO STATE VARIABLES ##
        self._anchor_equity = None
        self._secondary_equity = None
        self._cash_equity = None
        self._anchor_position = None
        self._secondary_position = None
        self._mm = None
        self._alpha = None
        self._leverage = None
        self._alpha_LB = None
        self._alpha_UB = None

        # define observation array
        NUM_OBS_VAR = 8
        self._obs = np.ndarray(NUM_OBS_VAR + num_features, dtype=float_dtype)

        self._idx = 0

        # state: features + portfolio_state
        obs_shape = len(self._obs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=float_dtype,
        )

        self.action_space = spaces.Box(
            low=-abs_ratio_bound,
            high=abs_ratio_bound,
            shape=(1,),
            dtype=float_dtype,
        )

        ### Set up memory arrays ###
        # The episode memory backs get_trajectory_dict()/get_trajectory_df().
        # With record_trajectory=False the arrays are neither allocated nor
        # populated: step()/reset() skip all memory bookkeeping and the
        # trajectory getters raise instead of returning data.
        for name in self.reward_component_names:
            if name not in self.KNOWN_RC:
                raise ValueError(f"Unknown reward component {name!r}; known: {sorted(self.KNOWN_RC)}")

        ### cvar_95 / vol / var configuration (validated only when a component is active) ###
        # dist_params carries the fitted DCC(1,1)-ARMA(0,0)-eGARCH(1,1) t
        # innovation constants: {'df': <degrees of freedom>, 'mu': (anchor
        # drift, secondary drift)}; only cvar_95 needs them. The feature indices
        # point at the step's conditional standard deviations / correlation
        # inside the feature row; cvar_95, vol and var need them.
        self._t_df: Optional[float] = None
        self._t_mu: Optional[np.ndarray] = None
        self._es_factor: Optional[float] = None
        self._condstd_feature_indices: Optional[Tuple[int, int]] = None
        self._condcorr_feature_index: Optional[int] = None
        if "cvar_95" in self.reward_component_names:
            if dist_params is None:
                raise ValueError(
                    "reward component 'cvar_95' requires dist_params="
                    "{'df': <t degrees of freedom>, 'mu': (anchor drift, secondary drift)} "
                    "with the fitted model constants"
                )
            try:
                df = float(dist_params["df"])
                mu_vec = np.asarray(dist_params["mu"], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, ValueError) as err:
                raise ValueError(
                    f"dist_params must provide a numeric 'df' and a length-2 'mu', got {dist_params!r}"
                ) from err
            if not df > 2.0:
                raise ValueError(
                    f"dist_params['df'] must be > 2 (finite t variance), got {df}"
                )
            if mu_vec.shape != (2,):
                raise ValueError(
                    f"dist_params['mu'] must have length 2 [anchor, secondary], got {mu_vec.tolist()}"
                )
            q = student_t.ppf(0.95, df)
            self._t_df = df
            self._t_mu = mu_vec
            # ES multiplier k_nu of the t-scale for the worst 5% tail
            # (k_nu -> 2.0627, the Gaussian ES factor, as df -> inf)
            self._es_factor = float(
                student_t.pdf(q, df) * (df + q * q) / (0.05 * (df - 1.0))
            )

        active_conditional_rc = [n for n in self.CONDITIONAL_RC if n in self.reward_component_names]
        if active_conditional_rc:
            # the base class knows nothing about the feature layout: the
            # indices must be supplied (ST/MT subclasses default them)
            if condstd_feature_indices is None or condcorr_feature_index is None:
                raise ValueError(
                    f"reward component(s) {active_conditional_rc} require condstd_feature_indices "
                    "(anchor, secondary conditional-standard-deviation feature columns) and "
                    "condcorr_feature_index (the conditional-correlation feature column)"
                )
            cs_idx = list(condstd_feature_indices)
            if len(cs_idx) != 2:
                raise ValueError(
                    f"condstd_feature_indices must hold exactly 2 columns (anchor, secondary), "
                    f"got {cs_idx!r}"
                )
            for idx in (*cs_idx, condcorr_feature_index):
                if not (isinstance(idx, (int, np.integer)) and 0 <= idx < num_features):
                    raise ValueError(
                        f"conditional-moment feature index {idx!r} is not an integer in [0, {num_features})"
                    )
            self._condstd_feature_indices = (int(cs_idx[0]), int(cs_idx[1]))
            self._condcorr_feature_index = int(condcorr_feature_index)

        if self.record_trajectory:
            self._portfolio_simple_returns_memory = self._init_episode_memory_array()
            assert self._portfolio_simple_returns_memory.shape==(self.timesteps,), "Something is wrong with memory array initialization"

            self._post_tx_equity_memory = self._init_episode_memory_array()
            self._post_evo_equity_memory = self._init_episode_memory_array()

            self._post_tx_leverage_memory = self._init_episode_memory_array()
            self._post_evo_leverage_memory = self._init_episode_memory_array()

            self._reward_memory = self._init_episode_memory_array()
            self._reward_components_memory = {
                name: self._init_episode_memory_array() for name in self.reward_component_names
            }
            self._transaction_cost_memory = self._init_episode_memory_array()
            self._cash_interest_memory = self._init_episode_memory_array()

            self._post_tx_alpha_memory = self._init_episode_memory_array()
            self._post_evo_alpha_memory = self._init_episode_memory_array()
            self._actions_memory = self._init_episode_memory_array()

            self._margin_calls_memory = self._init_episode_memory_array()
            self._infeasible_states_memory = self._init_episode_memory_array()

            self._post_tx_equities_memory = {
                "anchor": self._init_episode_memory_array(),
                "secondary": self._init_episode_memory_array(),
                "cash": self._init_episode_memory_array(),
            }

        ### Finish setting up ###
        self._terminal = True # terminal is set to True to make sure reset is called before step is attempted
        self._reward = None
        self._info = {}
        # fatal edge-case flags, refreshed by every set_portfolio_state() call
        self._alpha_undefined = False
        self._leverage_undefined = False

    def step(self, actions:ArrayLike):
        """
        Advances the environment by one timestep.

        Returns (obs, reward, terminated, truncated, info). If the returned
        observation is an invalid state (bankruptcy, undefined alpha,
        undefined leverage), the affected state variables hold finite default
        values and info is {"invalid_state": True, "reasons": [...]} listing
        the fatal edge cases that fired (see _fatal_state_reasons).
        """

        if self._terminal:
            raise RuntimeError("step() called after termination; call reset() first")
        
        total_friction_cost = 0.0
        time_idx = self._idx
        start_equity = self._anchor_equity + self._secondary_equity + self._cash_equity

        ### input handling ###
        # action = - e_s / e_a
        alpha = actions[0]
        if self.record_trajectory:
            self._actions_memory[self._idx] = actions[0]
        

        ## check initial margin constraints ##
        alpha_adjustment = 0

        if not self._infeasible:
            if alpha < self._alpha_LB:
                alpha_adjustment = (self._alpha_LB - alpha)
                alpha = self._alpha_LB
            elif alpha > self._alpha_UB:
                alpha_adjustment = (alpha - self._alpha_UB)
                alpha = self._alpha_UB
            ### action transaction ###
            ## calculate equity targets ##
            t_secondary_equity = - alpha * self._anchor_equity
            t_secondary_position = int(t_secondary_equity / self.prices[time_idx,1]) # int rounds toward zero, suitable for feasibility
            t_secondary_equity = t_secondary_position * self.prices[time_idx,1]

            ## calculate friction ##
            secondary_equity_diff = t_secondary_equity - self._secondary_equity
            transaction_cost = self.transaction_cost_rate * abs(secondary_equity_diff)
            total_friction_cost += transaction_cost

            ## commit transaction ##
            self.set_portfolio_state(
                anchor_position=self._anchor_position, # anchor position does not change
                anchor_price=self.prices[time_idx,0], # prices have not yet evolved
                secondary_position=t_secondary_position, # target secondary position
                secondary_price=self.prices[time_idx,1], # prices have not yet evolved
                cash_equity=self._cash_equity - transaction_cost - secondary_equity_diff, # C - cost - equity_diff
            )

        ### post-transaction book snapshot ###
        # unconditional locals: reward components need the ex-ante book also
        # when record_trajectory=False (no memory write happens then) and when
        # the step skipped trading (infeasible: post-tx == standing book)
        post_tx_anchor_equity = self._anchor_equity
        post_tx_secondary_equity = self._secondary_equity
        post_tx_cash_equity = self._cash_equity
        pre_evo_time_idx = time_idx  # features[time_idx] governs this step's return

        ## post tx memory ##
        if self.record_trajectory:
            self._post_tx_equity_memory[self._idx] = self._anchor_equity + self._secondary_equity + self._cash_equity
            self._post_tx_equities_memory['anchor'][self._idx] = self._anchor_equity
            self._post_tx_equities_memory['secondary'][self._idx] = self._secondary_equity
            self._post_tx_equities_memory['cash'][self._idx] = self._cash_equity
            self._post_tx_alpha_memory[self._idx] = self._alpha
            self._post_tx_leverage_memory[self._idx] = self._leverage

        ### price evolution ###
        # cash accrues the cash rate over the step: positive cash earns the
        # risk-free interest rate, negative (borrowed) cash pays the financing
        # rate; the two rates are equal (annual_financing_rate idealization)
        pre_accrual_cash = self._cash_equity
        financing_factor = self._step_financing_factors[self._idx]
        accrued_cash = pre_accrual_cash * financing_factor
        if self.record_trajectory:
            self._cash_interest_memory[self._idx] = accrued_cash - pre_accrual_cash
        time_idx += 1
        self.set_portfolio_state(
            anchor_position=self._anchor_position, # no change
            anchor_price=self.prices[time_idx,0], # prices evolve with time index
            secondary_position=self._secondary_position, # no change
            secondary_price=self.prices[time_idx,1], # prices evolve with time index
            cash_equity=accrued_cash, # cash accrues the financing/risk-free rate
        )

        ### check portfolio health ###
        E_a = self._anchor_equity # anchor equity
        q_a = self._anchor_position # anchor position
        p_a = self.prices[time_idx,0] # anchor price
        E_s = self._secondary_equity # secondary equity
        q_s = self._secondary_position # secondary position
        p_s = self.prices[time_idx,1] # secondary price
        C = self._cash_equity # cash equity
        E = E_a + E_s + C # portfolio equity
        MM = self._mm
        tcr = self.transaction_cost_rate
        
        ## check margin call ##
        bankruptcy = False
        if E <= MM: # margin call condition
            if self.record_trajectory:
                self._margin_calls_memory[self._idx] = 1
            ret = repair_closed_form(
                float(q_a), float(q_s), float(p_a), float(p_s), float(C),
                float(tcr), self.mm_long, self.mm_short,
            )

            if ret is None:
                bankruptcy = True
                delta_E_a_rep, delta_E_s_rep, C_rep, q_a_rep, q_s_rep, friction_cost_rep = self.get_bankruptcy_liquidation(
                    q_a, q_s, p_a, p_s, C, tcr
                )

                total_friction_cost += friction_cost_rep
            else:
                q_a_rep = int(ret[0])
                q_s_rep = int(ret[1])
                delta_E_a_rep = q_a_rep*p_a - E_a
                delta_E_s_rep = q_s_rep*p_s - E_s
                friction_cost_rep = tcr*(abs(delta_E_s_rep)+abs(delta_E_a_rep))
                C_rep = C - (delta_E_a_rep + delta_E_s_rep + friction_cost_rep)

                total_friction_cost += friction_cost_rep
                

                mm_a_rep = self.mm_long
                mm_s_rep = self.mm_long if q_s_rep >=0 else self.mm_short
                E_rep = E - tcr*(abs(delta_E_s_rep)+abs(delta_E_a_rep))
                MM_rep = mm_a_rep*abs(q_a_rep*p_a) + mm_s_rep*abs(q_s_rep*p_s)

                if E_rep < MM_rep:
                    bankruptcy = True
                    delta_E_a_rep, delta_E_s_rep, C_rep, q_a_rep, q_s_rep, friction_cost_rep = self.get_bankruptcy_liquidation(
                        q_a_rep, q_s_rep, p_a, p_s, C_rep, tcr
                    )

                    total_friction_cost += friction_cost_rep
                    print("WARNING: Unexpected behavior in margin call repair")

            # commit margin call effects
            self.set_portfolio_state(
                anchor_position=q_a_rep,
                anchor_price=p_a,
                secondary_position=q_s_rep,
                secondary_price=p_s,
                cash_equity=C_rep,
            )

        ### post-evo memory updates ###
        end_equity = self._anchor_equity + self._secondary_equity + self._cash_equity
        if self.record_trajectory:
            self._post_evo_equity_memory[self._idx] = end_equity
            self._post_evo_alpha_memory[self._idx] = self._alpha
            self._post_evo_leverage_memory[self._idx] = self._leverage
        if end_equity > 0:
            eq_change_ratio = end_equity / start_equity
            portfolio_simple_returns = eq_change_ratio - 1
            portfolio_log_returns = math.log(eq_change_ratio)
        else:
            # bankruptcy / wiped-out default values: total loss for the simple
            # return, bounded floor for the log return (no log of <= 0 exists);
            # bankruptcy_penalty (added below) is the step's economic signal
            bankruptcy = True
            eq_change_ratio = max(end_equity / start_equity, 1e-12) if start_equity > 0 else 1.0
            portfolio_simple_returns = -1.0 if start_equity > 0 else 0.0
            portfolio_log_returns = math.log(eq_change_ratio)   # bounded floor

        if self.record_trajectory:
            self._portfolio_simple_returns_memory[self._idx] = portfolio_simple_returns

        total_reward = 0.0
        if 'log_return' in self.reward_component_names:
            reward_contribution = 0 if bankruptcy else self.reward_component_weights['log_return'] * portfolio_log_returns
            if self.record_trajectory:
                self._reward_components_memory['log_return'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if 'downside_risk_penalty' in self.reward_component_names:
            d = min(max(0.0, -portfolio_simple_returns), self.DOWNSIDE_CAP)
            reward_contribution = 0 if bankruptcy else -self.reward_component_weights['downside_risk_penalty'] * d * d
            if self.record_trajectory:
                self._reward_components_memory['downside_risk_penalty'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if 'alpha_penalty' in self.reward_component_names:
            reward_contribution = 0 if bankruptcy else -self.reward_component_weights['alpha_penalty'] * abs(alpha_adjustment)
            if self.record_trajectory:
                self._reward_components_memory['alpha_penalty'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if 'cvar_95' in self.reward_component_names:
            # ex-ante 95% CVaR of the agent's post-transaction book under the
            # step-ahead conditional distribution (see _cvar_95); margin-call
            # repairs / bankruptcy during the step do not retroactively alter it
            cvar = self._cvar_95(pre_evo_time_idx, post_tx_anchor_equity, post_tx_secondary_equity, post_tx_cash_equity)
            reward_contribution = 0 if bankruptcy else -self.reward_component_weights['cvar_95'] * cvar
            if self.record_trajectory:
                self._reward_components_memory['cvar_95'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if 'vol' in self.reward_component_names:
            # ex-ante conditional volatility of the post-transaction book (see _vol)
            vol = self._vol(pre_evo_time_idx, post_tx_anchor_equity, post_tx_secondary_equity, post_tx_cash_equity)
            reward_contribution = 0 if bankruptcy else -self.reward_component_weights['vol'] * vol
            if self.record_trajectory:
                self._reward_components_memory['vol'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if 'var' in self.reward_component_names:
            # ex-ante conditional variance of the post-transaction book (see _var)
            var = self._var(pre_evo_time_idx, post_tx_anchor_equity, post_tx_secondary_equity, post_tx_cash_equity)
            reward_contribution = 0 if bankruptcy else -self.reward_component_weights['var'] * var
            if self.record_trajectory:
                self._reward_components_memory['var'][self._idx] = reward_contribution
            total_reward += reward_contribution
        if self._infeasible:
            reward_contribution = 0 if bankruptcy else self.infeasible_state_penalty
            if self.record_trajectory:
                self._infeasible_states_memory[self._idx] = 1
            total_reward += reward_contribution

        ## other components do not take effect if bankruptcy is active ##
        if bankruptcy:
            total_reward = self.bankruptcy_penalty
        
        self._reward = self.reward_scaling * total_reward
        if self.record_trajectory:
            self._reward_memory[self._idx] = total_reward # save w/o scaling
            self._transaction_cost_memory[self._idx] = total_friction_cost
        
        ### increment environment index ###
        self._idx = self._idx + 1
        
        ### set next state observation features ###
        self._obs[:self.num_features] = self.features[self._idx, :]

        ### check terminal condition ###
        self._terminal = self._idx >= self.timesteps
        if bankruptcy:
            self._terminal = True

        ### fatal edge-case bookkeeping ###
        # the returned observation may be an invalid state; defaults assigned
        # along the way keep it finite, the info dict communicates invalidity
        reasons = self._fatal_state_reasons(bankruptcy=bankruptcy)
        self._info = {"invalid_state": True, "reasons": reasons} if reasons else {}

        return self.get_observation(), self._reward, self._terminal, False, self._info


    @staticmethod
    def get_bankruptcy_liquidation(q_a, q_s, p_a, p_s, C, tcr):
        """
        Liquidates the whole portfolio in the bankruptcy case.

        All positions are closed at current prices, transaction (friction)
        costs are paid on both legs, and the proceeds and costs are settled
        in cash.

        Returns
        -------
        (delta_E_a, delta_E_s, C, q_a, q_s, friction_cost)
            Changes in anchor/secondary equity, updated cash equity,
            zeroed-out positions and the friction cost of liquidation.
        """
        delta_E_a = -q_a*p_a
        delta_E_s = -q_s*p_s
        friction_cost = tcr*(abs(delta_E_s)+abs(delta_E_a))
        C -= (delta_E_a + delta_E_s + friction_cost)
        q_a = 0
        q_s = 0

        return delta_E_a, delta_E_s, C, q_a, q_s, friction_cost

    def _cvar_95(
        self,
        time_idx: int,
        anchor_equity: float,
        secondary_equity: float,
        cash_equity: float,
    ) -> float:
        """Analytic 95% CVaR (expected shortfall) of the one-step-ahead portfolio return.

        Evaluates the ex-ante tail risk of the post-transaction book under the
        fitted DCC(1,1)-ARMA(0,0)-eGARCH(1,1) model with Student-t innovations,
        using the conditional standard deviation / correlation features of the
        step (``self.features[time_idx]``, the same row exposed by the
        observation at the start of the step) and the post-transaction equity
        weights

            w = (E_a, E_s, C) / E,    E = E_a + E_s + C    (E > 0 required).

        With conditional variances ``h = condstd**2`` and correlation rho the
        step-ahead return is linearized as

            r_p = w_a (mu_a + s_a z_a) + w_s (mu_s + s_s z_s) + w_c (f - 1),

        where (z_a, z_s) are unit-variance jointly t-distributed standardized
        innovations, mu are the fitted drift constants and f - 1 is the
        deterministic cash accrual of the step
        (``f = self._step_financing_factors[time_idx]``, 1.0 when
        annual_financing_rate == 0). The variance of the stochastic part is

            v = w_a^2 h_a + w_s^2 h_s + 2 w_a w_s rho s_a s_s,

        and expressing the return against a raw t_nu innovation gives the
        scale ``s_p = sqrt((nu - 2)/nu * v)`` (covariance convention: the
        unit-variance standardized t is the raw t scaled by sqrt((nu-2)/nu)).
        The expected shortfall of the worst 5% tail is then

            CVaR_95 = k_nu * s_p - drift,   drift = w.mu + w_c (f - 1),
            k_nu    = t.pdf(t.ppf(0.95, nu), nu) * (nu + q^2) / (0.05 (nu - 1))
                      -> 2.0627 (the Gaussian ES factor) as nu -> inf.

        Guards
        ------
        - ``E <= 0``: returns 0.0 (weights are undefined; the bankruptcy
          machinery dominates the step's economics anyway)
        - non-finite condstd/condcorr, ``condstd <= 0`` or ``|rho| > 1``:
          ValueError (the conditional distribution itself is malformed)

        Returns
        -------
        float
            CVaR_95 in per-step return units. Non-negative for zero-drift
            books; negative when the deterministic drift outweighs the tail
            risk (e.g. an all-cash book accruing interest).
        """
        moments = self._conditional_portfolio_moments(
            time_idx, anchor_equity, secondary_equity, cash_equity
        )
        if moments is None:
            return 0.0
        (w_a, w_s, w_c), variance = moments

        s_p = math.sqrt((self._t_df - 2.0) / self._t_df * variance)
        financing_factor = float(self._step_financing_factors[time_idx])
        drift = float(
            w_a * self._t_mu[0] + w_s * self._t_mu[1] + w_c * (financing_factor - 1.0)
        )
        return float(self._es_factor * s_p - drift)

    def _vol(
        self,
        time_idx: int,
        anchor_equity: float,
        secondary_equity: float,
        cash_equity: float,
    ) -> float:
        """Ex-ante conditional volatility of the one-step-ahead portfolio return.

        Returns ``sigma_p = sqrt(v)`` with the post-transaction weights and the
        conditional variance ``v`` defined as in _cvar_95 (same features, same
        linearization). Unlike cvar_95 it needs no distribution constants.

        Under the fitted model (multivariate t with constant nu) the standardized
        portfolio return has a fixed distribution, so

            CVaR_95 = c_nu * sigma_p - drift,   c_nu = k_nu * sqrt((nu - 2)/nu),

        (c_nu ~ 2.229 for nu = 5.349): 'cvar_95:lam' equals 'vol:c_nu*lam' up to
        the drift term, whose expectation 'log_return:lam' approximates.

        Guards are those of _cvar_95 (``E <= 0`` -> 0.0; malformed conditional
        moments -> ValueError).

        Returns
        -------
        float
            sigma_p in per-step return units (non-negative).
        """
        moments = self._conditional_portfolio_moments(
            time_idx, anchor_equity, secondary_equity, cash_equity
        )
        if moments is None:
            return 0.0
        _, variance = moments
        return math.sqrt(variance)

    def _var(
        self,
        time_idx: int,
        anchor_equity: float,
        secondary_equity: float,
        cash_equity: float,
    ) -> float:
        """Ex-ante conditional variance of the one-step-ahead portfolio return.

        Returns ``v = sigma_p^2`` with the post-transaction weights and the
        conditional variance ``v`` defined as in _cvar_95 (same features, same
        linearization). Like vol it needs no distribution constants.

        A 'var:lam' penalty is the mean-variance (quadratic utility) risk term:
        with 'log_return:1.0' the per-step reward approximates the
        Arrow-Pratt objective E[r] - lam * v (lam = gamma_RA / 2 for a CARA
        investor with risk aversion gamma_RA). Unlike vol its marginal penalty
        2 * lam * sigma_p vanishes near the minimum-variance book, so it
        punishes large risk super-linearly and leaves small residual risk
        almost free.

        Guards are those of _cvar_95 (``E <= 0`` -> 0.0; malformed conditional
        moments -> ValueError).

        Returns
        -------
        float
            v in squared per-step return units (non-negative).
        """
        moments = self._conditional_portfolio_moments(
            time_idx, anchor_equity, secondary_equity, cash_equity
        )
        if moments is None:
            return 0.0
        _, variance = moments
        return float(variance)

    def _conditional_portfolio_moments(
        self,
        time_idx: int,
        anchor_equity: float,
        secondary_equity: float,
        cash_equity: float,
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """Post-transaction weights and conditional portfolio return variance.

        Shared by _cvar_95, _vol and _var. Reads the conditional standard deviations
        and correlation of ``self.features[time_idx]`` (the step-ahead
        forecasts) and returns ``((w_a, w_s, w_c), v)`` with

            w = (E_a, E_s, C) / E,
            v = w_a^2 h_a + w_s^2 h_s + 2 w_a w_s rho s_a s_s,

        or None when ``E <= 0`` (weights undefined). Raises ValueError for
        non-finite features, ``condstd <= 0`` or ``|rho| > 1``.
        """
        E = float(anchor_equity + secondary_equity + cash_equity)
        if E <= 0.0:
            return None

        s_a = float(self.features[time_idx, self._condstd_feature_indices[0]])
        s_s = float(self.features[time_idx, self._condstd_feature_indices[1]])
        rho = float(self.features[time_idx, self._condcorr_feature_index])
        if not (math.isfinite(s_a) and math.isfinite(s_s) and math.isfinite(rho)):
            raise ValueError(
                f"conditional-moment reward components need finite condstd/condcorr features "
                f"at step {time_idx}, got condstd=({s_a!r}, {s_s!r}), condcorr={rho!r}"
            )
        if s_a <= 0.0 or s_s <= 0.0:
            raise ValueError(
                f"conditional-moment reward components need positive conditional standard "
                f"deviations at step {time_idx}, got condstd=({s_a!r}, {s_s!r})"
            )
        if abs(rho) > 1.0:
            raise ValueError(
                f"conditional-moment reward components need |condcorr| <= 1 at step "
                f"{time_idx}, got {rho!r}"
            )

        w_a = float(anchor_equity) / E
        w_s = float(secondary_equity) / E
        w_c = float(cash_equity) / E

        # conditional covariance of the step-ahead asset returns
        variance = (
            (w_a * s_a) ** 2 + (w_s * s_s) ** 2 + 2.0 * w_a * w_s * rho * s_a * s_s
        )
        # |rho| <= 1 makes v >= 0 in exact arithmetic; clip float noise
        variance = max(float(variance), 0.0)
        return (w_a, w_s, w_c), variance

    def _compute_step_financing_factors(self) -> np.ndarray:
        """Per-step multiplicative cash growth factors for interest accrual.

        Step i's factor is (1 + annual_financing_rate) ** dt_i, where dt_i is
        the step's year fraction derived from the sampling_times day count
        (ACT/365.25); datetime-like sampling_times yield the actual calendar
        intervals, matching the (possibly irregular) schedules of the ST/MT
        environments.

        With the default rate of 0 the factors are all 1.0 (no accrual) and the
        environment behaves exactly as before the financing mechanism existed;
        in particular non-datetime sampling_times are then accepted. A nonzero
        rate requires datetime-like sampling_times; numeric ones are rejected
        rather than guessing the accrual intervals. Object arrays of
        datetime.date objects (the parquet DATE32 convention of the test
        partition) are accepted and coerced.
        """
        if self.annual_financing_rate == 0.0:
            return np.ones(self.timesteps, dtype=np.float64)
        times = np.asarray(self.sampling_times)
        if times.dtype == object and times.size > 0 and all(
            isinstance(t, datetime.date) for t in times
        ):
            # datetime.datetime (incl. pandas.Timestamp) subclasses
            # datetime.date; only date-like objects are coerced -- numeric
            # arrays must NOT reach pd.to_datetime, which would silently read
            # them as epoch nanoseconds
            times = pd.to_datetime(pd.Series(times)).to_numpy()
        if not np.issubdtype(times.dtype, np.datetime64):
            raise ValueError(
                "annual_financing_rate != 0 requires datetime-like sampling_times "
                "to day-count the accrual intervals, got sampling_times of dtype "
                f"{times.dtype!r}; pass e.g. pandas date_range timestamps"
            )
        day_counts = np.diff(times).astype("timedelta64[D]").astype(np.float64)
        if np.any(day_counts <= 0):
            raise ValueError(
                "sampling_times must be strictly increasing for financing accrual"
            )
        year_fractions = day_counts / 365.25
        return (1.0 + self.annual_financing_rate) ** year_fractions

    def _fatal_state_reasons(self, bankruptcy: bool = False) -> List[str]:
        """Fatal edge-case audit of the currently committed portfolio state.

        Reasons reported:
          "bankruptcy":         margin-call repair found no feasible portfolio;
                                the book was fully liquidated (terminal step)
          "alpha_undefined":    anchor equity is zero -> alpha = -E_s/E_a is
                                undefined; _psi_alpha holds the 0.0 default
          "leverage_undefined": portfolio equity is exactly zero -> leverage
                                gross/|E| is undefined; _psi_leverage holds
                                the 0.0 default
        """
        reasons = []
        if bankruptcy:
            reasons.append("bankruptcy")
        if self._alpha_undefined:
            reasons.append("alpha_undefined")
        if self._leverage_undefined:
            reasons.append("leverage_undefined")
        return reasons

    def reset(self,*,seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[ArrayLike, dict]:
        """
        Resets the environment and returns it to its initial state.
        """
        super().reset(seed=seed)
        
        # reset memory arrays
        self._reset_memory()
        self.reset_portfolio_state() # set portfolio state variables
        self._obs[:self.num_features] = self.features[0,:] # set precomputed feature variables
        self._terminal = False
        self._reward = None
        self._idx = 0
        # initial state may itself be invalid (e.g. flat anchor); flag it like step() does
        reasons = self._fatal_state_reasons()
        self._info = {"invalid_state": True, "reasons": reasons} if reasons else {}

        return self.get_observation(), self._info

    def get_observation(self):
        E = self._anchor_equity + self._secondary_equity + self._cash_equity
        portfolio_obs = np.array([
            self._anchor_equity / E,
            self._secondary_equity / E,
            self._cash_equity / E,
            self._mm / E,
            1 - self._idx/self.timesteps,
            self._alpha / self.abs_ratio_bound,
            self._alpha_LB / self.abs_ratio_bound,
            self._alpha_UB / self.abs_ratio_bound,
        ])
        features_obs = self.features[self._idx, :]

        return np.concat([portfolio_obs,features_obs], dtype=self.float_dtype)

    def set_portfolio_state(
        self,
        anchor_position,
        anchor_price,
        secondary_position,
        secondary_price,
        cash_equity,
    ):
        anchor_equity = anchor_position*anchor_price
        secondary_equity = secondary_position*secondary_price
        # fatal edge case: anchor equity wiped -> alpha = -E_s/E_a undefined.
        # Default to 0.0 so the state stays finite; flagged for the info dict.
        self._alpha_undefined = (anchor_equity == 0)
        alpha = 0.0 if self._alpha_undefined else -secondary_equity/anchor_equity

        portfolio_equity = anchor_equity+secondary_equity+cash_equity
        # fatal edge case: portfolio equity exactly zero -> leverage undefined.
        # Default to 0.0 so the state stays finite; flagged for the info dict.
        self._leverage_undefined = (portfolio_equity == 0)
        leverage = 0.0 if self._leverage_undefined else (abs(anchor_equity)+abs(secondary_equity)) / portfolio_equity
        
        mm_secondary = self.mm_long if secondary_position >= 0 else self.mm_short

        self._anchor_equity = anchor_equity
        self._secondary_equity = secondary_equity
        self._cash_equity = cash_equity
        self._anchor_position = anchor_position
        self._secondary_position = secondary_position
        self._mm = self.mm_long*anchor_equity + mm_secondary*abs(secondary_equity)
        self._alpha = alpha
        self._leverage = leverage
        alpha_LB, alpha_UB, infeasible = alpha_bounds(
            C=cash_equity,
            q0_1=anchor_position, # represents post-tx position, agent's attempted action will not alter this position
            p0=anchor_price,
            q1_0=secondary_position,
            p1=secondary_price,
            tcr=self.transaction_cost_rate,
            im=self.im,
        )
        # infeasible (incl. anchor flat -> alpha undefined): fall back to the
        # current alpha (itself the 0.0 default in that case) so the bound
        # slots stay finite instead of NaN
        self._alpha_LB = max(alpha_LB, -self.abs_ratio_bound) if not infeasible else alpha
        self._alpha_UB = min(alpha_UB, self.abs_ratio_bound) if not infeasible else alpha

        # infeasibility means no legal alpha exists for which a valid position may be entered
        # any allowed alpha would fail the initial margin check
        self._infeasible = infeasible

        

    def reset_portfolio_state(self):
        anchor_position = self.initial_amount // self.starting_prices['anchor']
        cash_equity = self.initial_amount % self.starting_prices['anchor']

        self.set_portfolio_state(
            anchor_position=anchor_position,
            anchor_price=self.starting_prices['anchor'],
            secondary_position=0,
            secondary_price=self.starting_prices['secondary'],
            cash_equity=cash_equity,
        )

    def _init_episode_memory_array(self, timesteps=None, record_shape=(1,)) -> ArrayLike:
        shape = (self.timesteps,) if (timesteps is None) else (timesteps,)
        if record_shape != (1,):
            record_shape = (record_shape,) if isinstance(record_shape, int) else record_shape
            shape = shape + record_shape
        return np.zeros(shape, dtype=self.float_dtype)

    def _reset_memory(self):
        """Resets the environment's memory."""
        if not self.record_trajectory:
            return # memory arrays were never allocated
        self._post_tx_equity_memory.fill(0)
        self._post_evo_equity_memory.fill(0)
        self._post_tx_leverage_memory.fill(0)
        self._post_evo_leverage_memory.fill(0)
        self._post_tx_equities_memory["anchor"].fill(0)
        self._post_tx_equities_memory["secondary"].fill(0)
        self._post_tx_equities_memory["cash"].fill(0)
        self._portfolio_simple_returns_memory.fill(0)
        self._reward_memory.fill(0)
        for name in self.reward_component_names:
            self._reward_components_memory[name].fill(0)
        self._transaction_cost_memory.fill(0)
        self._cash_interest_memory.fill(0)
        self._post_evo_alpha_memory.fill(0)
        self._post_tx_alpha_memory.fill(0)
        self._actions_memory.fill(0)
        self._margin_calls_memory.fill(0)
        self._infeasible_states_memory.fill(0)

    def get_trajectory_dict(self):
        if not self.record_trajectory:
            raise RuntimeError(
                "get_trajectory_dict() needs trajectory memory, but this environment was "
                "constructed with record_trajectory=False; re-create it with "
                "record_trajectory=True to record trajectories"
            )
        trajectory = {
            "post_tx_equity": self._post_tx_equity_memory[:self._idx].copy(),
            "post_evo_equity": self._post_evo_equity_memory[:self._idx].copy(),
            "post_tx_anchor_asset_equity": self._post_tx_equities_memory["anchor"][:self._idx].copy(),
            "post_tx_secondary_asset_equity": self._post_tx_equities_memory["secondary"][:self._idx].copy(),
            "post_tx_cash_asset_equity": self._post_tx_equities_memory["cash"][:self._idx].copy(),
            "post_tx_leverage": self._post_tx_leverage_memory[:self._idx].copy(),
            "post_evo_leverage": self._post_evo_leverage_memory[:self._idx].copy(),
            "post_tx_alpha": self._post_tx_alpha_memory[:self._idx].copy(),
            "post_evo_alpha": self._post_evo_alpha_memory[:self._idx].copy(),
            "action": self._actions_memory[:self._idx].copy(),
            "portfolio_simple_returns": self._portfolio_simple_returns_memory[:self._idx].copy(),
            "reward": self._reward_memory[:self._idx], # unscaled total reward
            "transaction_cost": self._transaction_cost_memory[:self._idx].copy(), # total transaction costs incurred
            "cash_interest": self._cash_interest_memory[:self._idx].copy(), # signed interest accrued on cash this step: positive = risk-free income, negative = financing cost
            "margin_calls": self._margin_calls_memory[:self._idx].copy(), # boolean margin call indicator
            "infeasible_states": self._infeasible_states_memory[:self._idx].copy() # infeasible states indicator
        }
        for name in self.reward_component_names:
            trajectory[f"reward_component_{name}"] = self._reward_components_memory[name][:self._idx].copy()

        return trajectory
    
    def get_trajectory_df(self):
        data = self.get_trajectory_dict()
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def verify_data_alignment(
        prices:np.typing.ArrayLike,
        returns:np.typing.ArrayLike,
        innov:np.typing.ArrayLike,
        condstd:np.typing.ArrayLike,
        condcorr:np.typing.ArrayLike,
        mu:np.typing.ArrayLike,
    ):
        covar = condstd.prod(axis=1) * condcorr

        H = np.ndarray((prices.shape[0],2,2), dtype=np.float32) # covariance matrix
        H[:,0,0] = condstd[:,0]**2 # variance of asset 0
        H[:,1,1] = condstd[:,1]**2 # variance of asset 1
        H[:,1,0] = covar # covariance of asset 0, asset 1
        H[:,0,1] = covar

        H_sqrt = linalg.sqrtm(H)
        residuals = np.matmul(H_sqrt[:-1,:,:], np.expand_dims(innov[1:,:], axis=-1)).squeeze() # mu + H^{-1/2}_{t} @ z_t = r_t, H is offset by 1

        ok = True
        if not np.isclose(prices[:-1]  * np.exp(returns[1:]), prices[1:]).all():
            print("--"*50)
            print("Prices and returns are not aligned properly...") # exp(r_t) * p_{t-1} = p_t
            print("Prices calculated with returns:")
            print((prices[:-1]  * np.exp(returns[1:]))[:5])
            print("Prices in data")
            print(prices[1:6])
            print("--"*50)
            ok = False
        if not np.isclose((mu + residuals), returns[1:,], atol=1e-6).all():
            print("Covariance values, innovations and returns are not aligned properly...")
            print("--"*50)
            print("Prices and returns are not aligned properly...") # exp(r_t) * p_{t-1} = p_t
            print("Returns calculated via model:")
            print((mu+residuals)[:5])
            print("Returns in data")
            print(returns[1:6])
            print("--"*50)
            ok = False
        if not ok:
            raise ValueError("Data misaligned: prices/returns or innovations/model returns")

class ST_HedgeEnv(HedgeEnv):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: ArrayLike,
        prices: ArrayLike,
        sampling_times: ArrayLike,
        num_features: int,
        tics: List[str],
        initial_amount: float = 1e4,
        reward_scaling: float =  1,
        transaction_cost_rate: float = 0.0,
        annual_financing_rate: float = 0.0,
        reward_components: str = "log_return:1.0,downside_risk_penalty:0.2,alpha_penalty:0.01",
        float_dtype: type = np.float32,
        abs_ratio_bound: float = 2.5,
        im: float = 0.5,
        mm_l: float = 0.25,
        mm_s: float = 0.30,
        bankruptcy_penalty: float = -2.5,
        infeasible_state_penalty: float = -0.01,
        record_trajectory: bool = True,
        starting_prices: Optional[Dict[str,float]] = None,
        dist_params: Optional[Dict[str, Any]] = None,
        condstd_feature_indices: Optional[Sequence[int]] = None,
        condcorr_feature_index: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        features: ArrayLike
        (timesteps+1,) shaped array of state features

        prices: ArrayLike
        (timesteps+1,) shaped array of prices

        dist_params: Optional[Dict[str, Any]]
        Fitted model constants {'df': t dof, 'mu': (anchor drift, secondary
        drift)} required by the 'cvar_95' reward component (see HedgeEnv).

        condstd_feature_indices, condcorr_feature_index
        Feature columns of the (anchor, secondary) conditional standard
        deviations and of the conditional correlation used by 'cvar_95', 'vol' and 'var'.
        When None with the component active they default to the repo-wide
        feature layout [innov_a, condstd_a, innov_s, condstd_s, condcorr]
        -> (1, 3) and 4 (DataHandler's ST test-partition feature frame uses
        this ordering).
        """

        self.features = np.array(features, dtype=float_dtype)
        self.prices = np.array(prices, dtype=float_dtype)

        rc_names = {part.split(":")[0].strip().lower() for part in reward_components.split(",")}
        if rc_names.intersection(HedgeEnv.CONDITIONAL_RC):
            # repo-wide layout: [innov_a, condstd_a, innov_s, condstd_s, condcorr]
            if condstd_feature_indices is None:
                condstd_feature_indices = (1, 3)
            if condcorr_feature_index is None:
                condcorr_feature_index = 4

        if starting_prices is None:
            starting_prices = {
                'anchor': self.prices[0,0],
                'secondary': self.prices[0,1]
            }
        assert math.isclose(starting_prices['anchor'], float(self.prices[0, 0])), "anchor starting price != data row 0"
        assert math.isclose(starting_prices['secondary'], float(self.prices[0, 1])), "secondary starting price != data row 0"

        super().__init__(
            sampling_times = sampling_times,
            num_features = num_features,
            tics = tics,
            initial_amount = initial_amount,
            reward_scaling = reward_scaling,
            transaction_cost_rate = transaction_cost_rate,
            annual_financing_rate = annual_financing_rate,
            reward_components = reward_components,
            float_dtype = float_dtype,
            abs_ratio_bound = abs_ratio_bound,
            im = im,
            mm_long = mm_l,
            mm_short = mm_s,
            bankruptcy_penalty = bankruptcy_penalty,
            infeasible_state_penalty = infeasible_state_penalty,
            record_trajectory = record_trajectory,
            starting_prices=starting_prices,
            dist_params=dist_params,
            condstd_feature_indices=condstd_feature_indices,
            condcorr_feature_index=condcorr_feature_index,
        )

class MT_HedgeEnv(HedgeEnv):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        traj_indices: Iterable[int],
        sampling_times: ArrayLike,
        tics: List[str],
        initial_amount: float = 1e4,
        reward_scaling: float =  1,
        transaction_cost_rate: float = 0.0,
        annual_financing_rate: float = 0.0,
        reward_components: str = "log_return:1.0,downside_risk_penalty:0.2,alpha_penalty:0.01",
        float_dtype: type = np.float32,
        abs_ratio_bound: float = 2.5,
        im: float = 0.5,
        mm_l: float = 0.25,
        mm_s: float = 0.30,
        bankruptcy_penalty: float = -2.5,
        infeasible_state_penalty: float = -0.01,
        per_tic_prices_column_tag: str = "prices",
        per_tic_features_column_tags: List[str] = ["innov", "condstd"],
        standalone_features_column_tags: List[str] = ["condcorr"],
        silent_traj_idx_reset: bool = False,
        record_trajectory: bool = True,
        starting_prices: Optional[Dict[str,float]] = None,
        dist_params: Optional[Dict[str, Any]] = None,
        condstd_feature_indices: Optional[Sequence[int]] = None,
        condcorr_feature_index: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
        Simulation data

        traj_indices : Iterable[int]
        Iterable that yields trajectory indices

        per_tic_prices_column_tag: str
        Prices column tag in simulation data

        per_tic_features_column_tags: str
        Per tic features (e.g. innov, condstd) column tags in simulation data

        standalone_features_column_tags: str
        Standalone features (e.g. condcorr) column tags in simulation data

        silent_traj_idx_reset: bool
        When this flag is set to False, the environment prints a notice when it cycles through all trajectories

        dist_params: Optional[Dict[str, Any]]
        Fitted model constants {'df': t dof, 'mu': (anchor drift, secondary
        drift)} required by the 'cvar_95' reward component (see HedgeEnv).

        condstd_feature_indices, condcorr_feature_index
        Feature columns of the (anchor, secondary) conditional standard
        deviations and of the conditional correlation used by 'cvar_95', 'vol' and 'var'.
        When None with the component active they are derived from the column
        tags: condstd of tic j sits at j*len(per_tic_features_column_tags) +
        per_tic_features_column_tags.index('condstd'), and condcorr at
        num_tics*len(per_tic_features_column_tags) +
        standalone_features_column_tags.index('condcorr') (matching the
        per-tic-major feature layout built in _proceed_to_next_trajectory);
        missing tags raise unless the indices are passed explicitly.
        """
        num_tics = len(tics)
        num_features = num_tics * len(per_tic_features_column_tags) + len(standalone_features_column_tags)

        rc_names = {part.split(":")[0].strip().lower() for part in reward_components.split(",")}
        if rc_names.intersection(HedgeEnv.CONDITIONAL_RC):
            if condstd_feature_indices is None:
                try:
                    condstd_pos = per_tic_features_column_tags.index("condstd")
                except ValueError:
                    raise ValueError(
                        "reward components 'cvar_95'/'vol'/'var' need per-tic 'condstd' feature columns to "
                        "derive condstd_feature_indices; add the 'condstd' per-tic tag or pass "
                        "the indices explicitly"
                    ) from None
                condstd_feature_indices = tuple(
                    j * len(per_tic_features_column_tags) + condstd_pos for j in range(num_tics)
                )
            if condcorr_feature_index is None:
                try:
                    condcorr_pos = standalone_features_column_tags.index("condcorr")
                except ValueError:
                    raise ValueError(
                        "reward components 'cvar_95'/'vol'/'var' need a standalone 'condcorr' feature column "
                        "to derive condcorr_feature_index; add the 'condcorr' standalone tag or "
                        "pass the index explicitly"
                    ) from None
                condcorr_feature_index = num_tics * len(per_tic_features_column_tags) + condcorr_pos

        super().__init__(
            sampling_times=sampling_times,
            num_features=num_features,
            tics=tics,
            initial_amount=initial_amount,
            reward_scaling=reward_scaling,
            transaction_cost_rate=transaction_cost_rate,
            annual_financing_rate=annual_financing_rate,
            reward_components=reward_components,
            float_dtype=float_dtype,
            abs_ratio_bound=abs_ratio_bound,
            im=im,
            mm_long=mm_l,
            mm_short=mm_s,
            bankruptcy_penalty=bankruptcy_penalty,
            infeasible_state_penalty=infeasible_state_penalty,
            record_trajectory=record_trajectory,
            starting_prices=starting_prices,
            dist_params=dist_params,
            condstd_feature_indices=condstd_feature_indices,
            condcorr_feature_index=condcorr_feature_index,
        )

        self._data = data
        self._silent_traj_idx_reset = silent_traj_idx_reset
        self._traj_idx = None
        self._traj_idx_iterator, self._traj_idx_iterator_copy = itertools.tee(iter(traj_indices), 2)

        self.prices = None
        self.features = None

        self._per_tic_prices_column_tag = per_tic_prices_column_tag
        self._per_tic_features_column_tags = per_tic_features_column_tags
        self._standalone_features_column_tags = standalone_features_column_tags

    def reset(self,*,seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[ArrayLike, dict]:
        self._proceed_to_next_trajectory(silent=self._silent_traj_idx_reset)
        assert math.isclose(self.starting_prices['anchor'], float(self.prices[0, 0])), f"anchor starting price != data row 0 at {self._traj_idx}"
        assert math.isclose(self.starting_prices['secondary'], float(self.prices[0, 1])), f"secondary starting price != data row 0 at {self._traj_idx}"
        return super().reset(seed=seed)

    def _proceed_to_next_trajectory(self, silent=False):
        try:
            self._traj_idx = next(self._traj_idx_iterator)
        except StopIteration:
            if not silent:
                print(f"{self.__hash__()}::Resetting trajectory index iterator...")
            self._traj_idx_iterator, self._traj_idx_iterator_copy = itertools.tee(self._traj_idx_iterator_copy, 2)
            self._traj_idx = next(self._traj_idx_iterator)
        
        # set prices
        colnames = [
            f"{self._traj_idx}_{j}_{self._per_tic_prices_column_tag}"
            for j in range(self.num_tics)
        ]
        self.prices = self._data[colnames].values

        # set features
        per_tic_features_colnames = [
            f"{self._traj_idx}_{j}_{tag}"
            for j in range(self.num_tics)
            for tag in self._per_tic_features_column_tags
        ]
        standalone_features_colnames = [
            f"{self._traj_idx}_{tag}"
            for tag in self._standalone_features_column_tags
        ]
        colnames = per_tic_features_colnames + standalone_features_colnames
        self.features = self._data[colnames].values

    @staticmethod
    def verify_data_matrix_alignment(data: pd.DataFrame, traj_idx=0, mu=[0.001440, 0.001407]): # updated to SPY_XLU
        i = traj_idx
        HedgeEnv.verify_data_alignment(
            prices = data[[f"{i}_0_prices",f"{i}_1_prices"]].values,
            returns = data[[f"{i}_0_returns",f"{i}_1_returns"]].values,
            innov = data[[f"{i}_0_innov",f"{i}_1_innov"]].values,
            condstd = data[[f"{i}_0_condstd",f"{i}_1_condstd"]].values,
            condcorr = data[f"{i}_condcorr"].values,
            mu = np.array(mu),
        )