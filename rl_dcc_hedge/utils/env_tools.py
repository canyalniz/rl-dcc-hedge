from __future__ import annotations

import math
from typing import Optional, Tuple


class InfeasibleMarginError(ValueError):
    """No hedge ratio alpha satisfies the initial-margin constraint."""


def _h(alpha: float, S: float, V0: float, B: float, tcr: float, im: float) -> float:
    """Feasibility slack h(alpha) = equity after costs - IM requirement."""
    return S - tcr * abs(alpha * V0 + B) - im * abs(V0) * (1.0 + abs(alpha))


def alpha_bounds(
    C: float,
    q0_1: float,
    p0: float,
    q1_0: float,
    p1: float,
    tcr: float,
    im: float,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """Return (alpha_min, alpha_max): the IM-feasible interval for the hedge ratio.

    Parameters
    ----------
    C    : starting cash
    q0_1, p0 : post-trade position and price of asset 0 (V0 = q0_1 * p0, must be != 0)
    q1_0, p1 : pre-trade position and price of asset 1 (B = q1_0 * p1)
    tcr  : proportional transaction-cost rate (>= 0), charged on |te|
    im   : initial-margin rate (> 0), applied to gross notional |V0| + |V1|
    tol  : relative tolerance for region-membership / feasibility tests

    Raises
    ------
    ValueError            if V0 == 0 (alpha undefined) or inputs out of range
    InfeasibleMarginError if no alpha satisfies the IM constraint
    """
    V0 = q0_1 * p0
    B = q1_0 * p1
    if V0 == 0.0:
        return None,None,True
    if tcr < 0.0 or im <= 0.0:
        raise ValueError("Require tcr >= 0 and im > 0.")

    S = C + V0 + B
    aV0 = abs(V0)
    eps = tol * max(abs(V0), abs(B), abs(S), 1.0)

    def h(a: float) -> float:
        return _h(a, S, V0, B, tcr, im)

    # Four linear pieces of h. Each entry: (root numerator, root denominator,
    # region test as a function of (alpha, x) with x = alpha*V0 + B).
    # Region sign pattern (alpha, x) <=> (hedge side, net buy/sell of asset 1):
    #   I  (+, +) net seller   II (+, -) net buyer   III (-, +) net seller   IV (-, -) net buyer
    pieces = [
        (S - tcr * B - im * aV0, tcr * V0 + im * aV0,
         lambda a, x: a >= -eps and x >= -eps),                       # I
        (S + tcr * B - im * aV0, im * aV0 - tcr * V0,
         lambda a, x: a >= -eps and x <= eps),                        # II
        (S - tcr * B - im * aV0, tcr * V0 - im * aV0,
         lambda a, x: a <= eps and x >= -eps),                        # III
        (-(S + tcr * B - im * aV0), tcr * V0 + im * aV0,
         lambda a, x: a <= eps and x <= eps),                         # IV
    ]

    candidates: list[float] = []
    for num, den, in_region in pieces:
        if abs(den) <= tol * max(aV0, 1.0):
            continue  # zero-slope piece: no crossing (region fully on/off, kinks cover it)
        rho = num / den
        if math.isfinite(rho) and in_region(rho, rho * V0 + B):
            candidates.append(rho)

    # Kinks that are themselves feasible: covers tangency / single-point sets,
    # where h touches zero exactly at alpha = 0 or alpha* = -B/V0.
    for kink in (0.0, -B / V0):
        if h(kink) >= -eps:
            candidates.append(kink)

    if not candidates:
        LB = None
        UB = None
        infeasible = True
    else:
        LB = min(candidates)
        UB = max(candidates)
        infeasible = False
    
    return LB, UB, infeasible


# ----------------------------------------------------------------------------
# margin-call repair (moved here from the _margin_call_prototype.py mock-up)
# ----------------------------------------------------------------------------
def total_state(qa: float, qs: float, pa: float, ps: float, C: float) -> float:
    """Total portfolio equity E = qa*pa + qs*ps + C at post-evolution prices."""
    return qa * pa + qs * ps + C


def lhs_y(y: float, qs: float, ps: float, tcr: float, mm_l: float, mm_s: float) -> float:
    """Maintenance-margin + friction contribution of the secondary leg at position y."""
    mm = mm_l if y >= 0 else mm_s
    return mm * abs(y) * ps + tcr * abs(y - qs) * ps


def repair_closed_form(
    qa: float,
    qs: float,
    pa: float,
    ps: float,
    C: float,
    tcr: float,
    mm_l: float,
    mm_s: float,
) -> Optional[Tuple[int, int]]:
    """Exact O(1) closed-form solver for the margin-call repair problem.

    The state is the post-price-evolution portfolio on which the margin call
    fired (E <= MM): anchor/secondary integer positions (qa, qs), prices
    (pa, ps), cash equity C, transaction-cost rate tcr and maintenance-margin
    rates mm_l (long leg) / mm_s (short leg). The repair (x, y) solves,
    lexicographically,

        1. maximize x   (retain as much anchor as possible; x <= qa dominates)
        2. minimize |y - qs|   (secondary disruption; ranks like friction
           tcr*(...) whenever tcr > 0, stays well-defined at tcr = 0)

    subject to the post-repair maintenance-margin constraint

        E - tcr*((qa - x)*pa + |y - qs|*ps) >= mm_l*x*pa + mm(y)*|y|*ps,
        mm(y) = mm_l if y >= 0 else mm_s.

    Returns
    -------
    (x, y) : post-repair integer positions, or None if no integer repair
        exists (bankruptcy: even full liquidation cannot cover the margin).
    """
    # 1:C1 — doc §9.1 precondition: prices must be positive. pa == 0 stays in
    # scope as the worthless-anchor degenerate case (handled below); only
    # strictly invalid prices are rejected.
    if pa < 0.0 or ps <= 0.0:
        raise ValueError("prices must be positive (doc §9.1)")
    E = total_state(qa, qs, pa, ps, C)
    denom = (mm_l - tcr) * pa

    # bankruptcy / solvency: E >= tcr*(E_a + |E_s|)  (full-liquidation cost test)
    budget = E - tcr * (qa * pa + abs(qs) * ps)
    if budget < 0.0:
        return None
    # 1:C1 — degenerate anchor leg at pa == 0: the anchor is worthless and
    # margin-free (any x loads zero margin and sells for zero), so maximal
    # retention x = qa is free; dividing by denom = 0 would crash instead.
    if pa == 0.0:
        x = qa
    else:
        # max anchor retention: feasibility <=> x <= budget/denom (see doc, Lemma 2/3)
        x = min(qa, math.floor(budget / denom + 1e-9))
    if x < 0:
        return None
    # stage 2: cheapest integer y (closest feasible integer to qs) for this x
    R = E - tcr * (qa - x) * pa - mm_l * x * pa      # budget for the secondary leg

    def hi_side():
        # max y >= 0 with mm_l*y*ps + tcr*|y-qs|*ps <= R   (piecewise linear, kinks 0, qs)
        y2 = (R + tcr * qs * ps) / ((mm_l + tcr) * ps)
        if qs > 0:
            y1 = (R - tcr * qs * ps) / ((mm_l - tcr) * ps)
            return y1 if y1 < qs else y2
        return y2

    def lo_side():
        # min y <= 0 with mm_s*|y|*ps + tcr*|y-qs|*ps <= R  (mirror of hi_side)
        y2 = (tcr * qs * ps - R) / ((mm_s + tcr) * ps)
        if qs < 0:
            y1 = -(R + tcr * qs * ps) / ((mm_s - tcr) * ps)
            return y1 if y1 > qs else y2
        return y2

    y = min(max(qs, math.ceil(lo_side() - 1e-9)), math.floor(hi_side() + 1e-9))
    # numerical safety walk toward the always-feasible y = 0
    while y != 0 and lhs_y(y, qs, ps, tcr, mm_l, mm_s) > R + 1e-7 * max(1.0, abs(R)):
        y += 1 if y < 0 else -1
    return x, y