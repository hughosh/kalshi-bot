"""Analytical hierarchical Markov chain for tennis win probability.

The model follows the Newton & Keller (2005) framework:
    point → game → set → match

All functions are pure and stateless. The core insight is that the probability
of winning a tennis match can be expressed analytically (or via small DP) as a
function of a single parameter p — the server's probability of winning any given
point — plus the current score state.

References:
    Newton, P.K. & Keller, J.B. (2005). "Probability of winning at tennis I.
    Theory and data." Studies in Applied Mathematics, 114(3), 241-269.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

from tennis.types import MatchFormat, MatchState


# ---------------------------------------------------------------------------
# Point level
# ---------------------------------------------------------------------------

def game_win_prob(p: float) -> float:
    """P(server wins a standard deuce-advantage game) given point win prob p.

    Derivation:
        P(win 4-0) = p^4
        P(win 4-1) = C(4,1) * p^4 * (1-p)   [lose exactly 1 of first 4]
        P(win 4-2) = C(5,2) * p^4 * (1-p)^2  [lose exactly 2 of first 5]
        P(reach deuce) = C(6,3) * p^3 * (1-p)^3
        P(win from deuce) = p^2 / (p^2 + (1-p)^2)  [geometric series]
    """
    p = max(1e-9, min(float(p), 1.0 - 1e-9))  # fast scalar clip
    q = 1.0 - p
    no_deuce = p**4 + 4.0 * p**4 * q + 10.0 * p**4 * q**2
    p_reach_deuce = 20.0 * p**3 * q**3
    p_win_deuce = p**2 / (p**2 + q**2)
    return no_deuce + p_reach_deuce * p_win_deuce


def game_win_prob_from_score(p: float, pts_server: int, pts_returner: int) -> float:
    """P(server wins game) given current score (pts_server, pts_returner).

    Handles standard scoring including deuce/advantage.
    Scores above 3 are treated as in the deuce-advantage phase.
    """
    p = max(1e-9, min(float(p), 1.0 - 1e-9))
    q = 1.0 - p

    s, r = pts_server, pts_returner

    # Already won or lost
    if s >= 4 and s - r >= 2:
        return 1.0
    if r >= 4 and r - s >= 2:
        return 0.0

    # Deuce phase: both at 3+ and equal or within 1
    if s >= 3 and r >= 3:
        # Normalize to (0,0) or (1,0) or (0,1) relative to deuce
        adv = s - r  # +1 = server advantage, -1 = returner advantage, 0 = deuce
        if adv == 0:
            return p**2 / (p**2 + q**2)
        elif adv == 1:
            return p
        else:  # adv == -1
            return p * (p**2 / (p**2 + q**2))

    # Pre-deuce: use DP over remaining states
    # dp[i][j] = P(server wins game | server has i pts, returner has j pts)
    # We cache this small computation
    return _game_dp(p, s, r)


def _game_dp(p: float, s: int, r: int) -> float:
    """DP for game win probability. States: (server_pts, returner_pts), 0-3 each."""
    q = 1.0 - p
    # Memoised via dict for this call (no cross-call caching since p varies)
    memo: dict = {}

    def dp(i: int, j: int) -> float:
        if i >= 4 and i - j >= 2:
            return 1.0
        if j >= 4 and j - i >= 2:
            return 0.0
        if i >= 3 and j >= 3:
            adv = i - j
            if adv == 0:
                return p**2 / (p**2 + q**2)
            elif adv == 1:
                return p
            else:
                return p * (p**2 / (p**2 + q**2))
        if (i, j) in memo:
            return memo[(i, j)]
        val = p * dp(i + 1, j) + q * dp(i, j + 1)
        memo[(i, j)] = val
        return val

    return dp(s, r)


# ---------------------------------------------------------------------------
# Tiebreak level
# ---------------------------------------------------------------------------

def tiebreak_win_prob(p_odd: float, p_even: float | None = None) -> float:
    """P(first server wins tiebreak to 7, 2-point margin).

    In a tiebreak the first server serves 1 point, then players alternate every
    2 points. By convention:
        p_odd  = P(current server wins a point when they are serving)
        p_even = P(current server wins a point when opponent is serving)
               = 1 - P(opponent wins a point on their serve)

    If p_even is None we assume p_even = 1 - p_odd (symmetric).
    """
    p1 = max(1e-9, min(float(p_odd), 1.0 - 1e-9))
    p2 = max(1e-9, min(float(p_even) if p_even is not None else 1.0 - p1, 1.0 - 1e-9))
    return _tiebreak_dp(p1, p2)


def tiebreak_win_prob_from_score(
    p_odd: float,
    pts_server: int,
    pts_returner: int,
    p_even: float | None = None,
) -> float:
    """P(first server wins tiebreak) from current tiebreak score."""
    p1 = max(1e-9, min(float(p_odd), 1.0 - 1e-9))
    p2 = max(1e-9, min(float(p_even) if p_even is not None else 1.0 - p1, 1.0 - 1e-9))

    s, r = pts_server, pts_returner

    if s >= 7 and s - r >= 2:
        return 1.0
    if r >= 7 and r - s >= 2:
        return 0.0

    return _tiebreak_dp_from(p1, p2, s, r)


def _tiebreak_dp(p1: float, p2: float) -> float:
    return _tiebreak_dp_from(p1, p2, 0, 0)


def _tiebreak_dp_from(p1: float, p2: float, s: int, r: int) -> float:
    """DP solver for tiebreak from score (s, r).

    Service alternation: total points played so far = s + r.
    Point 0 is served by the 'first server' (p1 serves).
    Points 1-2 by opponent, 3-4 by first server, etc.
    So: first server serves on points where total_prev % 4 ∈ {0, 3}
        i.e., (s + r) % 4 in {0, 3}
    """
    memo: dict = {}

    def dp(i: int, j: int) -> float:
        if i >= 7 and i - j >= 2:
            return 1.0
        if j >= 7 and j - i >= 2:
            return 0.0
        if i >= 6 and j >= 6:
            # Mini deuce phase
            diff = i - j
            if diff == 0:
                # Symmetric: if p1 = 1-p2 this simplifies, but use general formula
                # P(first server wins from deuce) — solve 2-state chain
                # State: deuce. First serve goes to whoever's turn it is.
                return _tiebreak_deuce(p1, p2, i + j)
            elif diff == 1:
                serves_p1 = ((i + j) % 4) in (0, 3)
                p_win = p1 if serves_p1 else p2
                return p_win
            else:  # diff == -1
                serves_p1 = ((i + j) % 4) in (0, 3)
                p_win = p1 if serves_p1 else p2
                # first server is at disadvantage
                # need to win next 2 points (with alternating serve)
                p_next = p1 if serves_p1 else p2
                serves_p1_after = (((i + j) + 1) % 4) in (0, 3)
                p_after = p1 if serves_p1_after else p2
                return p_next * p_after  # must win both

        if (i, j) in memo:
            return memo[(i, j)]

        total = i + j
        serves_p1 = (total % 4) in (0, 3)
        p_win = p1 if serves_p1 else p2
        val = p_win * dp(i + 1, j) + (1.0 - p_win) * dp(i, j + 1)
        memo[(i, j)] = val
        return val

    return dp(s, r)


def _tiebreak_deuce(p1: float, p2: float, total_so_far: int) -> float:
    """P(first server wins from tiebreak deuce 6-6).

    The next two points are served by whoever's turn it is.
    Let A = P(first server wins next point when they serve),
        B = P(first server wins next point when opponent serves).
    """
    serves_p1_first = (total_so_far % 4) in (0, 3)
    serves_p1_second = ((total_so_far + 1) % 4) in (0, 3)
    a = p1 if serves_p1_first else p2
    b = p1 if serves_p1_second else p2
    # P(win both) = a*b; P(lose both) = (1-a)*(1-b); P(deuce again) = a*(1-b) + (1-a)*b
    p_win_2 = a * b
    p_lose_2 = (1.0 - a) * (1.0 - b)
    p_deuce_again = 1.0 - p_win_2 - p_lose_2
    # Geometric series: P(win) = p_win_2 / (1 - p_deuce_again) but need to be careful
    # that "deuce again" uses the same total_so_far+2 service rotation
    # For simplicity and since total_so_far+2 has the same parity as total_so_far mod 4
    # (because we add 2), we can use the same formula recursively — which is the same
    # state. So:
    if p_deuce_again >= 1.0 - 1e-9:
        return 0.5
    return p_win_2 / (p_win_2 + p_lose_2)


# ---------------------------------------------------------------------------
# Set level
# ---------------------------------------------------------------------------

def set_win_prob(
    g: float,
    g_opp: float | None = None,
    match_format: MatchFormat | None = None,
    is_final_set: bool = False,
) -> float:
    """P(server wins set) given game win probabilities.

    Args:
        g: P(server wins a game when they serve)
        g_opp: P(server wins a game when opponent serves) = 1 - P(opp wins game on serve)
               If None, assume symmetric: g_opp = 1 - g
        match_format: used to determine final set rules
        is_final_set: if True, apply final-set rules from match_format
    """
    g = float(np.clip(g, 1e-9, 1 - 1e-9))
    if g_opp is None:
        g_opp = 1.0 - g
    g_opp = float(np.clip(g_opp, 1e-9, 1 - 1e-9))

    if match_format is None:
        match_format = MatchFormat.best_of_3_tiebreak()

    final_set_tb = match_format.final_set_tiebreak
    final_set_adv = match_format.final_set_advantage

    return _set_dp(g, g_opp, 0, 0, is_final_set, final_set_tb, final_set_adv)


def set_win_prob_from_score(
    g: float,
    games_server: int,
    games_returner: int,
    match_format: MatchFormat,
    is_final_set: bool,
    g_opp: float | None = None,
) -> float:
    """P(server wins set) from current game score."""
    g = float(np.clip(g, 1e-9, 1 - 1e-9))
    if g_opp is None:
        g_opp = 1.0 - g
    g_opp = float(np.clip(g_opp, 1e-9, 1 - 1e-9))
    return _set_dp(
        g, g_opp,
        games_server, games_returner,
        is_final_set,
        match_format.final_set_tiebreak,
        match_format.final_set_advantage,
    )


def _set_dp(
    g: float,
    g_opp: float,
    s: int,
    r: int,
    is_final_set: bool,
    final_set_tb: bool,
    final_set_adv: bool,
) -> float:
    """DP over game states for a single set."""
    memo: dict = {}

    # At 6-6 what happens?
    # - regular set: tiebreak (tb_win = tiebreak_win_prob using game-level p)
    # - final set w/ advantage: keep playing until 2 game lead
    # - final set w/ tiebreak: tiebreak
    use_tiebreak_at_66 = (not is_final_set) or (is_final_set and final_set_tb)
    use_advantage_final = is_final_set and final_set_adv

    def dp(i: int, j: int) -> float:
        # Winning condition
        if i >= 6 and i - j >= 2:
            return 1.0
        if j >= 6 and j - i >= 2:
            return 0.0

        # Tiebreak at 6-6 (for applicable sets)
        if i == 6 and j == 6:
            if use_tiebreak_at_66:
                # Use game win probs as proxy for tiebreak point win prob
                # More precisely: p_serve at tiebreak ≈ original p_serve
                # We use g (game win prob) to back out approximate point win prob
                # but for simplicity, treat tiebreak as a 2-point game
                # The correct approach: pass p through. We use g as approximation.
                # g ≈ game_win_prob(p), but inverting is complex; use tb directly.
                # For set-level DP we treat the tiebreak as a single "game" with
                # win prob = tiebreak_win_prob(p_tb) where p_tb ≈ implied from g.
                # Approximate: p_tb ≈ invert game_win_prob(g) — done numerically.
                p_tb = _invert_game_win_prob(g)
                return tiebreak_win_prob(p_tb)
            elif use_advantage_final:
                # Advantage set continues indefinitely
                # P(win from 6-6 advantage) = g*g_opp-solved geometric
                # Service alternates each game. At 6-6, odd total → server serves.
                # Actually we treat each game independently with alternating serve.
                # From 6-6: solve 2-outcome chain (win by 2)
                return _advantage_set_from_66(g, g_opp)

        if (i, j) in memo:
            return memo[(i, j)]

        # Who serves? In real tennis serve alternates each game.
        # Total games played so far = i + j.
        # In a set the first server serves first game, then alternates.
        # P(server wins current game) depends on parity.
        total = i + j
        server_serves = (total % 2 == 0)  # server serves on even game counts
        p_win_game = g if server_serves else g_opp

        val = p_win_game * dp(i + 1, j) + (1.0 - p_win_game) * dp(i, j + 1)
        memo[(i, j)] = val
        return val

    return dp(s, r)


def _advantage_set_from_66(g: float, g_opp: float) -> float:
    """P(server wins advantage set from 6-6 score)."""
    # At 6-6: server serves (12 games played, even total → server serves).
    # Win condition: lead by 2. Solve the 3-state Markov chain:
    #   States: 0=tied, +1=server ahead by 1, -1=server behind by 1
    # From state 0 (tied, server serves): P(win game) = g
    # From state +1 (server ahead, opponent serves): P(win game) = g_opp
    # From state -1 (server behind, server serves): P(win game) = g
    # Let x0, x1, xm1 = P(win from each state)
    # x1 = g_opp + (1-g_opp)*xm1... wait let me think carefully.
    # At state "tied" (even total games), server serves → g wins game → go to +1
    # At state "+1" (odd total), opponent serves → g_opp wins game → win; else → tied
    # At state "-1" (odd total), server serves... wait, depends on parity.
    # Let's just enumerate. From 6-6 (12 total, server serves):
    # p_s = g (server serves), p_r = g_opp (when opp serves)
    # x = P(win from deuce-like state where server serves)
    # y = P(win from deuce-like state where opp serves)
    # x = g*1 + (1-g)*y'   [win game → win set; lose game → server is behind, opp serves]
    # y = g_opp*1 + (1-g_opp)*x  [win game → win set? No, we need 2-game lead]
    # Need to be more careful. Let's enumerate all states.
    # State = (games ahead): ..., -2, -1, 0, +1, +2, ...
    # From state 0, server serves: win→+1, lose→-1
    # From state +1, opp serves: win(from server's view=g_opp)→win set; lose→0
    # From state -1, server serves: win→0; lose→lose set
    # So: Let x = P(win | state 0, server serves)
    # x = g * (g_opp * 1 + (1-g_opp) * x) + (1-g) * (g * x + (1-g) * 0)
    # x = g*g_opp + g*(1-g_opp)*x + (1-g)*g*x
    # x = g*g_opp + x*(g*(1-g_opp) + (1-g)*g)
    # x = g*g_opp + x*g*(1 - g_opp + 1 - g)
    # x*(1 - g*(2 - g - g_opp)) = g*g_opp
    # x = g*g_opp / (1 - g*(2 - g - g_opp))
    denom = 1.0 - g * (2.0 - g - g_opp)
    if abs(denom) < 1e-12:
        return 0.5
    return g * g_opp / denom


def _invert_game_win_prob(g: float, n_iter: int = 30) -> float:
    """Numerically invert game_win_prob to find p given g = game_win_prob(p)."""
    # Binary search on [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if game_win_prob(mid) < g:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Match level
# ---------------------------------------------------------------------------

def match_win_prob(state: MatchState, p_serve: float) -> float:
    """P(player 1 wins match) from current match state.

    Args:
        state: Full match state including current score and format.
        p_serve: P(current server wins a point on serve). This is the key
                 parameter estimated by the GBDT model and Bayesian updater.

    Returns:
        Probability that player 1 wins the match, in [0, 1].

    The calculation propagates p_serve through the hierarchy:
        point → game → set → match
    using the current score as the starting state.
    """
    p_serve = max(1e-9, min(float(p_serve), 1.0 - 1e-9))
    fmt = state.match_format

    # Compute game-level win probabilities from point-level
    # g_server = P(current server wins a game when they serve)
    g_server = game_win_prob(p_serve)
    # g_server_opp = P(current server wins a game when opponent serves)
    # We use symmetry: p_return ≈ 1 - p_serve
    p_return = 1.0 - p_serve
    g_server_opp = game_win_prob(p_return)

    # Who is player 1 relative to current server?
    if state.server == 0:
        g_p1_serving = g_server
        g_p1_returning = g_server_opp
    else:
        g_p1_serving = g_server_opp
        g_p1_returning = g_server

    # Pre-compute P(p1 wins tiebreak) once — passed to all set DPs to
    # avoid repeated _invert_game_win_prob calls inside the DP (major speedup).
    # Approximate: p_serve for tiebreak ≈ original p_serve for p1 when serving.
    _p_tb_p1 = tiebreak_win_prob(p_serve if state.server == 0 else p_return)

    # ---------- Step 1: P(win current game) ----------
    if state.is_tiebreak:
        p_tb = p_serve  # use point win prob directly
        p_win_game_server = tiebreak_win_prob_from_score(
            p_tb, state.points_p1 if state.server == 0 else state.points_p2,
            state.points_p2 if state.server == 0 else state.points_p1,
        )
        # P(player 1 wins current tiebreak game)
        p_win_cur_game_p1 = p_win_game_server if state.server == 0 else (1.0 - p_win_game_server)
    else:
        if state.server == 0:
            s_pts, r_pts = state.points_p1, state.points_p2
        else:
            s_pts, r_pts = state.points_p2, state.points_p1
        p_win_cur_game_server = game_win_prob_from_score(p_serve, s_pts, r_pts)
        p_win_cur_game_p1 = p_win_cur_game_server if state.server == 0 else (1.0 - p_win_cur_game_server)

    # ---------- Step 2: P(win current set) ----------
    # We need P(p1 wins current set) given current game score + p_win_cur_game_p1.
    # We compute this as: P(win set from current games if p1 wins current game) * P(p1 wins cur game)
    #                   + P(win set from current games if p1 loses current game) * P(p1 loses cur game)
    is_fs = state.is_final_set()

    if state.server == 0:
        # p1 is server
        g1 = g_p1_serving    # p1 wins game when serving
        g2 = g_p1_returning  # p1 wins game when returning
    else:
        g1 = g_p1_returning  # p1 wins game when returning (p2 is server)
        g2 = g_p1_serving    # p1 wins game when p1 serves

    # Current game score
    gs1, gs2 = state.games_p1, state.games_p2
    total_games_so_far = gs1 + gs2

    if state.is_tiebreak:
        # The tiebreak IS the set-deciding game: winning it wins the set (7-6),
        # losing it loses the set (6-7). No further set DP needed.
        p_win_set_if_win_game = 1.0
        p_win_set_if_lose_game = 0.0
    else:
        # After current game finishes, p1 either has gs1+1 or gs1 games.
        p_win_set_if_win_game = _set_dp_p1(
            g1, g2, gs1 + 1, gs2,
            is_fs, fmt.final_set_tiebreak, fmt.final_set_advantage,
            total_games_so_far + 1, _p_tb_p1,
        )
        p_win_set_if_lose_game = _set_dp_p1(
            g1, g2, gs1, gs2 + 1,
            is_fs, fmt.final_set_tiebreak, fmt.final_set_advantage,
            total_games_so_far + 1, _p_tb_p1,
        )

    p_win_cur_set_p1 = (
        p_win_cur_game_p1 * p_win_set_if_win_game
        + (1.0 - p_win_cur_game_p1) * p_win_set_if_lose_game
    )

    # ---------- Step 3: P(win match) ----------
    ss1, ss2 = state.sets_p1, state.sets_p2
    sets_needed = state.sets_needed()

    p_win_match_if_win_set = _match_dp_p1(
        g1, g2, ss1 + 1, ss2, fmt, sets_needed, total_games_so_far + 1, _p_tb_p1
    )
    p_win_match_if_lose_set = _match_dp_p1(
        g1, g2, ss1, ss2 + 1, fmt, sets_needed, total_games_so_far + 1, _p_tb_p1
    )

    return (
        p_win_cur_set_p1 * p_win_match_if_win_set
        + (1.0 - p_win_cur_set_p1) * p_win_match_if_lose_set
    )


def _set_dp_p1(
    g1: float,  # P(p1 wins game when p1 serves)
    g2: float,  # P(p1 wins game when p2 serves)
    s: int,
    r: int,
    is_final_set: bool,
    final_set_tb: bool,
    final_set_adv: bool,
    total_games_played: int,
    p_tb_p1: float = 0.5,  # pre-computed P(p1 wins tiebreak); avoids inversion
) -> float:
    """P(player 1 wins set) from (s, r) game score.

    total_games_played: total games played in set so far INCLUDING the just-finished game.
    Used to determine who serves each subsequent game.

    p_tb_p1: P(player 1 wins the tiebreak at 6-6), pre-computed by the caller.
             Passing this avoids repeated calls to _invert_game_win_prob.
    """
    if s >= 6 and s - r >= 2:
        return 1.0
    if r >= 6 and r - s >= 2:
        return 0.0

    memo: dict = {}
    use_tb_66 = (not is_final_set) or (is_final_set and final_set_tb)
    use_adv_final = is_final_set and final_set_adv

    def dp(i: int, j: int, games_done: int) -> float:
        if i >= 6 and i - j >= 2:
            return 1.0
        if j >= 6 and j - i >= 2:
            return 0.0

        if i >= 6 and j >= 6 and i == j:
            # Tied at 6-6 or beyond (only reachable in advantage final sets for > 6-6)
            if use_tb_66:
                return p_tb_p1  # use pre-computed value — no inversion needed
            elif use_adv_final:
                # Tied state at 6+ in advantage final set.
                p1_serves = (games_done % 2 == 0)
                ga = g1 if p1_serves else g2
                gb = g2 if p1_serves else g1
                return _advantage_set_from_66_p1(ga, gb)
            else:
                pass  # fall through to normal DP

        key = (i, j, games_done % 2)
        if key in memo:
            return memo[key]

        p1_serves = (games_done % 2 == 0)
        p_win = g1 if p1_serves else g2
        val = p_win * dp(i + 1, j, games_done + 1) + (1.0 - p_win) * dp(i, j + 1, games_done + 1)
        memo[key] = val
        return val

    return dp(s, r, total_games_played)


def _advantage_set_from_66_p1(g_p1_serves: float, g_p1_returns: float) -> float:
    """P(player 1 wins advantage final set from 6-6), p1 serves first."""
    # States: 0=tied(p1 serves), 1=p1+1(p2 serves), -1=p1-1(p2 serves → p1 serves next)
    # More precisely track who serves.
    # Let x = P(win | tied, p1 serves)
    # x = g_p1_serves * y + (1-g_p1_serves) * z
    # y = P(win | p1 ahead 1, p2 serves) = g_p1_returns * 1 + (1-g_p1_returns) * x
    # z = P(win | p1 behind 1, p2 serves) = g_p1_returns * x + (1-g_p1_returns) * 0
    # Substituting:
    # y = g_p1_returns + (1-g_p1_returns) * x
    # z = g_p1_returns * x
    # x = g_p1_serves * (g_p1_returns + (1-g_p1_returns)*x) + (1-g_p1_serves)*(g_p1_returns*x)
    # x = g_p1_serves*g_p1_returns + g_p1_serves*(1-g_p1_returns)*x + (1-g_p1_serves)*g_p1_returns*x
    # x*(1 - g_p1_serves*(1-g_p1_returns) - (1-g_p1_serves)*g_p1_returns) = g_p1_serves*g_p1_returns
    coeff = g_p1_serves * (1.0 - g_p1_returns) + (1.0 - g_p1_serves) * g_p1_returns
    denom = 1.0 - coeff
    if abs(denom) < 1e-12:
        return 0.5
    return g_p1_serves * g_p1_returns / denom


def _match_dp_p1(
    g1: float,
    g2: float,
    ss1: int,
    ss2: int,
    fmt: MatchFormat,
    sets_needed: int,
    total_games_played: int,
    p_tb_p1: float = 0.5,
) -> float:
    """P(player 1 wins match) from current set score (ss1, ss2)."""
    if ss1 >= sets_needed:
        return 1.0
    if ss2 >= sets_needed:
        return 0.0

    memo: dict = {}

    def dp(i: int, j: int, games_done: int) -> float:
        if i >= sets_needed:
            return 1.0
        if j >= sets_needed:
            return 0.0

        key = (i, j)
        if key in memo:
            return memo[key]

        is_fs = (i + j) == fmt.best_of - 1
        p_win_set = _set_dp_p1(
            g1, g2, 0, 0, is_fs,
            fmt.final_set_tiebreak, fmt.final_set_advantage,
            games_done, p_tb_p1,
        )
        val = p_win_set * dp(i + 1, j, games_done) + (1.0 - p_win_set) * dp(i, j + 1, games_done)
        memo[key] = val
        return val

    return dp(ss1, ss2, total_games_played)


# ---------------------------------------------------------------------------
# Vectorized batch version for Monte Carlo sampling
# ---------------------------------------------------------------------------

def match_win_prob_batch(state: MatchState, p_serves: np.ndarray) -> np.ndarray:
    """Vectorized match_win_prob for an array of p_serve samples.

    Used by the engine's Monte Carlo confidence interval calculation.
    Falls back to a loop for correctness; the DP per call is tiny so this
    is still fast for N=500.
    """
    return np.array([match_win_prob(state, float(p)) for p in p_serves])
