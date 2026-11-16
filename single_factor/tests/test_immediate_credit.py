from single_factor.hierarchical.ppo import complete_returns, immediate_targets


def test_immediate_credit_keeps_actual_episode_returns_and_old_policy_evidence():
    rows = [{"reward": 0., "value": .3, "logp": -.7}, {"reward": 1.5, "value": .2, "logp": -.9}]
    assert complete_returns(rows) == 1.5
    immediate_targets(rows)
    assert [r["return"] for r in rows] == [1.5, 1.5]
    assert [r["value_target"] for r in rows] == [0., 1.5]
    assert [r["advantage"] for r in rows] == [-.3, 1.3]
    assert [r["logp"] for r in rows] == [-.7, -.9]
