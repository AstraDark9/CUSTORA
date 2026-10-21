from src.recommendations.retention_recommendations import (
    generate_recommendation,
    generate_recommendation_row,
)


def _row(state, recency=10, engagement=80, complaints=0, returns=0):
    return {
        "CustomerState": state,
        "Recency": recency,
        "EngagementScore": engagement,
        "Complaints": complaints,
        "Returns": returns,
    }


def test_recommendation_for_critical_customer_is_marked_urgent():
    action, description, priority, evidence = generate_recommendation(
        _row("Critical Retention", recency=200, engagement=20)
    )
    assert priority == "Urgent"
    assert action
    assert description
    assert evidence


def test_recommendation_for_healthy_stable_customer_is_low_priority():
    action, description, priority, evidence = generate_recommendation(
        _row("Stable", recency=5, engagement=90)
    )
    assert priority == "Low"


def test_critical_customer_with_many_complaints_gets_service_recovery():
    action, _, priority, evidence = generate_recommendation(
        _row("Critical Retention", recency=10, engagement=80, complaints=5)
    )
    assert action == "Service recovery"
    assert priority == "Urgent"
    assert "complaints" in evidence[0].lower()


def test_every_recommendation_returns_a_four_part_tuple_with_evidence():
    for state in ["Critical Retention", "High Risk", "Watch", "Stable", "Monitor"]:
        result = generate_recommendation(_row(state))
        assert len(result) == 4
        action, description, priority, evidence = result
        assert isinstance(action, str) and action
        assert isinstance(description, str) and description
        assert priority in {"Urgent", "High", "Medium", "Low"}
        assert isinstance(evidence, list) and len(evidence) >= 1


def test_more_matching_signals_gives_stronger_evidence():
    # Two signals fire (long recency AND low engagement) -> Strong evidence
    _, _, _, _, evidence_strength, signal_count = generate_recommendation_row(
        _row("Critical Retention", recency=200, engagement=20)
    )
    assert signal_count == 2
    assert evidence_strength == "Strong evidence"

    # Only the state itself, no extra signal -> Moderate evidence or State-based
    _, _, _, _, evidence_strength, signal_count = generate_recommendation_row(
        _row("Monitor", recency=10, engagement=90)
    )
    assert signal_count == 1
    assert evidence_strength in {"Moderate evidence", "State-based"}
