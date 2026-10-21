from src.health.relationship_health import calculate_health, classify_health


def _row(recency, engagement, complaints, returns):
    return {
        "Recency": recency,
        "EngagementScore": engagement,
        "Complaints": complaints,
        "Returns": returns,
    }


def test_health_score_is_always_between_0_and_100():
    cases = [
        _row(0, 100, 0, 0),
        _row(400, 0, 10, 10),
        _row(50, 50, 2, 2),
    ]
    for row in cases:
        score = calculate_health(row, max_recency=400, max_friction=20)
        assert 0 <= score <= 100


def test_recent_engaged_low_friction_customer_scores_higher_than_the_opposite():
    good_customer = _row(recency=1, engagement=95, complaints=0, returns=0)
    bad_customer = _row(recency=350, engagement=5, complaints=8, returns=8)
    good_score = calculate_health(good_customer, max_recency=400, max_friction=20)
    bad_score = calculate_health(bad_customer, max_recency=400, max_friction=20)
    assert good_score > bad_score


def test_health_categories_are_one_of_the_three_valid_labels():
    for score in [0, 25, 39, 40, 55, 69, 70, 90, 100]:
        assert classify_health(score) in {"Healthy", "At Risk", "Critical"}


def test_health_category_boundaries_match_the_documented_thresholds():
    # HEALTHY_THRESHOLD = 70, AT_RISK_THRESHOLD = 40 in relationship_health.py
    assert classify_health(70) == "Healthy"
    assert classify_health(69.9) == "At Risk"
    assert classify_health(40) == "At Risk"
    assert classify_health(39.9) == "Critical"
