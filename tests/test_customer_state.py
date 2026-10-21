from src.state.customer_state import determine_customer_state, classify_churn_risk


def test_high_risk_customer_state_for_critical_health_and_high_churn():
    state = determine_customer_state(health="Critical", churn_probability=0.85)
    assert state == "Critical Retention"


def test_low_risk_customer_state_for_healthy_relationship_and_low_churn():
    state = determine_customer_state(health="Healthy", churn_probability=0.10)
    assert state == "Stable"


def test_hidden_risk_when_churn_is_high_but_relationship_looks_healthy():
    # This is the interesting/valuable case: the health score alone
    # would miss this customer, but churn probability catches them.
    state = determine_customer_state(health="Healthy", churn_probability=0.75)
    assert state == "Hidden Risk"


def test_classify_churn_risk_respects_the_documented_bands():
    assert classify_churn_risk(0.65) == "High"
    assert classify_churn_risk(0.60) == "High"
    assert classify_churn_risk(0.45) == "Medium"
    assert classify_churn_risk(0.40) == "Medium"
    assert classify_churn_risk(0.10) == "Low"
