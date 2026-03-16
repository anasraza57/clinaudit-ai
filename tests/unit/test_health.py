"""Tests for health check endpoints."""


def test_health_check(client):
    """Health endpoint returns 200 with expected fields."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ClinAuditAI"
    assert "timestamp" in data


def test_readiness_check(client):
    """Readiness endpoint returns 200 with check details."""
    response = client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("ready", "degraded")
    assert "checks" in data
    assert "ai_provider" in data["checks"]
