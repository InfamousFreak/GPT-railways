import os
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY")
@pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="No MONGODB_URI")
def test_ask_smoke():
    r = client.post("/ask", json={"question":"Hello"})
    assert r.status_code == 200
    assert "answer" in r.json()
