import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_texts():
    return {
        "joy": "I am so happy and excited today!",
        "sadness": "I feel really sad and lonely",
        "anger": "This makes me absolutely furious!",
        "fear": "I'm scared about what might happen",
        "surprise": "Wow, I can't believe this!",
        "love": "I love you more than anything"
    }