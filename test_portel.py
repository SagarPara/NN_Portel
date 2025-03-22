import requests
import pytest
from sales_prediction import app



#proxy to a live server
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client  # This ensures a clean test client for each test
#    return app.test_client

def test_home(client):
    resp = client.get("/")
    print(resp)
    assert resp.status_code == 200

def test_submit(client):
    test_home = {
        "Region_Code": "R1",
        "Location_Type": "L2",
        "Store_Type": "S1",
        "Holiday": 1,
        "Discount": "Yes",
        "Order": 20
        }
    
    resp = client.post("/submit", json=test_home)
    assert resp.status_code == 200

    
