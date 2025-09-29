import requests
import pytest
from hello_portel_single_value_postman import app



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
        "market_id": 1,
        "created_at": 42041.9335300926,
        "actual_delivery_time": 42041.9772685185,
        "store_id": 2,
        "store_primary_category": 3,
        "order_protocol": 1,
        "total_items": 5,
        "subtotal": 5000,
        "num_distinct_items": 3,
        "min_item_price": 100,
        "max_item_price": 200,
        "total_onshift_partners": 50,
        "total_busy_partners": 10,
        "total_outstanding_orders": 5,
        "diff": "01:02:59",
        "weekday": 5
        }

    
    # Sending data as query parameters for GET request
    resp = client.post("/submit", query_string=test_home)
    
    # Assert that the response status code is 200 (OK)
    assert resp.status_code == 200

