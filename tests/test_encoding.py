import numpy as np
import json
from evga.encoding import decode_solution


def load_data():
    with open("data/sample_data_leplateaumontroyal.json") as f:
        return json.load(f)


def test_decode_solution():
    data = load_data()
    solution = np.array([1, 0, 2])  # 1 outlet at site 0, 2 outlets at site 2
    decoded = decode_solution(solution, data)

    # Check structure
    assert isinstance(decoded, dict)
    assert "sites" in decoded
    assert "total_outlets" in decoded

    # Check site-level details
    assert decoded["sites"][0]["outlets"] == 1
    assert decoded["sites"][2]["outlets"] == 2

    # Check total outlets
    assert decoded["total_outlets"] == 3
