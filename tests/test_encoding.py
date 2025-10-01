import numpy as np
import json
from evga.encoding import decode_solution


def load_data():
    with open("data/sample_data.json") as f:
        return json.load(f)


def test_decode_solution():
    data = load_data()
    solution = np.array([1, 0, 2])  # 1 outlet at site 0, 2 outlets at site 2
    decoded = decode_solution(solution, data)

    assert isinstance(decoded, dict)
    assert "sites" in decoded
    assert "served" in decoded
    assert decoded["sites"][0]["outlets"] == 1
    assert decoded["sites"][2]["outlets"] == 2