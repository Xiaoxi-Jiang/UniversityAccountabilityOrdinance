import json

from src.data.context.rentsmart import build_query_payload, parse_powerbi_response


def test_build_query_payload_includes_restart_tokens_when_present():
    payload = build_query_payload(
        ["full_address", "date"],
        page_size=50000,
        restart_tokens=[["'12 Main St'", "datetime'2024-01-01T00:00:00'"]],
    )

    window = payload["queries"][0]["Query"]["Commands"][0]["SemanticQueryDataShapeCommand"]["Binding"]["DataReduction"]["Primary"]["Window"]
    assert window["Count"] == 50000
    assert window["RestartTokens"] == [["'12 Main St'", "datetime'2024-01-01T00:00:00'"]]


def test_parse_powerbi_response_unpacks_rows_and_converts_dates():
    response = {
        "results": [
            {
                "result": {
                    "data": {
                        "descriptor": {
                            "Select": [
                                {"Value": "G0", "Name": "rentsmart.full_address"},
                                {"Value": "G1", "Name": "rentsmart.date", "Format": "MM/dd/yyyy"},
                            ]
                        },
                        "metrics": {"Events": [{"Metrics": {"RowCount": 2}}]},
                        "dsr": {
                            "DS": [
                                {
                                    "PH": [
                                        {
                                            "DM0": [
                                                {
                                                    "S": [
                                                        {"N": "G0", "T": 1, "DN": "D0"},
                                                        {"N": "G1", "T": 7},
                                                    ],
                                                    "C": [0, 1704067200000],
                                                },
                                                {
                                                    "C": [1],
                                                    "R": 2,
                                                },
                                            ]
                                        }
                                    ],
                                    "ValueDicts": {"D0": ["12 Main St", "13 Main St"]},
                                    "IC": False,
                                    "RT": [["'13 Main St'", "datetime'2024-01-01T00:00:00'"]],
                                }
                            ]
                        },
                    }
                }
            }
        ]
    }

    parsed = parse_powerbi_response(json.loads(json.dumps(response)))

    assert parsed["row_count"] == 2
    assert parsed["is_complete"] is False
    assert parsed["restart_tokens"] == [["'13 Main St'", "datetime'2024-01-01T00:00:00'"]]
    assert parsed["rows"] == [
        {"rentsmart.full_address": "12 Main St", "rentsmart.date": "2024-01-01"},
        {"rentsmart.full_address": "13 Main St", "rentsmart.date": "2024-01-01"},
    ]
