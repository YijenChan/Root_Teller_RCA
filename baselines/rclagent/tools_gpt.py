print_result_function = {
    "type": "function",
    "function": {
        "name": "print_results",
        "description": (
            "Output a list of at least TEN potential root cause identifiers. "
            "Each identifier is a string representing a pod (e.g., 'productcatalogservice-0'), "
            "a service (e.g., 'recommendationservice'), or a node (e.g., 'node-1').\n\n"
            "✅ Example output:\n"
            "{\n"
            '  "root_causes": [\n'
            '    "productcatalogservice-0",\n'
            '    "recommendationservice",\n'
            '    "node-1"\n'
            '    "shippingservice-1"\n'
            '    "paymentservice"\n'
            '    "currencyservice-0"\n'
            '    "node-2"\n'
            '    "emailservice"\n'
            '    "node-4"\n'
            '    "recommendationservice2-0"\n'
            '    "frontend-2"\n'
            "  ]\n"
            "}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root_causes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 10,
                    "description": "At least 10 candidate root causes as plain strings."
                }
            },
            "required": ["root_causes"]
        }
    }
}


search_logs_function = {
    "type": "function",
    "function": {
        "name": "search_logs",
        "description": "Retrieve logs for a given service around a specific timestamp(+-60s).",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The name of the service to be queried."
                },
                "timestamp": {
                    "type": "string",
                    "description": "The timestamp around which logs should be retrieved."
                }
            },
            "required": [
                "service_name",
                "timestamp"
            ]
        }
    }
}

search_fluctuating_metrics_function = {
    "type": "function",
    "function": {
        "name": "search_fluctuating_metrics",
        "description": "Get all fluctuating metrics with the input service_name and around the input timestamp",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The service_name to be queried."
                },
                "timestamp": {
                    "type": "string",
                    "description": "The timestamp to be queried."
                }
            },
            "required": [
                "service_name",
                "timestamp"
            ]
        }
    }
}
