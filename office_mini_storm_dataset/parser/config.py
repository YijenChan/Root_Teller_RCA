# config.py (anonymized & open-source safe version)

TRACE_PARSER_CONFIG = {
    'http_methods': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    'status_codes': ['STATUS_CODE_OK', 'STATUS_CODE_ERROR', 'STATUS_CODE_UNSET'],
    'db_operations': ['SELECT', 'INSERT', 'UPDATE', 'DELETE'],

    # Use alias names instead of real service names
    'services': ['S1', 'S2', 'S3', 'S4'],

    # Representative trace paths only — all generic and non-sensitive
    'trace_paths': [
        '/favicon.ico', '/robots.txt', '/sitemap.xml',
        '/cgi/test', '/cgi/env',
        '/db/query', '/api/query',
        '/uploads/temp', '/logs/access.log',
        '/api/slow', '/login', '/auth',
        '/index', '/api/endpoint', '/search', '/product',
        '/category', '/user', '/download', '/file', '/image', '/document'
    ],

    # Common suspicious keywords for detecting anomalies
    'suspicious_patterns': ["OR", "UNION", "SELECT", "--", "../", "script", "eval"],

    # Relative file paths for portability and safety
    'input_file': 'data/raw_data/fluent_trace.log',
    'output_pkl': 'data/dataset/trace_graphs.pkl',
    'output_npy': 'data/dataset/trace_features.npy'
}

GAT_CONFIG = {
    'hidden_dim': 64,
    'output_dim': 32,
    'num_heads': 4,
    'num_layers': 3,

    'input_pkl': 'data/dataset/trace_graphs.pkl',
    'output_txt': 'outputs/trace_node_embeddings.txt',
    'output_npy': 'outputs/trace_node_embeddings.npy',
    'model_path': 'outputs/trace_node_encoder.pt'
}

METRICS_PARSER_CONFIG = {
    'input_file': 'data/raw_data/metrics.log',
    'output_file': 'data/dataset/fluent_metrics.csv'
}
