import os

# ── LLM ──────────────────────────────────────────────────────────────────────
# RCLAgent calls any OpenAI-compatible chat completion endpoint.
# You MUST set LLM_API_KEY before running. LLM_API_URL and LLM_MODEL may be
# overridden to target a different provider (Claude, GPT-4, Llama, etc.).
LLM_API_URL      = os.environ.get("LLM_API_URL",  "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
LLM_API_KEY      = os.environ.get("LLM_API_KEY",  "")
LLM_MODEL        = os.environ.get("LLM_MODEL",    "qwen-plus")
LLM_FORCE_STREAM = os.environ.get("LLM_FORCE_STREAM", "").lower() in ("1", "true", "yes")

if not LLM_API_KEY:
    # Warn but do not abort — allows `python3 -c 'import config'` etc. in tooling.
    import warnings
    warnings.warn(
        "LLM_API_KEY is not set. Export it before running the coordinator: "
        "`export LLM_API_KEY=<your-key>`",
        RuntimeWarning,
    )

# ── Data ─────────────────────────────────────────────────────────────────────
# Root directory of the dataset under analysis.
# For quick verification, leave at the default `sample_data`.
# For a full AIOPS 2022 subset, e.g.: data/2022-03-21-cloudbed1
DATA_ROOT = os.environ.get("DATA_ROOT", "sample_data")

# Subdirectory (relative to DATA_ROOT) where results are written.
RESULT_SUB_DIR = os.environ.get("RESULT_SUB_DIR", "result")

# ── Tool server ──────────────────────────────────────────────────────────────
TOOL_SERVER_HOST = os.environ.get("TOOL_SERVER_HOST", "127.0.0.1")
TOOL_SERVER_PORT = int(os.environ.get("TOOL_SERVER_PORT", "5000"))
TOOL_SERVER_URL  = f"http://{TOOL_SERVER_HOST}:{TOOL_SERVER_PORT}"

# ── Agent ────────────────────────────────────────────────────────────────────
# Max concurrent Dedicated Agents. When running multiple subsets in parallel,
# keep this moderate to avoid LLM rate-limit errors.
MAX_AGENT_PARALLEL = int(os.environ.get("MAX_AGENT_PARALLEL", "32"))

# ── Dataset type ─────────────────────────────────────────────────────────────
# One of: "aiops2022" | "nezha" | "re2ob"
DATASET_TYPE = os.environ.get("DATASET_TYPE", "aiops2022")

# ── Preprocessing thresholds ─────────────────────────────────────────────────
# Duration above which a root span is classified as an error trace (µs).
DURATION_THRESHOLD_US = int(os.environ.get("DURATION_THRESHOLD_US", "10000000"))

# ── Per-dataset known services ───────────────────────────────────────────────
KNOWN_SERVICES = {
    "aiops2022": [
        "adservice", "adservice2", "cartservice", "cartservice2",
        "checkoutservice", "checkoutservice2", "currencyservice", "currencyservice2",
        "emailservice", "emailservice2", "frontend", "frontend2",
        "paymentservice", "paymentservice2", "productcatalogservice", "productcatalogservice2",
        "recommendationservice", "recommendationservice2",
        "shippingservice", "shippingservice2", "redis-cart",
    ],
    "nezha": [
        "ts-admin-basic-info-service", "ts-admin-order-service", "ts-admin-route-service",
        "ts-admin-travel-service", "ts-admin-user-service", "ts-assurance-service",
        "ts-auth-service", "ts-basic-service", "ts-config-service",
        "ts-contacts-service", "ts-delivery-service", "ts-execute-service",
        "ts-food-service", "ts-gateway-service", "ts-inside-payment-service",
        "ts-order-other-service", "ts-order-service", "ts-payment-service",
        "ts-preserve-other-service", "ts-preserve-service", "ts-price-service",
        "ts-route-service", "ts-seat-service", "ts-security-service",
        "ts-station-food-service", "ts-station-service", "ts-train-food-service",
        "ts-train-service", "ts-travel-service", "ts-travel2-service",
        "ts-user-service", "ts-verification-code-service",
    ],
    "re2ob": [
        "adservice", "cartservice", "checkoutservice", "currencyservice",
        "emailservice", "frontend", "frontendservice", "paymentservice",
        "productcatalogservice", "recommendationservice",
        "shippingservice", "redis",
    ],
}

def get_known_services():
    return KNOWN_SERVICES.get(DATASET_TYPE, KNOWN_SERVICES["aiops2022"])
