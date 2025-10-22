# Office Mini-Storm: A Synthetic Microservice Log Dataset

## Overview

**Office Mini-Storm** is a synthetic multimodal log dataset designed to simulate a microservice-based application environment.  
It contains both normal and stress-test scenarios generated for research on anomaly detection, causal analysis, and distributed tracing.  
Data were collected via a Fluent Bit pipeline and include three synchronized modalities: application logs, system metrics, and distributed traces.

---

## Data Collection Architecture

### Fluent Bit Pipeline

- **OpenTelemetry Instrumentation** — Each service was instrumented with the OpenTelemetry SDK to generate distributed trace data.
- **Application Error Logs** — Fluent Bit collects structured error logs from a web server, capturing runtime exceptions and failures.
- **Cluster Metrics Monitoring** — Fluent Bit also gathers system-level metrics such as CPU, memory, disk, and swap utilization.
- **Unified Data Stream** — All data modalities are merged through a unified Fluent Bit channel and exported to local storage.

### Microservice Components (4 services)

| Service Name | Port | Description |
|---------------|-------|-------------|
| `frontend-service` | 8080 | Front-end service |
| `auth-service` | 8081 | Authentication service |
| `api-gateway` | 8082 | API gateway |
| `database-service` | 5432 | Database service |

---

## Log Modalities

### Three Log Types

1. **Error Logs** — Web server and application error messages  
2. **Metrics Logs** — System-level metrics (CPU, memory, disk, swap)  
3. **Trace Logs** — OpenTelemetry distributed trace data  

---

## Scenario Categories

> Note: Labels are anonymized and correspond to different operational or stress scenarios used for evaluation.  
> These are **synthetic** and not related to real network or security incidents.

### Normal Scenarios (Label 0 – 4)

| Scenario Name | Label | Description |
|----------------|--------|-------------|
| Resource Missing | 0 | Request returned “not found” or unavailable resource |
| Script Execution Error | 1 | Error during dynamic script execution |
| Database Connection Error | 2 | Database query or connection failure |
| Permission Restricted | 3 | Request denied due to access control |
| Request Timeout | 4 | Request exceeded response time threshold |

### Stress / Anomalous Scenarios (Label 5 – 8)

| Scenario Name | Label | Description |
|----------------|--------|-------------|
| Excessive Authentication Attempts | 5 | Repeated credential verification requests |
| High-Load Burst | 6 | Heavy concurrent request bursts |
| Abnormal Query Pattern | 7 | Irregular query parameter pattern |
| Directory Access Misuse | 8 | Accessing unexpected file paths |

---

## Dataset Statistics

| Version | Duration | Sampling Interval | Total Records | Combined Entries |
|----------|-----------|-------------------|----------------|------------------|
| 2-hour subset | 7,200 s | 5 s | 1,440 | ≈ 4,300 lines |
| 14-hour subset | 50,400 s | 5 s | 10,080 | ≈ 30,000 lines |

---

## File Descriptions

error.log - Web/application error log (JSON lines, collected by Fluent Bit)
metrics.log - System metrics log (JSON lines, collected by Fluent Bit)
trace.log - Distributed trace log (JSON lines, via OpenTelemetry pipeline)
metadata.json - Scenario metadata (label information)


### File Format Details

- **error.log** — One JSON record per line, including `timestamp` and `log` fields  
- **metrics.log** — One JSON record per line, including `timestamp` and a `metrics` object  
- **trace.log** — One JSON record per line, following the OpenTelemetry schema with `timestamp`, `trace_id`, and `spans`  
- **metadata.json** — A JSON array, each entry containing `timestamp`, `scenario`, and `label`

---

