import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess

from RCAEval.e2e.rcd import (
    _match_columns,
    add_fnode_and_concat,
    run_multi_phase,
)


F_NODE = "F-node"


def _rcd_multimodal(data, inject_time, dataset=None, gamma=5, localized=True, bins=5, verbose=False, **kwargs):
    """RCD variant that accepts multimodal dict input (metric + logts).
    Ported from cfm/e2e/rcd.py multimodal branch."""
    metric = data["metric"]
    logts = data["logts"]

    # === metric ===
    metric = metric.iloc[::15, :]

    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=False)
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=False)
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]

    normal_data = normal_metric
    anomal_data = anomal_metric

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])

    normal_data = pd.concat([normal_data, normal_logts], axis=1)
    anomal_data = pd.concat([anomal_data, anomal_logts], axis=1)

    normal_data = normal_data.loc[:, ~normal_data.columns.duplicated()]
    normal_data = normal_data.fillna(0)

    anomal_data = anomal_data.loc[:, ~anomal_data.columns.duplicated()]
    anomal_data = anomal_data.fillna(0)

    normal_df = drop_constant(convert_mem_mb(drop_time(normal_data)))
    anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_data)))

    normal_df, anomal_df = _match_columns(normal_df, anomal_df)

    df = add_fnode_and_concat(normal_df, anomal_df)
    normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    rc = run_multi_phase(normal_df, anomal_df, gamma, localized, bins, verbose)
    return {"ranks": rc}


def torai(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, normalize=True, addup=False, borda=False, service=None, fault_type=None, case=None, rank=None, **kwargs):
    scaler_function = kwargs.get("scaler_function", StandardScaler)

    metric = data["metric"]
    logts = data["logts"]
    traces_err = data.get("tracets_err", pd.DataFrame())
    traces_lat = data.get("tracets_lat", pd.DataFrame())

    has_traces = traces_err is not None and len(traces_err) > 0

    # ==== PREPARE DATA ====
    # the metric is sampled every second, resample for 15s
    metric = metric.iloc[::15, :]

    # == metric ==
    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])

    # == traces_err ==
    if has_traces:
        traces_err = traces_err.ffill()
        traces_err = traces_err.fillna(0)
        traces_err = drop_constant(traces_err)

        normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])

    # == traces_lat ==
    if has_traces:
        traces_lat = traces_lat.ffill()
        traces_lat = traces_lat.fillna(0)
        traces_lat = drop_constant(traces_lat)
        normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])

    # ==== PROCESS ====
    ranks = []
    metric_ranks = []
    log_ranks = []
    trace_err_ranks = []
    trace_lat_ranks = []

    # == metric ==
    for col in normal_metric.columns:
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        zscores = np.abs(zscores)
        score = max(zscores)
        metric_ranks.append((col, score))
    metric_ranks = sorted(metric_ranks, key=lambda x: x[1], reverse=True)
    metric_ranks = [(x[0], x[1] / sum([x[1] for x in metric_ranks])) for x in metric_ranks]
    ranks = metric_ranks.copy()

    # == logs ==
    for col in normal_logts.columns:
        a = normal_logts[col].to_numpy()
        b = anomal_logts[col].to_numpy()

        if a.size == 0:
            continue

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        zscores = np.abs(zscores)
        score = max(zscores)
        log_ranks.append((col, score))
    log_ranks = sorted(log_ranks, key=lambda x: x[1], reverse=True)
    log_ranks = [(x[0], x[1] / sum([x[1] for x in log_ranks])) for x in log_ranks]
    ranks.extend(log_ranks)

    # == traces_err ==
    if has_traces:
        for col in normal_traces_err.columns:
            a = normal_traces_err[col].to_numpy()[:-2]
            b = anomal_traces_err[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            zscores = np.abs(zscores)
            score = max(zscores)
            if scaler.mean_ == 0 and scaler.var_ == 0:
                score = score * 1e9
            trace_err_ranks.append((col, score))
        trace_err_ranks = sorted(trace_err_ranks, key=lambda x: x[1], reverse=True)
        trace_err_ranks = [(x[0], x[1] / sum([x[1] for x in trace_err_ranks])) for x in trace_err_ranks]
        ranks.extend(trace_err_ranks)

    # == traces_lat ==
    if has_traces:
        for col in normal_traces_lat.columns:
            a = normal_traces_lat[col].to_numpy()
            b = anomal_traces_lat[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            zscores = np.abs(zscores)
            score = max(zscores)
            trace_lat_ranks.append((col, score))
        trace_lat_ranks = sorted(trace_lat_ranks, key=lambda x: x[1], reverse=True)
        trace_lat_ranks = [(x[0], x[1] / sum([x[1] for x in trace_lat_ranks])) for x in trace_lat_ranks]
        ranks.extend(trace_lat_ranks)

    def fine2coarse_addup(fine_ranks):
        _coarse_ranks = [(i.split("_")[0], s) for i, s in fine_ranks]
        score_dict = {}
        for i, s in _coarse_ranks:
            if i in score_dict:
                score_dict[i] += s
            else:
                score_dict[i] = s
        coarse_ranks = [(i, s) for i, s in score_dict.items()]
        coarse_ranks = sorted(coarse_ranks, key=lambda x: x[1], reverse=True)
        return coarse_ranks

    def fine2coarse_highest(fine_ranks):
        if not fine_ranks:
            return []
        fine_ranks = sorted(fine_ranks, key=lambda x: x[1], reverse=True)
        _coarse_ranks = [(i.split("_")[0], s) for i, s in fine_ranks]

        coarse_ranks = [_coarse_ranks[0]]
        for svc, score in _coarse_ranks[1:]:
            if svc not in [i for i, _ in coarse_ranks]:
                coarse_ranks.append((svc, score))

        coarse_ranks = [(i, s / sum([s for _, s in coarse_ranks])) for i, s in coarse_ranks]
        return coarse_ranks

    svc_metric_ranks = fine2coarse_addup(metric_ranks)
    svc_log_ranks = fine2coarse_highest(log_ranks)
    svc_trace_lat_ranks = fine2coarse_addup(trace_lat_ranks)
    svc_trace_err_ranks = fine2coarse_addup(trace_err_ranks)

    score_dict = {}
    for i, s in svc_metric_ranks + svc_log_ranks:
        if i in score_dict:
            score_dict[i] += s
        else:
            score_dict[i] = s
    svc_ranks = [(i, s) for i, s in score_dict.items()]
    svc_ranks = sorted(svc_ranks, key=lambda x: x[1], reverse=True)

    # change service "frontendservice" to "frontend"
    svc_trace_lat_ranks = [("frontend", s) if i == "frontendservice" else (i, s) for i, s in svc_trace_lat_ranks]
    svc_trace_err_ranks = [("frontend", s) if i == "frontendservice" else (i, s) for i, s in svc_trace_err_ranks]

    m = {
        "metric": pd.Series([i[1] for i in svc_metric_ranks], index=[i[0] for i in svc_metric_ranks]),
        "log": pd.Series([i[1] for i in svc_log_ranks], index=[i[0] for i in svc_log_ranks]),
        "trace_lat": pd.Series([i[1] for i in svc_trace_lat_ranks], index=[i[0] for i in svc_trace_lat_ranks]),
        "trace_err": pd.Series([i[1] for i in svc_trace_err_ranks], index=[i[0] for i in svc_trace_err_ranks]),
    }

    m = pd.DataFrame(m)
    m = m.fillna(0)

    service_list = m.index.to_list()

    X_train = m.to_numpy()
    bic_score_all = []

    for n_comp in range(1, X_train.shape[0] + 1):
        estimator = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            max_iter=50, random_state=0
        )
        estimator.fit(X_train)
        bic_score = estimator.bic(X_train)
        bic_score_all.append(bic_score)
    idx_min = np.argmin(bic_score_all)
    n_comp_opt = idx_min + 1
    print("Optimal number of cluters: {}".format(n_comp_opt))

    estimator = GaussianMixture(
        n_components=n_comp_opt,
        covariance_type="full",
        max_iter=50, random_state=0
    )
    estimator.fit(X_train)
    y_pred = estimator.predict(X_train)
    y_train = np.mean(X_train, axis=1)

    cluster_rank = []
    for cluster_idx in list(set(y_pred)):
        service_indices_of_this_cluster = np.where(y_pred == cluster_idx)
        services_of_this_cluster = [service_list[i] for i in service_indices_of_this_cluster[0]]

        scores_of_them = y_train[service_indices_of_this_cluster]
        cluster_score = np.mean(scores_of_them)
        cluster_rank.append((cluster_idx, cluster_score))

    cluster_rank.sort(key=lambda x: x[1], reverse=True)

    service_ranks = []
    service_ranks_rcd = []
    for cluster_idx, score in cluster_rank:
        service_indices_of_this_cluster = np.where(y_pred == cluster_idx)
        services_of_this_cluster = [service_list[i] for i in service_indices_of_this_cluster[0]]
        scores_of_them = y_train[service_indices_of_this_cluster]

        if len(services_of_this_cluster) == 1:
            service_ranks.append(services_of_this_cluster[0])
            service_ranks_rcd.append(services_of_this_cluster[0])
            continue

        # sort by score within cluster
        aa = list(zip(services_of_this_cluster, scores_of_them))
        aa.sort(key=lambda x: x[1], reverse=True)
        for a, s in aa:
            service_ranks.append(a)

        # get metric subset for cluster
        tmp_metric = metric.loc[:, metric.columns.str.startswith(tuple(services_of_this_cluster))]
        tmp_metric["time"] = metric["time"]

        # get log subset for cluster
        tmp_logts = logts.loc[:, logts.columns.str.startswith(tuple(services_of_this_cluster))]
        tmp_logts["time"] = logts["time"]

        # get trace subsets for cluster
        tmp_traces_err = None
        tmp_traces_lat = None
        if has_traces:
            tmp_traces_err = traces_err.loc[:, traces_err.columns.str.startswith(tuple(services_of_this_cluster))]
            tmp_traces_err["time"] = traces_err["time"]

            tmp_traces_lat = traces_lat.loc[:, traces_lat.columns.str.startswith(tuple(services_of_this_cluster))]
            tmp_traces_lat["time"] = traces_lat["time"]

        tmp_ranks = _rcd_multimodal(
            data={
                "metric": tmp_metric,
                "logts": tmp_logts,
                "tracets_err": tmp_traces_err,
                "tracets_lat": tmp_traces_lat,
            },
            inject_time=inject_time,
            dataset=dataset,
        )
        tmp_ranks = [s.split("_")[0] for s in tmp_ranks["ranks"]]
        internal_service_ranks = []
        if tmp_ranks:
            internal_service_ranks = [tmp_ranks[0]]
            for s in tmp_ranks[1:]:
                if s not in internal_service_ranks:
                    internal_service_ranks.append(s)

        if len(internal_service_ranks) == len(services_of_this_cluster):
            service_ranks_rcd.extend(internal_service_ranks)
        else:
            for a, s in aa:
                service_ranks_rcd.append(a)

    ranks = [(f"{svc}_A", 1.0) for svc in service_ranks_rcd]

    output = {
        "ranks": [x[0] for x in ranks],
    }

    return output
