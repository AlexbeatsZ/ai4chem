"""Balanced endpoint-only training regions and an explicit geometric fallback."""

import numpy as np
from scipy.cluster.vq import kmeans2
from expert_routing import endpoint_features


def matrix(records, names=None):
    features = [endpoint_features(row) for row in records]
    names = names or list(features[0])
    return names, np.asarray([[row[name] for name in names] for row in features], float)


def standardized(values, medians, scales):
    return (np.where(np.isfinite(values), values, medians) - medians) / scales


def fit_gate(records, seed):
    if len(records) % 2 or len(records) < 4:
        raise ValueError("Two balanced regions require an even training pool")
    names, raw = matrix(records)
    medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any()
                        else 0. for column in raw.T])
    filled = np.where(np.isfinite(raw), raw, medians)
    scales = filled.std(axis=0)
    scales[scales < 1e-12] = 1.
    values = standardized(raw, medians, scales)
    centers, _ = kmeans2(values, 2, minit="++", seed=seed)
    previous = None
    for _ in range(100):
        distances = np.mean((values[:, None, :] - centers[None, :, :])**2, axis=2)
        order = np.argsort(distances[:, 0]-distances[:, 1], kind="stable")
        labels = np.ones(len(values), int)
        labels[order[:len(values)//2]] = 0
        centers = np.stack([values[labels == k].mean(axis=0) for k in (0, 1)])
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
    else:
        raise RuntimeError("Balanced endpoint clustering did not converge")
    distances = np.mean((values[:, None, :] - centers[None, :, :])**2, axis=2)
    delta = distances[:, 0]-distances[:, 1]
    sorted_delta = np.sort(delta)
    low, high = sorted_delta[len(values)//2-1:len(values)//2+1]
    if high <= low:
        raise ValueError("Geometrically identical boundary cases cannot be split using reaction identity")
    threshold = float((low+high)/2)
    predicted = (delta > threshold).astype(int)
    if not np.array_equal(predicted, labels):
        raise ValueError("Deployable endpoint boundary differs from training-region assignment")
    radii = [float(np.quantile(distances[labels == k, k], .99)) for k in (0, 1)]
    return {"feature_names": names, "medians": medians.tolist(), "scales": scales.tolist(),
            "centers": centers.tolist(), "boundary": threshold, "training_radius_q99": radii,
            "seed": seed, "training_reactions": len(records),
            "scope": "training endpoint geometry only; fallback is a geometric OOD heuristic, not calibrated accuracy"}, labels


def predict_gate(records, gate):
    _, raw = matrix(records, gate["feature_names"])
    values = standardized(raw, np.asarray(gate["medians"]), np.asarray(gate["scales"]))
    distances = np.mean((values[:, None, :] - np.asarray(gate["centers"])[None, :, :])**2, axis=2)
    labels = (distances[:, 0]-distances[:, 1] > gate["boundary"]).astype(int)
    accepted = distances[np.arange(len(labels)), labels] <= np.asarray(gate["training_radius_q99"])[labels]
    return labels, accepted
