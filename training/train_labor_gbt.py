from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


DEFAULT_VALIDATION_EPISODE = 104830155


def _load_datasets(dataset_dir: Path):
    arrays = []
    feature_names = None
    agent_names = None
    group_offset = 0

    for path in sorted(dataset_dir.glob("*.npz")):
        data = np.load(path)
        names = data["feature_names"].tolist()
        agents = data["agent_names"].tolist()
        if feature_names is None:
            feature_names = names
            agent_names = agents
        elif names != feature_names or agents != agent_names:
            raise ValueError(f"Schema mismatch in {path}")

        group = data["group"].astype(np.int64) + group_offset
        group_offset = int(group.max()) + 1 if group.size else group_offset
        arrays.append({
            "X": data["X"].astype(np.float32, copy=False),
            "y": data["y"].astype(np.uint8, copy=False),
            "weight": data["weight"].astype(np.float32, copy=False),
            "episode": data["episode"].astype(np.int64, copy=False),
            "agent": data["agent"].astype(np.uint8, copy=False),
            "group": group,
            "movement": data["movement"].astype(np.uint8, copy=False),
        })

    if not arrays:
        raise RuntimeError(f"No .npz datasets found in {dataset_dir}")

    merged = {
        key: np.concatenate([a[key] for a in arrays], axis=0)
        for key in arrays[0]
    }
    return merged, feature_names, agent_names


def _balance_experts(weights, agent_ids, train_mask):
    balanced = weights.astype(np.float32, copy=True)
    totals = {}
    for agent_id in np.unique(agent_ids[train_mask]):
        mask = train_mask & (agent_ids == agent_id)
        totals[int(agent_id)] = float(balanced[mask].sum())

    target = sum(totals.values()) / len(totals)
    for agent_id, total in totals.items():
        balanced[train_mask & (agent_ids == agent_id)] *= target / total
    return balanced, totals


def _group_top1_accuracy(scores, labels, groups, mask, movement=None):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return float("nan")

    order = np.argsort(groups[indices], kind="stable")
    indices = indices[order]
    correct = 0
    count = 0
    start = 0

    while start < len(indices):
        group_id = groups[indices[start]]
        end = start + 1
        while end < len(indices) and groups[indices[end]] == group_id:
            end += 1
        group_indices = indices[start:end]
        if movement is not None and not movement[group_indices[0]]:
            start = end
            continue
        best = group_indices[np.argmax(scores[group_indices])]
        correct += int(labels[best] == 1)
        count += 1
        start = end

    return correct / count if count else float("nan")


def _heuristic_scores(X, feature_names):
    """Frozen heuristic ordering. Groups already contain one priority tier."""
    idx = {name: i for i, name in enumerate(feature_names)}
    distance = X[:, idx["distance"]]
    delivery = np.ones(len(X), dtype=np.float32)
    for task_type in {"pickup_animal", "place_animal"}:
        is_type = X[:, idx[f"task__{task_type}"]] > 0.5
        delivery[is_type] = 0
    return -(delivery * 100.0 + distance)


def _params():
    return dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--validation-episode", type=int, default=DEFAULT_VALIDATION_EPISODE)
    args = parser.parse_args()

    data, feature_names, agent_names = _load_datasets(args.dataset_dir)
    X = data["X"]
    y = data["y"]
    validation = data["episode"] == args.validation_episode
    train = ~validation
    if not validation.any():
        raise ValueError(f"Validation episode {args.validation_episode} not found")

    weights, raw_agent_totals = _balance_experts(data["weight"], data["agent"], train)
    model = XGBClassifier(**_params())
    model.fit(X[train], y[train], sample_weight=weights[train])

    scores = np.zeros(len(X), dtype=np.float32)
    scores[validation] = model.predict_proba(X[validation])[:, 1]
    heuristic = _heuristic_scores(X, feature_names)

    print(f"rows: {len(X):,}; train={train.sum():,}; validation={validation.sum():,}")
    print(f"same-tier decisions: {len(np.unique(data['group'])):,}")
    print(f"features: {len(feature_names)}")
    print("raw training weight by expert:")
    for agent_id, total in raw_agent_totals.items():
        print(f"  {agent_names[agent_id]}: {total:.1f}")
    print(f"validation ROC AUC: {roc_auc_score(y[validation], scores[validation]):.4f}")
    print(f"validation same-tier top-1, heuristic: {_group_top1_accuracy(heuristic, y, data['group'], validation):.4f}")
    print(f"validation same-tier top-1, GBT: {_group_top1_accuracy(scores, y, data['group'], validation):.4f}")
    print(f"movement same-tier top-1, heuristic: {_group_top1_accuracy(heuristic, y, data['group'], validation, data['movement']):.4f}")
    print(f"movement same-tier top-1, GBT: {_group_top1_accuracy(scores, y, data['group'], validation, data['movement']):.4f}")

    importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    print("top features:")
    for name, value in importance:
        print(f"  {name}: {value:.4f}")

    # Refit deployable model on all rows after the episode-level diagnostic.
    all_mask = np.ones(len(X), dtype=bool)
    final_weights, _ = _balance_experts(data["weight"], data["agent"], all_mask)
    final_model = XGBClassifier(**_params())
    final_model.fit(X, y, sample_weight=final_weights)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(args.model_path)
    metadata_path = args.model_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps({
        "feature_names": feature_names,
        "experts": agent_names,
        "training_rows": int(len(X)),
        "same_tier_decisions": int(len(np.unique(data["group"]))),
        "movement_decisions": int(len(np.unique(data["group"][data["movement"] > 0]))),
        "validation_episode": int(args.validation_episode),
        "ranking_mode": "heuristic_priority_then_gbt_with_continuity",
        "training_target": "expert target versus same-heuristic-priority alternatives",
    }, indent=2))
    print(f"saved model: {args.model_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
