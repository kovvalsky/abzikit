from typing import List, Set, Any

def set_f1(pred: Set[Any], ref: Set[Any]) -> float:
    """F1 between two sets."""
    if pred is None: return {"f1": 0, "p": 1.0, "r": 0, "tp":0}
    if not pred and not ref: return {"f1": 1.0, "p": 1.0, "r": 1.0, "tp":0}
    if not pred: return {"f1": 0, "p": 1.0, "r": 0, "tp":0}
    if not ref: return {"f1": 0, "p": 0, "r": 1.0, "tp":0}

    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)

    if precision + recall == 0: return {"f1": 0, "p": 0, "r": 0, "tp":tp}
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "p": precision, "r": recall, "tp":tp}


def f1_em_score(
    references: List[List[Set[Any]]],
    preds: List[Set[Any]],
    lower_case = True,
    ignore_failed = True,
    tie_breaker = "r" # r recall, tp true positives, p precision
):
    """
    Multi-reference F1/EM for set-valued predictions.
    references[i] = list of acceptable reference sets for item i
    preds[i]      = predicted set for item i, None for always wrong answers
    F1 relation-level set F1 against the best matching reference.
    Returns: (exact_match, f1-micro, f1-macro)
    """
    assert len(references) == len(preds),\
        f"references and preds must have same length: "\
        f"{len(references)} != {len(preds)}"

    exact_matches, per_item_f1s, per_item_precs, per_item_recs = [], [], [], []

    total_tp, total_preds, total_refs = 0, 0, 0

    for pred, refs in zip(preds, references):
        if ignore_failed and pred is None:
            continue

        if not refs:
            raise ValueError("Each datapoint must have at least one reference set.")

        if lower_case:
            if pred is not None:
                pred = set(p.lower() for p in pred)
            refs = [ set(r.lower() for r in ref) for ref in refs ]

        # EM: prediction must exactly match at least one reference
        exact_matches.append(any(pred == ref for ref in refs))

        # choose best reference, SQuAD-style
        ranked_ref = [ (ref, set_f1(pred, ref)) for ref in refs ]
        ranked_ref = sorted(ranked_ref,
                            key=lambda ref: (ref[1]["f1"], ref[1][tie_breaker]), reverse=True)

        best_ref = ranked_ref[0][0]

        best_f1_p_r = set_f1(pred, best_ref)
        per_item_f1s.append(best_f1_p_r["f1"])
        per_item_precs.append(best_f1_p_r["p"])
        per_item_recs.append(best_f1_p_r["r"])

        if pred is not None:
            total_tp += len(pred & best_ref)
            total_preds += len(pred)
        total_refs += len(best_ref)

    em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    # micro
    denom = total_preds + total_refs
    f1_micro = 1.0 if denom == 0 else (2 * total_tp) / denom
    p_micro = 1.0 if total_preds == 0 else total_tp / total_preds
    r_micro = 1.0 if total_refs == 0 else total_tp / total_refs
    # macro
    f1_macro = sum(per_item_f1s) / len(per_item_f1s) if per_item_f1s else 0.0
    p_macro = sum(per_item_precs) / len(per_item_precs) if per_item_precs else 0.0
    r_macro = sum(per_item_recs) / len(per_item_recs) if per_item_recs else 0.0

    return {"em":em,
            "micro-f1": f1_micro, "micro-p": p_micro, "micro-r": r_micro,
            "macro-f1": f1_macro, "macro-p": p_macro, "macro-r": r_macro}