
from dataclasses import dataclass

@dataclass
class Decision:
    label: str
    conf: float
    features: dict
    bbox: tuple|None

@dataclass
class Thresholds:
    score_valid: float
    score_review: float

VALID="VALID STAMP"; NONE="NO STAMP"; REVIEW="REVIEW REQUIRED"

def decide(score: float, geo_ok: bool, feats: dict, bbox=None, th: Thresholds=Thresholds(0.6,0.35)) -> Decision:
    if score >= th.score_valid and geo_ok:
        return Decision(VALID, score, feats, bbox)
    if score < th.score_review:
        return Decision(NONE, score, feats, bbox)
    return Decision(REVIEW, score, feats, bbox)
