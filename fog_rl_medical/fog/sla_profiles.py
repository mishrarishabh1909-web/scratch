from dataclasses import dataclass

@dataclass
class SLAProfile:
    level: int
    name: str
    latency_ms: float
    compute_budget: float

PRIORITY_1 = SLAProfile(1, "Critical", 100.0, 0.8)
PRIORITY_2 = SLAProfile(2, "Urgent", 500.0, 0.6)
PRIORITY_3 = SLAProfile(3, "Semi-urgent", 2000.0, 0.4)
PRIORITY_4 = SLAProfile(4, "Routine", 10000.0, 0.2)

SLA_PROFILES = {
    1: PRIORITY_1,
    2: PRIORITY_2,
    3: PRIORITY_3,
    4: PRIORITY_4
}
