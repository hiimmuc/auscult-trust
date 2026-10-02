"""Fixed patient-level aggregation (proposal 4.2). Not learned."""
import pandas as pd

PHASES = ("insp", "exp")


def patient_features(df):
    """Aggregate cycle probabilities into 8 patient-level audio features.

    Max and mean of p(crackle) and p(wheeze), per phase, across all sites.
    4-class probabilities map as p(crackle) = P(crackle) + P(both),
    p(wheeze) = P(wheeze) + P(both); do that before calling.

    Args:
        df: DataFrame with `patient`, `phase` ("insp"/"exp"), `p_crackle`, `p_wheeze`.

    Returns:
        DataFrame indexed by patient with columns like `insp_crackle_max`.
        Missing phase/class combinations are NaN.
    """
    out = {}
    for ph in PHASES:
        g = df[df.phase == ph].groupby("patient")
        for c in ("crackle", "wheeze"):
            for stat in ("max", "mean"):
                out[f"{ph}_{c}_{stat}"] = g[f"p_{c}"].agg(stat)
    return pd.DataFrame(out).reindex(sorted(df.patient.unique()))
