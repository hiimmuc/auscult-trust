import numpy as np

from src.processing.splits import lodo_split


def _data():
    # patients 1-10 on device 0, 11-20 on device 1, 21-30 on device 2; patient 5 also recorded on device 2
    pats, devs = [], []
    for p in range(1, 31):
        d = 0 if p <= 10 else 1 if p <= 20 else 2
        pats += [p] * 3
        devs += [d] * 3
    pats += [5] * 2
    devs += [2] * 2
    return np.array(pats), np.array(devs)


def test_lodo_split_is_patient_disjoint_and_handles_shared_patient():
    pats, devs = _data()
    tr, ca, te = lodo_split(pats, devs, held=2, cal_frac=0.2, seed=0)
    assert set(devs[te]) == {2} and len(te) == (10 * 3 + 2)  # every cycle of the held-out device
    tp, cp, ep = set(pats[tr]), set(pats[ca]), set(pats[te])
    assert not tp & cp and not tp & ep and not cp & ep  # patient 5 is in test only
    assert 5 in ep and 5 not in tp | cp
    assert set(devs[tr]) | set(devs[ca]) == {0, 1}
    assert len(cp) == 2 + 2  # 20% of 9 (patient 5 left) -> 2 on device 0, 20% of 10 -> 2 on device 1


def test_lodo_split_seed_and_drop():
    pats, devs = _data()
    a = lodo_split(pats, devs, 2, seed=0)[1]
    assert np.array_equal(a, lodo_split(pats, devs, 2, seed=0)[1])
    assert not np.array_equal(a, lodo_split(pats, devs, 2, seed=1)[1])
    tr, ca, te = lodo_split(pats, devs, 2, drop=[1, 25])
    assert not {1, 25} & set(pats[np.concatenate([tr, ca, te])])
