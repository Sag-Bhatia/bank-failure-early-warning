"""Leakage guards. Run against the real built data: pytest test_validity.py"""
import pandas as pd
import pytest

from features import HORIZON, LAG, TEST_START, build, failures
from fetch import DATA


@pytest.fixture(scope="module")
def d():
    return pd.read_parquet(DATA / "features.parquet")


def test_public_date_is_lagged(d):
    assert (d.public_date - d.repdte == LAG).all()


def test_no_report_public_after_failure(d):
    assert not (d.public_date >= d.faildate).any()


def test_label_window(d):
    pos = d[d.y == 1]
    assert ((pos.faildate > pos.public_date) & (pos.faildate <= pos.public_date + HORIZON)).all()
    neg_failed = d[(d.y == 0) & d.faildate.notna()]
    assert (neg_failed.faildate > neg_failed.public_date + HORIZON).all()


def test_train_labels_resolve_before_test_is_public(d):
    train = d[d.split == "train"]
    assert train.repdte.max() < TEST_START
    # windows meet on 2019-05-15; same-day is fine as long as no failure lands on it (checked below)
    assert (train.public_date + HORIZON).max() <= d[d.split == "test"].public_date.min()
    assert not (d.faildate == d[d.split == "test"].public_date.min()).any()


def test_no_failure_event_positive_in_both_splits(d):
    pos = d[d.y == 1]
    assert not set(pos[pos.split == "train"].CERT) & set(pos[pos.split == "test"].CERT)


def test_growth_uses_only_past_report():
    fin = pd.DataFrame({"CERT": [1, 1], "REPDTE": ["20200331", "20210331"], "LNLSGR": [100, 150], "DEP": [200, 100]})
    for c in "NAME CITY STALP".split():
        fin[c] = "x"
    for c in ("ASSET EQ INTAN LNATRES NCLNLS ORE RBCT1J RBC1AAJ ROA LNRECONS LNRENRES LNREMULT BRO NTLNLS "
              "SCHA SCHF SCAA SCAF DEPUNINS SC CHBAL").split():
        fin[c] = 1.0
    out = build(fin, pd.DataFrame({"CERT": [], "faildate": pd.to_datetime([])})).set_index("repdte")
    assert pd.isna(out.loc["2020-03-31", "loan_growth"])
    assert out.loc["2021-03-31", "loan_growth"] == pytest.approx(0.5)
    assert out.loc["2021-03-31", "dep_growth"] == pytest.approx(-0.5)


def test_named_2023_failures_present():
    names = failures().NAME.str.upper()
    for n in ["SILICON VALLEY BANK", "SIGNATURE BANK", "FIRST REPUBLIC BANK"]:
        assert names.eq(n).any(), n


def test_v2_lens_direction_and_flow_rules():
    from evaluate_v2 import flows
    from lenses import lenses
    base = dict(NCLNLS=1.0, ORE=0.0, EQ=100.0, INTAN=0.0, LNATRES=1.0, ASSET=1000.0, SCHA=200.0, SCHF=200.0,
                SCAF=100.0, CHBAL=100.0, DEPUNINS=100.0, DEP=800.0, SCPLEDGE=0.0, OTHBFHLB=0.0)
    rows = [dict(base, CERT=1), dict(base, CERT=2, SCHF=120.0, DEPUNINS=700.0), dict(base, CERT=3, NCLNLS=90.0)]
    fin = pd.DataFrame([dict(r, REPDTE="20221231") for r in rows]
                       + [dict(base, CERT=1, REPDTE="20230331", DEPUNINS=90.0), dict(base, CERT=3, REPDTE="20230331")])
    lz = lenses(fin[fin.REPDTE == "20221231"]).set_index("CERT")
    assert lz.run.idxmax() == 2 and lz.credit.idxmax() == 3
    extra = fin[["CERT", "REPDTE", "SCPLEDGE", "OTHBFHLB"]]
    fails = pd.DataFrame({"CERT": [2.0], "faildate": pd.to_datetime(["2023-03-10"])})
    s = flows(fin.drop(columns=["SCPLEDGE", "OTHBFHLB"]), extra, fails, "20221231", "20230331", "2023-05-15").set_index("CERT")
    assert s.loc[2, "unins_loss"] == 1.0            # failed before filing -> worst outcome
    assert s.loc[1, "unins_loss"] == pytest.approx(0.1)
    fails_none = fails.iloc[:0]
    s2 = flows(fin.drop(columns=["SCPLEDGE", "OTHBFHLB"]), extra, fails_none, "20221231", "20230331", "2023-05-15")
    assert 2 not in set(s2.CERT)                     # vanished without failing -> excluded
