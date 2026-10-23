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
