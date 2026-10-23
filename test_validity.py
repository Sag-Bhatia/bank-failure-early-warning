"""Leakage guards. Run against the real built data: pytest test_validity.py"""
import numpy as np
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


def test_factcheck_catches_invented_numbers():
    from factcheck import check
    p = [{"risk_rank_pct": 5.0, "values": {"a": 1.04}, "peer_percentile": {"a": 98.8}, "score_contribution": {"a": 0.35}}]
    assert check("Losses are 104% of capital (98.8th pct), contribution +0.35.", p)["untraced"] == []
    assert set(check("Losses are 140% of capital and 77.7th pct.", p)["untraced"]) == {"140", "77.7"}


def _raw_v3(rows):
    from fetch_v3 import FIELDS
    base = {c: 0.0 for c in FIELDS.split(",")}
    base.update(NAME="X", CITY="Y", STALP="CA", ASSET=1000.0, DEP=800.0, EQ=100.0, RBCT1J=100.0, LNLSGR=500.0, DEPUNINS=400.0,
                CHBAL=50.0, SC=100.0, SCHA=100.0, SCHF=90.0, SCAA=0.0, SCAF=0.0)
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["repdte"] = pd.to_datetime(df.REPDTE, format="%Y%m%d")
    return df


def test_v3_loan_mtm_and_run_labels():
    from features_v3 import build
    yields = pd.DataFrame({c: 1.0 for c in ["DGS3MO", "DGS1", "DGS3", "DGS5", "DGS10"]},
                          index=pd.date_range("2020-03-31", periods=12, freq="QE"))
    yields.iloc[8:, :] = 3.0  # +2pp over 8 quarters by 2022Q1
    rows = [dict(CERT=1, REPDTE="20220331", LNRS5T15=100.0),                     # 100 * 6y * 2% = 12
            dict(CERT=1, REPDTE="20220630", DEPUNINS=300.0),                      # -25% -> run event
            dict(CERT=2, REPDTE="20220331"),                                       # vanishes, no failure -> excluded
            dict(CERT=3, REPDTE="20220331"),                                       # fails before next filing -> event
            dict(CERT=4, REPDTE="20220331", DEPUNINS=40.0),                        # uninsured 5% -> not eligible
            dict(CERT=4, REPDTE="20220630", DEPUNINS=10.0)]
    fails = pd.DataFrame({"CERT": [3.0], "NAME": ["Z"], "faildate": pd.to_datetime(["2022-07-15"])})
    out = build(_raw_v3(rows), yields, fails).set_index(["CERT", "repdte"])
    r1 = out.loc[(1, pd.Timestamp("2022-03-31"))]
    assert r1.loan_mtm_t1 == pytest.approx(12 / 100)
    assert r1.mtm_cap == pytest.approx((100 - 10 - 12) / 1000)
    assert r1.run_event == 1.0
    assert pd.isna(out.loc[(2, pd.Timestamp("2022-03-31"))].run_event)
    assert out.loc[(3, pd.Timestamp("2022-03-31"))].run_event == 1.0
    assert pd.isna(out.loc[(4, pd.Timestamp("2022-03-31"))].run_event)


def test_fed_name_matching():
    from fed_borrowing import match, norm
    assert norm("FIRST NB OF SHINER") == norm("First National Bank of Shiner")
    assert norm("SENTRY BK") == norm("Sentry Bank")
    banks = pd.DataFrame({"CERT": [1, 2, 3], "RSSDID": [11.0, 22.0, 33.0], "NAME": ["SENTRY BANK", "WAYNE BANK", "WAYNE BANK"],
                          "CITY": ["SAINT JOSEPH", "HONESDALE", "OTHERTOWN"], "STALP": ["MN", "PA", "PA"]})
    loans = pd.DataFrame({"name": ["SENTRY BK", "WAYNE BK", "MYSTERY FCU"], "city": ["SAINT JOSEPH", "HONESDALE", "X"],
                          "state": ["MN", "PA", "PA"], "aba": ["1", "2", "3"], "amount": [1.0, 1.0, 1.0],
                          "date": pd.to_datetime(["2023-03-14"] * 3)})
    m = match(loans, banks, pd.Series(dtype=float))
    assert m.CERT.tolist()[:2] == [1, 2] and pd.isna(m.CERT.iloc[2]) and m.credit_union.iloc[2]


def test_v31_fixes_capital_zero_and_efficiency_ratio():
    from features_v3 import build
    yields = pd.DataFrame({c: 1.0 for c in ["DGS3MO", "DGS1", "DGS3", "DGS5", "DGS10"]},
                          index=pd.date_range("2020-03-31", periods=12, freq="QE"))
    raw = _raw_v3([dict(CERT=1, REPDTE="20220331", IDT1RWAJR=0.0, EEFF=60.0),
                   dict(CERT=2, REPDTE="20220331", IDT1RWAJR=12.0, EEFF=50.0)]).assign(IEFF=100.0)
    none = pd.DataFrame({"CERT": [99.0], "NAME": ["Z"], "faildate": pd.to_datetime(["2001-01-01"])})
    old = build(raw, yields, none).set_index("CERT")
    new = build(raw, yields, none, fixed=True).set_index("CERT")
    assert old.loc[1, "tier1_rbc"] == 0 and pd.isna(new.loc[1, "tier1_rbc"]) and new.loc[2, "tier1_rbc"] == 12
    assert old.loc[1, "efficiency"] == 60 and new.loc[1, "efficiency"] == pytest.approx(0.6)


def test_v4_trajectory_regional_and_peer_label():
    from features_v4 import regional, run_label_v4, trajectory
    d = pd.DataFrame({"CERT": [1, 1], "repdte": pd.to_datetime(["2020-03-31", "2021-03-31"]), "STALP": ["CA", "CA"],
                      "leverage": [10.0, 8.0], "npl_oreo": [0.01, 0.03], "roa": [1.0, 0.5], "uninsured": [0.3, 0.4],
                      "liquid": [0.2, 0.1], "mtm_cap": [0.08, 0.05]})
    t = trajectory(d)
    assert pd.isna(t.d_leverage.iloc[0]) and t.d_leverage.iloc[1] == pytest.approx(-2.0) and t.d_uninsured.iloc[1] == pytest.approx(0.1)
    states = pd.DataFrame({"STALP": "CA", "repdte": pd.date_range("2020-03-31", periods=5, freq="QE"),
                           "unemp": [4.0, 5, 6, 7, 8.0], "hpi": [100.0, 101, 102, 103, 110.0]})
    g = regional(d, states)
    assert g.state_unemp.iloc[1] == 8.0 and g.state_unemp_chg.iloc[1] == pytest.approx(4.0) and g.state_hpi_chg.iloc[1] == pytest.approx(0.10)
    # peer label: 100 banks in one quarter, 20 per size quintile; the worst 5% of each quintile (1 bank) is the event
    n = 100
    lab = pd.DataFrame({"repdte": pd.Timestamp("2015-12-31"), "log_assets": np.arange(n, dtype=float), "uninsured": 0.3,
                        "DEPUNINS_next": 1.0, "faildate": pd.NaT, "outcome_q": pd.Timestamp("2016-03-31"),
                        "unins_chg_next": np.tile(np.linspace(-0.5, 0.5, 20), 5)})
    ev = run_label_v4(lab)
    assert ev.sum() == 5 and ev.groupby(np.arange(n) // 20).sum().tolist() == [1.0] * 5
