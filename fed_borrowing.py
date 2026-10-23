"""Fed emergency lending (BTFP + discount window) per FDIC bank, matched by RSSD, then ABA via BTFP, then name+city+state."""
import re

import pandas as pd

from fetch_v3 import V3

ABBR = {"BK": "BANK", "BKG": "BANKING", "BNK": "BANK", "TR": "TRUST", "TRST": "TRUST", "NB": "NATIONAL BANK",
        "NATL": "NATIONAL", "CO": "COMPANY", "ST": "STATE", "SVG": "SAVINGS", "SVGS": "SAVINGS", "SAV": "SAVINGS",
        "SB": "SAVINGS BANK", "FSB": "FEDERAL SAVINGS BANK", "FNB": "FIRST NATIONAL BANK", "1ST": "FIRST",
        "CMNTY": "COMMUNITY", "CMTY": "COMMUNITY", "CNTY": "COUNTY", "AMER": "AMERICAN", "FED": "FEDERAL",
        "BT": "BANK AND TRUST", "B&T": "BANK AND TRUST", "TC": "TRUST COMPANY", "&": "AND", "MTG": "MORTGAGE",
        "CTR": "CENTER", "CNTRL": "CENTRAL", "CENT": "CENTRAL", "MERCH": "MERCHANTS", "FARM": "FARMERS",
        "FRMRS": "FARMERS", "SECUR": "SECURITY", "CITIZ": "CITIZENS", "CTZN": "CITIZENS", "CTZNS": "CITIZENS"}
DROP = {"THE", "OF", "INC", "NA", "N", "A", "LTD", "CORP", "CORPORATION"}
CREDIT_UNION = re.compile(r"\b(?:CU|FCU|CREDIT UNION|C U)\b")


def norm(name):
    s = re.sub(r"[^A-Z0-9& ]", " ", str(name).upper().replace("B&T", " B&T ").replace("N.A.", " NA "))
    words = [ABBR.get(w, w) for w in s.split()]
    return " ".join(w for w in " ".join(words).split() if w not in DROP)


def load_btfp():
    b = pd.read_excel(V3 / "fed" / "btfp.xlsx", sheet_name=0)
    return pd.DataFrame({"date": pd.to_datetime(b["Date Of Advance"]), "name": b["Borrower Name"], "city": b["Borrower City"],
                         "state": b["Borrower State"], "rssd": pd.to_numeric(b["Borrower RSSD"], errors="coerce"),
                         "aba": b["Borrower ABA"].astype(str).str.zfill(9), "amount": pd.to_numeric(b["Advance Amount"], errors="coerce")})


def load_dw(files):
    parts = []
    for f in files:
        x = pd.read_excel(V3 / "fed" / f, sheet_name=0, header=3)
        parts.append(pd.DataFrame({"date": pd.to_datetime(x["Loan date"], errors="coerce"), "name": x["Borrower"],
                                   "city": x["Borrower city"], "state": x["Borrower state"],
                                   "aba": x["Borrower ABA number"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9),
                                   "amount": pd.to_numeric(x["Loan amount"], errors="coerce")}))
    return pd.concat(parts, ignore_index=True).dropna(subset=["date", "amount"])


def match(loans, banks, aba_to_rssd):
    """Attach CERT to each loan. banks: CERT, RSSDID, NAME, CITY, STALP for one quarter."""
    loans = loans.copy()
    rssd_to_cert = banks.dropna(subset=["RSSDID"]).set_index("RSSDID").CERT
    if "rssd" not in loans:
        loans["rssd"] = loans.aba.map(aba_to_rssd)
    loans["CERT"] = loans.rssd.map(rssd_to_cert)
    key = banks.assign(k=banks.NAME.map(norm) + "|" + banks.STALP.str.upper(),
                       kc=banks.NAME.map(norm) + "|" + banks.CITY.str.upper() + "|" + banks.STALP.str.upper())
    uniq = key.drop_duplicates("k", keep=False).set_index("k").CERT
    by_city = key.drop_duplicates("kc", keep=False).set_index("kc").CERT
    todo = loans.CERT.isna()
    n = loans.loc[todo, "name"].map(norm)
    loans.loc[todo, "CERT"] = (n + "|" + loans.loc[todo, "city"].str.upper() + "|" + loans.loc[todo, "state"].str.upper()).map(by_city)
    todo = loans.CERT.isna()
    loans.loc[todo, "CERT"] = (loans.loc[todo, "name"].map(norm) + "|" + loans.loc[todo, "state"].str.upper()).map(uniq)
    loans["credit_union"] = loans.name.str.upper().str.contains(CREDIT_UNION)
    return loans


def borrowing(start, end, banks, with_btfp):
    """Total emergency advances ($ thousands) per CERT with dates in [start, end], plus a match report."""
    btfp = load_btfp()
    aba_to_rssd = btfp.dropna(subset=["rssd"]).drop_duplicates("aba").set_index("aba").rssd
    year = pd.Timestamp(start).year
    dw = load_dw([f"dw_{year}_q1.xlsx", f"dw_{year}_q2.xlsx"])
    loans = [match(dw, banks, aba_to_rssd).assign(src="dw")]
    if with_btfp:
        loans.append(match(btfp, banks, aba_to_rssd).assign(src="btfp"))
    loans = pd.concat(loans, ignore_index=True)
    loans = loans[loans.date.between(start, end)]
    banks_only = loans[~loans.credit_union]
    report = {"loans": len(loans), "credit_union_loans": int(loans.credit_union.sum()),
              "bank_borrowers": int(banks_only.name.nunique()),
              "bank_borrowers_matched": int(banks_only.dropna(subset=["CERT"]).name.nunique()),
              "bank_amount_matched_share": round(float(banks_only.dropna(subset=["CERT"]).amount.sum() / banks_only.amount.sum()), 4)}
    total = loans.dropna(subset=["CERT"]).groupby("CERT").amount.sum() / 1000
    return total, report
