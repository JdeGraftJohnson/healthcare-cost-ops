"""Smoke tests for the DAX static validator."""
from services.dax.validate import validate, summarize


def test_ok_passes():
    dax = "[Total] := SUM ( fact_sdud[total_reimb] )"
    f = validate(dax, schema_columns={"fact_sdud": {"total_reimb"}})
    assert summarize(f)["MISS"] == 0


def test_unknown_function_warns():
    dax = "[X] := FROBNICATE ( fact_sdud[total_reimb] )"
    f = validate(dax, schema_columns={"fact_sdud": {"total_reimb"}})
    assert any(x.code == "UNKNOWN_FN" for x in f)


def test_unknown_column_misses():
    dax = "[X] := SUM ( fact_sdud[does_not_exist] )"
    f = validate(dax, schema_columns={"fact_sdud": {"total_reimb"}})
    assert any(x.code == "UNKNOWN_COLUMN" and x.severity == "MISS" for x in f)


def test_raw_divide_warns():
    dax = "[Ratio] := fact_sdud[a] / fact_sdud[b]"
    f = validate(dax, schema_columns={"fact_sdud": {"a", "b"}})
    assert any(x.code == "RAW_DIVIDE" for x in f)


def test_template_01_loads():
    from services.dax.templates import render
    r = render("01")
    assert "Total Reimbursement" in r.dax
    assert "YoY Growth %" in r.dax
