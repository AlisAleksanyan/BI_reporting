# generate_shiftslots.py
import argparse, os, sys
from datetime import datetime, timedelta
import calendar

import numpy as np
import pandas as pd
import pytz

# ------------------------------- Date helpers -------------------------------

def first_iso_week_start_of_month(dt: datetime) -> datetime:
    d0 = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if d0.weekday() == 6:  # Sunday
        d0 += timedelta(days=1)
    iso = d0.isocalendar()
    start = d0 - timedelta(days=iso.weekday - 1)  # monday
    return start.replace(hour=0, minute=0, second=0, microsecond=0)

def last_working_day(dt: datetime) -> datetime:
    last_day = dt.replace(
        day=calendar.monthrange(dt.year, dt.month)[1],
        hour=0, minute=0, second=0, microsecond=0
    )
    cur = last_day
    while cur.weekday() >= 5:  # Sat, Sun
        cur -= timedelta(days=1)
    return cur

def friday_of_iso_week(d: datetime) -> datetime:
    return (d - timedelta(days=d.weekday())) + timedelta(days=4)

def planning_window_for(snapshot_dt: datetime) -> tuple[datetime, datetime]:
    month_start = first_iso_week_start_of_month(snapshot_dt)
    nm_year = snapshot_dt.year + (1 if snapshot_dt.month == 12 else 0)
    nm_month = 1 if snapshot_dt.month == 12 else snapshot_dt.month + 1
    next_month = snapshot_dt.replace(year=nm_year, month=nm_month, day=1)
    lwd_next = last_working_day(next_month)
    end = friday_of_iso_week(lwd_next)
    return month_start.date(), end.date()

def business_days(start: datetime, end: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="B")

# ------------------------------- Metrics ------------------------------------

def compute_week_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    num_cols = [
        "TotalHours","OpenHours","First Visit","Pre-Sales","After-Sales",
        "PersonalBlockHours","BusinessBlockHours","AdminBlockHours","WeBlockHours","OtherBlockHours",
        "[Agenda_Appointments]"
    ]
    for c in num_cols:
        if c not in d.columns:
            d[c] = 0.0

    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tot = d["TotalHours"].replace(0, np.nan)

    d["FTE_week"] = (d["TotalHours"] / 40).round(2)
    d["SaturationPercentage_week"] = (100 - (d["OpenHours"] * 100.0 / tot)).fillna(0).clip(0, 100)
    d["FirstVisitPercentage_week"] = ((d["First Visit"] * 5.0 / 60.0) / tot).fillna(0).clip(0, 1)
    d["PreSalesPercentage_week"] = ((d["Pre-Sales"] * 5.0 / 60.0) / tot).fillna(0).clip(0, 1)
    d["AfterSalesPercentage_week"] = ((d["After-Sales"] * 5.0 / 60.0) / tot).fillna(0).clip(0, 1)
    d["PersonalBlockPercentage_week"] = (d["PersonalBlockHours"] / tot).fillna(0).clip(0, 1)
    d["BusinessBlockPercentage_week"] = (d["BusinessBlockHours"] / tot).fillna(0).clip(0, 1)
    d["AdminBlockPercentage_week"] = (d["AdminBlockHours"] / tot).fillna(0).clip(0, 1)
    d["WeBlockPercentage_week"] = (d["WeBlockHours"] / tot).fillna(0).clip(0, 1)

    # Ensure ratio inputs exist; seed with [Agenda_Appointments] when zeros
    for scope in ["", "_area", "_reg"]:
        cur_col = f"Current_Week_Sum{scope}"
        upc_col = f"Upcoming_3_Weeks_Sum{scope}"
        if cur_col not in d.columns:
            d[cur_col] = 0.0
        if upc_col not in d.columns:
            d[upc_col] = 0.0
        seed = d.get("[Agenda_Appointments]", pd.Series(0, index=d.index)).fillna(0)
        d[cur_col] = d[cur_col].where(d[cur_col].ne(0), seed)
        d[upc_col] = d[upc_col].where(d[upc_col].ne(0), (seed * 3).replace(0, 1))

    # Ratios
    for scope in ["", "_area", "_reg"]:
        rcol = f"ratioadelanto{scope}"
        cur_col = f"Current_Week_Sum{scope}"
        upc_col = f"Upcoming_3_Weeks_Sum{scope}"
        if rcol not in d.columns:
            d[rcol] = 0.0
        d[rcol] = (d[cur_col] / d[upc_col].replace(0, np.nan)).fillna(0).round(3)

    d["Current_Week_Sum_week"] = d["Current_Week_Sum"]
    d["ratioadelanto_week"] = (
        d["Current_Week_Sum"] / d["Upcoming_3_Weeks_Sum"].replace(0, np.nan)
    ).fillna(0).round(3)

    return d

def compute_daily_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create row-level percentage columns that some parts of the app expect.
    All percentages here are 0..100 scale (not 0..1).
    """
    d = df.copy()
    for c in [
        "TotalHours","OpenHours","First Visit","Pre-Sales","After-Sales",
        "PersonalBlockHours","BusinessBlockHours","AdminBlockHours","WeBlockHours","OtherBlockHours"
    ]:
        if c not in d.columns:
            d[c] = 0.0
    d = d.replace([np.inf, -np.inf], np.nan)
    tot = d["TotalHours"].astype(float).replace(0, np.nan)

    # 0..100 scale
    d["SaturationPercentage"]    = (100 - (d["OpenHours"] * 100.0 / tot)).fillna(0).clip(0, 100)
    d["FirstVisitPercentage"]    = ((d["First Visit"] * 5.0 / 60.0) / tot * 100).fillna(0).clip(0, 100)
    d["PreSalesPercentage"]      = ((d["Pre-Sales"] * 5.0 / 60.0) / tot * 100).fillna(0).clip(0, 100)
    d["AfterSalesPercentage"]    = ((d["After-Sales"] * 5.0 / 60.0) / tot * 100).fillna(0).clip(0, 100)
    d["PersonalBlocksPercentage"]= (d["PersonalBlockHours"] / tot * 100).fillna(0).clip(0, 100)
    d["BusinessBlocksPercentage"]= (d["BusinessBlockHours"] / tot * 100).fillna(0).clip(0, 100)
    d["AdminBlocksPercentage"]   = (d["AdminBlockHours"] / tot * 100).fillna(0).clip(0, 100)
    d["WeBlocksPercentage"]      = (d["WeBlockHours"] / tot * 100).fillna(0).clip(0, 100)
    d["OtherBlocksPercentage"]   = (d["OtherBlockHours"] / tot * 100).fillna(0).clip(0, 100)

    # Convenience (some code uses this name)
    d["BlockedHoursPercentage"]  = (
        (d["PersonalBlockHours"] + d["BusinessBlockHours"] + d["AdminBlockHours"] + d["WeBlockHours"] + d["OtherBlockHours"])
        / tot * 100
    ).fillna(0).clip(0, 100)

    return d

# ----------------------------- Schema expectations --------------------------

ID_COLS = ["GT_ShopCode__c", "Shop[Name]", "Region", "Area", "CATEGORY"]

HOUR_COLS = [
    "OpenHours","BlockedHours","BookedHours","TotalHours",
    "After-Sales","First Visit","Pre-Sales",
    "PersonalBlockHours","BusinessBlockHours","AdminBlockHours","WeBlockHours","OtherBlockHours"
]

OPTIONAL_APP_COLS = [
    # slots/shift columns the Streamlit app touches
    "30min_slots","60min_slots",
    "ShiftDurationHours","OverlapHours","OpenHours_MB",
    # common app metrics / flags that often appear
    "tmk_flag","extra_pos",
    "[HA_Sales__Units_]","[Agenda_Appointments]","[Appointments_Cancelled]",
    "[Last_3M_Sales_Value]","[Last_3M_Appointments]","[Last_3M_Cancelled]",
]

# ------------------------------- Core builder -------------------------------

def extend_to_full_window(template_df: pd.DataFrame, snapshot_dt: datetime, tz_str="Europe/Madrid") -> pd.DataFrame:
    tz = pytz.timezone(tz_str)
    snap = snapshot_dt if snapshot_dt.tzinfo else tz.localize(snapshot_dt)

    df = template_df.copy()

    # Require a date column in template
    if "date" not in df.columns:
        raise ValueError("Template must contain a 'date' column with daily rows.")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Window boundaries
    start_d, end_d = planning_window_for(snap)
    all_bd = business_days(start_d, end_d).date  # ndarray of datetime.date

    # Per-shop base row
    if "GT_ShopCode__c" not in df.columns:
        raise ValueError("Template must contain 'GT_ShopCode__c' column.")
    base_rows = (
        df.sort_values("date")
          .groupby("GT_ShopCode__c", as_index=False)
          .first()
    )

    # Shop x BusinessDay Cartesian product
    bd_df = (
        pd.MultiIndex
          .from_product([base_rows["GT_ShopCode__c"].values, all_bd], names=["GT_ShopCode__c", "date"])
          .to_frame(index=False)
    )

    # Attach shop attributes (Region/Area/etc.)
    attrs = base_rows[ID_COLS].drop_duplicates("GT_ShopCode__c")
    full = bd_df.merge(attrs, on="GT_ShopCode__c", how="left")

    # Bring numeric columns from template when exact day exists
    numeric_cols_in_template = [c for c in HOUR_COLS if c in df.columns]
    df_num = df[["GT_ShopCode__c","date"] + numeric_cols_in_template].copy()
    # Bring appointments if present
    df_num["[Agenda_Appointments]"] = df.get("[Agenda_Appointments]", np.nan)

    full = full.merge(df_num, on=["GT_ShopCode__c","date"], how="left")

    # Make sure all numeric columns exist
    for c in HOUR_COLS + ["[Agenda_Appointments]"]:
        if c not in full.columns:
            full[c] = np.nan

    # Sort and forward/backward fill per shop (avoid index-alignment issues)
    full.sort_values(["GT_ShopCode__c","date"], inplace=True)
    cols_to_fill = HOUR_COLS + ["[Agenda_Appointments]"]
    full[cols_to_fill] = (
        full.groupby("GT_ShopCode__c", group_keys=False)[cols_to_fill]
            .apply(lambda g: g.ffill().bfill())
            .reset_index(drop=True)
    )

    # Baselines if still NaN
    for c in HOUR_COLS:
        full[c] = full[c].fillna(0.0)
    full["[Agenda_Appointments]"] = full["[Agenda_Appointments]"].fillna(0.0)

    # Ensure TotalHours >= Open+Blocked+Booked
    approx = (
        full["OpenHours"].astype(float).fillna(0) +
        full["BlockedHours"].astype(float).fillna(0) +
        full["BookedHours"].astype(float).fillna(0)
    )
    full["TotalHours"] = np.maximum(full["TotalHours"].astype(float).fillna(0), approx).round(2)

    # Calendar fields
    full["date"] = pd.to_datetime(full["date"])
    full["weekday"] = full["date"].dt.dayofweek
    iso = full["date"].dt.isocalendar()
    full["iso_week"] = iso["week"].astype(int)
    full["iso_year"] = iso["year"].astype(int)

    # ---------------------- Snapshot-aware variation ----------------------
    rng = np.random.default_rng(int(snap.strftime("%Y%m%d")))
    days_ahead = (full["date"] - pd.Timestamp(snap.date())).dt.days
    scale = np.clip(np.where((days_ahead >= -2) & (days_ahead <= 14), 0.06, 0.02), 0.0, 0.08).astype(float)

    for c in HOUR_COLS:
        base = full[c].astype(float).fillna(0.0).values
        col_noise = rng.normal(loc=0.0, scale=scale, size=len(full))
        full[c] = np.maximum(0.0, base * (1.0 + col_noise)).round(2)

    # Tiny deterministic drift per snapshot to guarantee different outputs for different snapshots
    snap_hash = (int(snap.strftime("%Y%m%d")) * 2654435761) % 2**32
    rng2 = np.random.default_rng(snap_hash)
    global_drift = rng2.uniform(-0.008, 0.008)

    future_mask = (days_ahead.to_numpy() > 0)

    def _bias(col, factor):
        arr = full[col].astype(float).to_numpy()
        arr[future_mask] = np.maximum(0.0, arr[future_mask] * (1.0 + factor))
        full[col] = np.round(arr, 2)

    booked_bump  = 0.010 + rng2.uniform(-0.003, 0.003) + global_drift
    open_dip     = -0.010 + rng2.uniform(-0.003, 0.003) + global_drift / 2
    after_wiggle = rng2.uniform(-0.004, 0.004) + global_drift / 3
    first_wiggle = rng2.uniform(-0.004, 0.004) - global_drift / 3

    if "BookedHours" in full.columns:   _bias("BookedHours", booked_bump)
    if "OpenHours" in full.columns:     _bias("OpenHours",  open_dip)
    if "After-Sales" in full.columns:   _bias("After-Sales", after_wiggle)
    if "First Visit" in full.columns:   _bias("First Visit", first_wiggle)

    # Re-enforce TotalHours consistency again
    approx2 = full["OpenHours"] + full["BlockedHours"] + full["BookedHours"]
    full["TotalHours"] = np.maximum(full["TotalHours"], approx2).round(2)

    # Convenience app columns
    full["ShiftDurationHours"] = (full["OpenHours"] + full["BlockedHours"] + full["BookedHours"]).round(2)
    full["60min_slots"] = np.floor(full["OpenHours"]).astype(int)
    full["30min_slots"] = np.round(full["OpenHours"] * 2).astype(int)
    if "OverlapHours" not in full.columns:
        full["OverlapHours"] = 0.0
    if "OpenHours_MB" not in full.columns:
        full["OpenHours_MB"] = full["OpenHours"].astype(float)

    # Weekly-style fields (FTE, percentages as 0..1 in *_week where applicable, and ratios)
    full = compute_week_fields(full)

    # Row-level daily percentage columns (0..100 scale) that the app expects
    full = compute_daily_percentages(full)

    # Make sure optional columns exist (flags/sales/appointments recent history)
    for c in OPTIONAL_APP_COLS:
        if c not in full.columns:
            full[c] = 0

    # Column order: keep template first, then any new columns we added
    template_cols = list(template_df.columns)
    new_cols = [c for c in full.columns if c not in template_cols]
    out_cols = [c for c in template_cols if c in full.columns] + new_cols
    return full[out_cols]

# --------------------------------- CLI --------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate daily shiftslots_YYYY-MM-DD.xlsx snapshots from a template with a rolling planning window."
    )
    p.add_argument("--template", required=True, help="Path to a real shiftslots_*.xlsx to use as schema")
    p.add_argument("--outdir", required=True, help="Directory to write generated files")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dates", nargs="+",
                     help="Snapshot dates: today yesterday month-start last-working or explicit YYYY-MM-DD ...")
    grp.add_argument("--range", nargs=2, metavar=("YYYY-MM-DD","YYYY-MM-DD"),
                     help="Generate a snapshot per day in this inclusive range")
    p.add_argument("--tz", default="Europe/Madrid")
    return p.parse_args()

def resolve_keyword(s: str, now: datetime, tz) -> datetime:
    s = s.lower()
    if s == "today":       return now
    if s == "yesterday":   return now - timedelta(days=1)
    if s == "month-start": return now.replace(day=1)
    if s == "last-working":
        d = now if now.weekday() < 5 else now - timedelta(days=now.weekday() - 4)
        if now.weekday() == 0:
            d = now - timedelta(days=3)
        return d
    return tz.localize(datetime.strptime(s, "%Y-%m-%d"))

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global template_df
    template_df = pd.read_excel(args.template)

    # Ensure ID columns so merges don't fail
    for c in ID_COLS:
        if c not in template_df.columns:
            template_df[c] = ""

    # Ensure optional app columns exist in template (harmless if duplicates)
    for c in OPTIONAL_APP_COLS:
        if c not in template_df.columns:
            template_df[c] = 0

    tz = pytz.timezone(args.tz)
    now = datetime.now(tz)

    if args.dates:
        snapshots = [resolve_keyword(s, now, tz) for s in args.dates]
    else:
        start = tz.localize(datetime.strptime(args.range[0], "%Y-%m-%d"))
        end   = tz.localize(datetime.strptime(args.range[1], "%Y-%m-%d"))
        snapshots = []
        cur = start
        while cur <= end:
            snapshots.append(cur)
            cur += timedelta(days=1)

    for snap in snapshots:
        out_path = os.path.join(args.outdir, f"shiftslots_{snap.strftime('%Y-%m-%d')}.xlsx")
        df_out = extend_to_full_window(template_df, snap, tz_str=args.tz)
        df_out.to_excel(out_path, index=False)
        print("Wrote", out_path)

if __name__ == "__main__":
    sys.exit(main())
