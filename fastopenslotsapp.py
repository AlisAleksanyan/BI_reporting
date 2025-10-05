import os, io, json, base64, calendar, requests
from io import BytesIO
from urllib.parse import quote
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, JsCode
from st_aggrid.shared import GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode, AgGridTheme, ExcelExportMode
import pytz
st.set_page_config(layout="wide")
# --- GitHub credentials from Streamlit secrets ---
github_secrets = st.secrets.get("github")
if not github_secrets:
    st.error("FATAL: Missing '[github]' section in Streamlit secrets!")
    st.stop()

GITHUB_TOKEN = github_secrets.get("token")
REPO_FULL = github_secrets.get("repo")
if not GITHUB_TOKEN or not REPO_FULL or "/" not in REPO_FULL:
    st.error("FATAL: Missing or malformed github.token / github.repo ('owner/name').")
    st.stop()

REPO_OWNER, REPO_NAME = REPO_FULL.split("/", 1)

# ---------- Helpers ----------
def find_last_working_day(start_date: datetime) -> datetime:
    cur = start_date
    while True:
        if cur.weekday() < 5:  # Mon-Fri
            return cur
        cur -= timedelta(days=1)

def get_first_iso_week_start_date_current_month() -> datetime:
    today = datetime.today()
    first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if first_day.weekday() == 6:  # Sunday -> move to Monday
        first_day += timedelta(days=1)
    _, _, iso_weekday = first_day.isocalendar()
    start_date = first_day - timedelta(days=iso_weekday - 1)
    return start_date.replace(hour=0, minute=0, second=0, microsecond=0)

def get_last_iso_week_end_date_current_month() -> datetime:
    today = datetime.today()
    last_day = today.replace(
        day=calendar.monthrange(today.year, today.month)[1],
        hour=0, minute=0, second=0, microsecond=0
    )
    _, _, iso_weekday = last_day.isocalendar()
    end_date = last_day + timedelta(weeks=4) + timedelta(days=(7 - iso_weekday))
    return end_date.replace(hour=0, minute=0, second=0, microsecond=0)

def get_start_and_end_of_current_month():
    today = datetime.today()
    month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day = calendar.monthrange(today.year, today.month)[1]
    month_end = today.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)
    return month_start, month_end

# -------------- GitHub helpers (commit-keyed caching) --------------

@st.cache_data(ttl="0", show_spinner=False)  # cache commit lookups ~30 min
def get_latest_commit(repo_owner: str, repo_name: str, path_in_repo: str, token: str):
    """
    Returns (sha, committed_datetime_utc) of the latest commit that touched the path.
    If no commits, returns (None, None).
    """
    encoded = quote(path_in_repo.replace("\\", "/"))
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={encoded}&page=1&per_page=1"
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        commits = r.json()
        if not commits:
            return None, None
        sha = commits[0]["sha"]
        dt_str = commits[0]["commit"]["committer"]["date"]  # ISO with Z
        committed_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return sha, committed_dt
    except Exception:
        return None, None


def _download_github_file(repo_owner: str, repo_name: str, path_in_repo: str, token: str) -> bytes | None:
    """
    Efficiently downloads a repo file via Contents API.
    Handles small (<1MB) and large files (via download_url).
    """
    path_in_repo = path_in_repo.replace("\\", "/")
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path_in_repo}"
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}

    meta = requests.get(url, headers=headers, timeout=30)
    if meta.status_code == 404:
        return None
    meta.raise_for_status()
    data = meta.json()

    if data.get("download_url"):
        raw = requests.get(data["download_url"], headers={'Authorization': f'token {token}'}, timeout=120)
        raw.raise_for_status()
        return raw.content

    if "content" in data and data["content"]:
        return base64.b64decode(data["content"])

    return None


@st.cache_data(show_spinner=False)  # cache by arguments; the SHA argument busts cache on file change
def load_excel_from_github(repo_owner: str, repo_name: str, path_in_repo: str, token: str, commit_sha: str, **pd_kwargs) -> pd.DataFrame | None:
    """
    Load Excel from GitHub; cache is keyed by commit_sha.
    When the file is updated on GitHub, its latest SHA changes → cache invalidates automatically.
    """
    raw = _download_github_file(repo_owner, repo_name, path_in_repo, token)
    if raw is None:
        return None
    try:
        return pd.read_excel(BytesIO(raw), **pd_kwargs)
    except Exception as e:
        st.warning(f"Error reading Excel '{path_in_repo}': {e}")
        return None


def load_best_effort_github(paths_in_repo: list[str], *, repo_owner: str, repo_name: str, token: str, **pd_kwargs):
    """
    Try a list of repo paths (today, yesterday, last working day...).
    Returns (df, used_path, commit_dt_utc) or (None, None, None).
    """
    for p in paths_in_repo:
        sha, commit_dt = get_latest_commit(repo_owner, repo_name, p, token)
        if not sha:
            continue
        df = load_excel_from_github(repo_owner, repo_name, p, token, sha, **pd_kwargs)
        if df is not None and not df.empty:
            return df, p, commit_dt
    return None, None, None

# ---- GitHub upsert helpers (create or update a file on repo) ----
def _get_file_sha_if_exists(repo_owner: str, repo_name: str, path_in_repo: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path_in_repo}"
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")

# --- helper: create/update a file in GitHub (idempotent) ---
def github_upsert_file(owner, repo, path, token, content_bytes, message):
    """Create or update a file via GitHub Contents API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
    # Get current SHA if file exists
    r = requests.get(url, headers=headers, timeout=20)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    put = requests.put(url, headers=headers, json=payload, timeout=30)
    put.raise_for_status()

# ---------- UI Styles ----------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { width: 100px; background-color: #cc0641; }
    .css-1lcbmhc { max-width: calc(100% - 200px); margin-left: -200px; }
    [data-testid="stSidebar"] label { color: white; font-weight: bold; }
    .sidebar-stats-box { color: white; font-weight: bold; border: 2px solid white; background-color: #cc0641;
                         padding: 10px; margin-bottom: 5px; border-radius: 5px; }
    .custom-title { font-size: 2em; color: #cc0641; font-weight: bold; text-align: center; margin-top: -50px;
                    margin-bottom: 20px; background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .circle{height:15px;width:15px;display:inline-block;border-radius:50%;margin-right:10px;}
    .red-circle{background-color:#cc0641;} .orange-circle{background-color:#f1b84b;} .green-circle{background-color:#95cd41;}
    .label-container{text-align:center;margin-bottom:20px;} .label-text{display:inline-block;vertical-align:middle;font-size:1em;margin-right:20px;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="custom-title">AGENDA APP</h1>', unsafe_allow_html=True)

# ---------- Dates / Paths ----------
# ---------- Dates / Paths (unchanged date logic) ----------
folder_path_in_repo = "shiftslots"  # this is the repo folder

month_start_date, month_end_date = get_start_and_end_of_current_month()
current_date = datetime.now()
yesterday_date = current_date - timedelta(days=1)

# Monday logic -> prefer last Friday when needed
if current_date.weekday() == 0:  # Monday
    last_working_day = find_last_working_day(current_date - timedelta(days=3))
else:
    last_working_day = find_last_working_day(current_date)

# Build repo-relative file paths
today_repo_path = f"{folder_path_in_repo}/shiftslots_{current_date.strftime('%Y-%m-%d')}.xlsx"
yday_repo_path = f"{folder_path_in_repo}/shiftslots_{yesterday_date.strftime('%Y-%m-%d')}.xlsx"
lwd_repo_path = f"{folder_path_in_repo}/shiftslots_{last_working_day.strftime('%Y-%m-%d')}.xlsx"
month_first_repo_path = f"{folder_path_in_repo}/shiftslots_{month_start_date.strftime('%Y-%m-%d')}.xlsx"

with st.spinner("⬇️ Loading agenda data from GitHub..."):
    shift_slots, used_repo_path, commit_dt_utc = load_best_effort_github(
        [today_repo_path, yday_repo_path, lwd_repo_path],
        repo_owner=REPO_OWNER, repo_name=REPO_NAME, token=GITHUB_TOKEN
    )
    if shift_slots is None:
        st.error("FATAL: No valid agenda data could be loaded from GitHub.")
        st.stop()

    # nice "last updated" in Europe/Madrid (no file path)
    if commit_dt_utc:
        madrid = pytz.timezone("Europe/Madrid")
        st.write("Last updated:", commit_dt_utc.astimezone(madrid).strftime("%Y-%m-%d %H:%M:%S"))

    # Load “yesterday” fallback (if not already used)
    shift_slots_yesterday, _, _ = load_best_effort_github(
        [yday_repo_path, lwd_repo_path],
        repo_owner=REPO_OWNER, repo_name=REPO_NAME, token=GITHUB_TOKEN
    )

    # Month-first (your “sep6”)
    shift_slots_sep6, _, _ = load_best_effort_github(
        [month_first_repo_path],
        repo_owner=REPO_OWNER, repo_name=REPO_NAME, token=GITHUB_TOKEN
    )

# Secondary files from repo (adjust paths if needed)
hcm, _, _ = load_best_effort_github(
    ["output/hcm_sf_merged.xlsx"],
    repo_owner=REPO_OWNER, repo_name=REPO_NAME, token=GITHUB_TOKEN
)
cluster, _, _ = load_best_effort_github(
    ["output/cluster_data.xlsx"],
    repo_owner=REPO_OWNER, repo_name=REPO_NAME, token=GITHUB_TOKEN
)

# ---------- Sidebar: ISO week / Region / Area / Category / Shop / Top100 ----------
current_iso_year, current_iso_week, _ = datetime.now().isocalendar()

if 'date_dt' not in shift_slots.columns and 'date' in shift_slots.columns:
    shift_slots = shift_slots.copy()
    shift_slots['date_dt'] = pd.to_datetime(shift_slots['date'], errors='coerce')

available_weeks = sorted(shift_slots['iso_week'].dropna().unique().tolist()) if 'iso_week' in shift_slots.columns else []
current_week_index = available_weeks.index(current_iso_week) if current_iso_week in available_weeks else 0
iso_week_filter = st.sidebar.selectbox('Seleccione la Semana ISO:', available_weeks, index=current_week_index if available_weeks else 0)
# Calculate the previous ISO week and year based on the selected ISO week
selected_iso_year = datetime.now().year  # Assuming current year, adjust if you have a different dataset
previous_iso_week = iso_week_filter - 1
previous_iso_year = selected_iso_year if previous_iso_week == 0 else selected_iso_year - 1
previous_iso_week = 52 if (pd.Timestamp(f"{previous_iso_year}-12-28").isocalendar()[1] == 52) else 53

region_list = sorted(shift_slots['Region'].dropna().unique().tolist()) if 'Region' in shift_slots.columns else []
region_options = ["All"] + region_list
selected_region = st.sidebar.selectbox('Select Region:', options=region_options, index=0)

filtered_by_region = shift_slots if selected_region == "All" else shift_slots.loc[shift_slots['Region'] == selected_region].copy()

area_list = sorted(filtered_by_region['Area'].dropna().unique().tolist()) if 'Area' in filtered_by_region.columns else []
area_options = ["All"] + area_list
selected_area = st.sidebar.selectbox('Select Area:', options=area_options, index=0)

filtered_by_area = filtered_by_region if selected_area == "All" else filtered_by_region.loc[filtered_by_region['Area'] == selected_area].copy()

shop_type_list = sorted(filtered_by_area['CATEGORY'].dropna().unique().tolist()) if 'CATEGORY' in filtered_by_area.columns else []
shop_type_options = ["All"] + shop_type_list
selected_category = st.sidebar.multiselect('Select Shop Type:', options=shop_type_options, default=["All"])
if "All" in selected_category:
    selected_category = shop_type_list

shop_list = sorted(filtered_by_area['Shop[Name]'].dropna().unique().tolist()) if 'Shop[Name]' in filtered_by_area.columns else []
shop_options = ["All"] + shop_list
selected_shop = st.sidebar.selectbox('Select Shop:', options=shop_options, index=0)

# Build Top 100 (today only)
today_dt = datetime.today().date()
mask_today = shift_slots['date_dt'].dt.date.eq(today_dt) if 'date_dt' in shift_slots.columns else pd.Series(False, index=shift_slots.index)
shift_slots_top_100 = shift_slots.loc[mask_today].copy()
col_sales12 = '[Last_12M_Cumulative_HA_Sales_Value]'
if col_sales12 in shift_slots_top_100.columns:
    shift_slots_top_100[col_sales12] = shift_slots_top_100[col_sales12].fillna(0)
    top_100_shops = (
        shift_slots_top_100.groupby('Shop[Name]')[col_sales12]
        .sum().reset_index()
        .sort_values(col_sales12, ascending=False)
        .head(100)['Shop[Name]'].tolist()
    )
else:
    top_100_shops = []

selected_top_shop = st.sidebar.selectbox('Select Top 100 Shops:', options=["All", "Top 100"], index=0)

# ---------- Filters (cached) ----------
@st.cache_data(ttl="30m")
def filter_data(data, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, week_column='iso_week'):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data
    mask = pd.Series(True, index=df.index)
    if week_column in df.columns and iso_week_filter is not None:
        mask &= df[week_column] == iso_week_filter
    if selected_region != "All" and 'Region' in df.columns:
        mask &= df['Region'] == selected_region
    if selected_area != "All" and 'Area' in df.columns:
        mask &= df['Area'] == selected_area
    if selected_shop != "All" and 'Shop[Name]' in df.columns:
        mask &= df['Shop[Name]'] == selected_shop
    if selected_category and 'CATEGORY' in df.columns:
        mask &= df['CATEGORY'].isin(selected_category)
    if selected_top_shop == "Top 100" and 'Shop[Name]' in df.columns:
        mask &= df['Shop[Name]'].isin(top_100_shops)
    return df.loc[mask].copy()

@st.cache_data(ttl="30m")
def filter_data_tab4(data, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, week_column='iso_week'):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data
    mask = pd.Series(True, index=df.index)
    if week_column in df.columns and iso_week_filter is not None:
        mask &= df[week_column] == iso_week_filter
    if selected_region != "All" and 'Region' in df.columns:
        mask &= df['Region'] == selected_region
    if selected_area != "All" and 'Area' in df.columns:
        mask &= df['Area'] == selected_area
    # Note: your HCM has 'Shop Name' not 'Shop[Name]'
    if selected_shop != "All":
        if 'Shop Name' in df.columns:
            mask &= df['Shop Name'] == selected_shop
        elif 'Shop[Name]' in df.columns:
            mask &= df['Shop[Name]'] == selected_shop
    if selected_category and 'CATEGORY' in df.columns:
        mask &= df['CATEGORY'].isin(selected_category)
    if selected_top_shop == "Top 100":
        col = 'Shop Name' if 'Shop Name' in df.columns else 'Shop[Name]' if 'Shop[Name]' in df.columns else None
        if col:
            mask &= df[col].isin(top_100_shops)
    return df.loc[mask].copy()

@st.cache_data(ttl="30m")
def filter_hcm_data(data, selected_region, selected_area, selected_shop, selected_category, selected_top_shop):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data
    mask = pd.Series(True, index=df.index)
    if selected_region != "All" and 'Region' in df.columns:
        mask &= df['Region'] == selected_region
    if selected_area != "All" and 'Area' in df.columns:
        mask &= df['Area'] == selected_area
    if selected_shop != "All":
        if 'Shop Name' in df.columns:
            mask &= df['Shop Name'] == selected_shop
        elif 'Shop[Name]' in df.columns:
            mask &= df['Shop[Name]'] == selected_shop
    if selected_category and 'CATEGORY' in df.columns:
        mask &= df['CATEGORY'].isin(selected_category)
    if selected_top_shop == "Top 100":
        col = 'Shop Name' if 'Shop Name' in df.columns else 'Shop[Name]' if 'Shop[Name]' in df.columns else None
        if col:
            mask &= df[col].isin(top_100_shops)
    return df.loc[mask].copy()

@st.cache_data(ttl="30m")
def filter_hcp_shift_slots(data, selected_region, selected_area, selected_shop, selected_category, selected_top_shop):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data
    mask = pd.Series(True, index=df.index)
    if selected_region != "All" and 'Region' in df.columns:
        mask &= df['Region'] == selected_region
    if selected_area != "All" and 'Area' in df.columns:
        mask &= df['Area'] == selected_area
    if selected_shop != "All" and 'Shop[Name]' in df.columns:
        mask &= df['Shop[Name]'] == selected_shop
    if selected_category and 'CATEGORY' in df.columns:
        mask &= df['CATEGORY'].isin(selected_category)
    if selected_top_shop == "Top 100" and 'Shop[Name]' in df.columns:
        mask &= df['Shop[Name]'].isin(top_100_shops)
    return df.loc[mask].copy()

# ---------- Apply filters / proceed with your existing pipeline ----------
filtered_data = filter_data(shift_slots, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, 'iso_week')
filtered_data_yesterday = filter_data(shift_slots_yesterday, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, 'iso_week') if shift_slots_yesterday is not None else pd.DataFrame()
filtered_data_mf = filter_data(shift_slots_sep6, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, 'iso_week') if shift_slots_sep6 is not None else pd.DataFrame()

weekly_shift_slots = filter_hcp_shift_slots(shift_slots, selected_region, selected_area, selected_shop, selected_category, selected_top_shop)
weekly_shift_slots_yesterday = filter_hcp_shift_slots(shift_slots_yesterday, selected_region, selected_area, selected_shop, selected_category, selected_top_shop) if shift_slots_yesterday is not None else pd.DataFrame()
weekly_shift_sep6 = filter_hcp_shift_slots(shift_slots_sep6, selected_region, selected_area, selected_shop, selected_category, selected_top_shop) if shift_slots_sep6 is not None else pd.DataFrame()
weekly_shift_slots_tab8 = weekly_shift_slots.copy()

if weekly_shift_slots_tab8.empty:
    st.warning("No shops available after filtering for Top 100. Please try another filter.")
    st.stop()

filtered_hcm = filter_hcm_data(hcm, selected_region, selected_area, selected_shop, selected_category, selected_top_shop)
filtered_hcm_4 = filter_data_tab4(hcm, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, 'iso_week') if hcm is not None else pd.DataFrame()
filtered_cluster = filter_data(cluster, iso_week_filter, selected_region, selected_area, selected_shop, selected_category, selected_top_shop, 'iso_week') if cluster is not None else pd.DataFrame()
# Filter data for the previous ISO week without applying the current filters (directly from shift_slots)
mask_prev = shift_slots['iso_week'] == previous_iso_week
if selected_region != "All":
    mask_prev &= shift_slots['Region'] == selected_region
if selected_area != "All":
    mask_prev &= shift_slots['Area'] == selected_area
if selected_shop != "All":
    mask_prev &= shift_slots['Shop[Name]'] == selected_shop

previous_week_data = shift_slots.loc[mask_prev].copy()

# SIDEBAR STATISTICS
# Calculate Open Hours for the current and previous weeks
open_hours_this_week = filtered_data['60min_slots'].sum()
open_hours_last_week = previous_week_data['60min_slots'].sum()
# Calculate percentage change from last week, with checks to prevent division by zero
if open_hours_last_week != 0:
    change_from_last_week = ((open_hours_this_week - open_hours_last_week) / open_hours_last_week) * 100
else:
    change_from_last_week = 0  # Or handle differently, depending on your needs
# Calculate the start and end dates for the selected ISO week
selected_week_start = pd.Timestamp(selected_iso_year, 1, 1) + pd.offsets.Week(weekday=0) * (int(iso_week_filter) - 1)
selected_week_end = selected_week_start + pd.offsets.Week(weekday=6)
today = pd.Timestamp(datetime.now().date())
end_of_month = today.replace(day=1) + pd.offsets.MonthEnd(0)
month_mask = (shift_slots['date_dt'] >= today) & (shift_slots['date_dt'] <= end_of_month)
month_to_go_data = shift_slots.loc[month_mask].copy()
open_hours_month_to_go = month_to_go_data['60min_slots'].sum()
# Determine the best configured region      
best_configured_region = shift_slots.groupby('Region')['SaturationPercentage'].mean().idxmax() if not filtered_data.empty else 'N/A'
st.sidebar.markdown(f"<div class='sidebar-stats-box'>1H slots open for the selected week: {open_hours_this_week:,.0f}</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='sidebar-stats-box'>1H slots open for the upcoming month: {open_hours_month_to_go:,.0f}</div>", unsafe_allow_html=True)


# Aggregating data by GT_ShopCode__c, Shop[Name], date, and weekday
aggregated_data = filtered_data.groupby(['GT_ShopCode__c', 'Shop[Name]', 'CATEGORY','date', 'weekday']).agg(
    OpenHours=('OpenHours', 'sum'),
    TotalHours=('TotalHours', 'sum'),
    BlockedHoursPercentage=('BlockedHoursPercentage', 'mean')
).reset_index()
aggregated_data = aggregated_data.copy()
aggregated_data['date'] = pd.to_datetime(aggregated_data['date']).dt.date

aggregated_data_tab11 = filtered_data.groupby(['Region', 'Area', 'GT_ShopCode__c', 'Shop[Name]', 'CATEGORY','iso_week']).agg(
    OpenHours=('OpenHours', 'sum'),
    TotalHours=('TotalHours', 'sum'),
    AfterSales= ('After-Sales', 'sum'),
    FirstVisit=('First Visit', 'sum'),
    PreSales=('Pre-Sales', 'sum'),
    BlockedHours=('BlockedHours', 'sum'),
    PersonalBlockHours=('PersonalBlockHours', 'sum'),
    BusinessBlockHours = ('BusinessBlockHours', 'sum'),
    AdminBlockHours=('AdminBlockHours', 'sum'),
    WeBlockHours=('WeBlockHours', 'sum'),
    OtherBlockHours=('OtherBlockHours', 'sum'),
    ratioadelanto = ('ratioadelanto', 'first'),
    ratioadelanto_area = ('ratioadelanto_area', 'first'),
    ratioadelanto_reg = ('ratioadelanto_reg', 'first'),
    Upcoming_3_Weeks_Sum=('Upcoming_3_Weeks_Sum', 'first'),
    Upcoming_3_Weeks_Sum_area = ('Upcoming_3_Weeks_Sum_area', 'first'),
    Upcoming_3_Weeks_Sum_reg = ('Upcoming_3_Weeks_Sum_reg', 'first'),
    Current_Week_Sum=('Current_Week_Sum', 'first'),
    Current_Week_Sum_area = ('Current_Week_Sum_area', 'first'),
    Current_Week_Sum_reg = ('Current_Week_Sum_reg', 'first')
).reset_index()
aggregated_data_tab11_yesterday = filtered_data_yesterday.groupby(['Region', 'Area', 'GT_ShopCode__c', 'Shop[Name]', 'CATEGORY','iso_week']).agg(
    OpenHours=('OpenHours', 'sum'),
    TotalHours=('TotalHours', 'sum'),
    AfterSales= ('After-Sales', 'sum'),
    FirstVisit=('First Visit', 'sum'),
    PreSales=('Pre-Sales', 'sum'),
    BlockedHours=('BlockedHours', 'sum'),
    PersonalBlockHours=('PersonalBlockHours', 'sum'),
    BusinessBlockHours = ('BusinessBlockHours', 'sum'),
    AdminBlockHours=('AdminBlockHours', 'sum'),
    WeBlockHours=('WeBlockHours', 'sum'),
    OtherBlockHours=('OtherBlockHours', 'sum'),
    ratioadelanto = ('ratioadelanto', 'first'),
    ratioadelanto_area = ('ratioadelanto_area', 'first'),
    ratioadelanto_reg = ('ratioadelanto_reg', 'first'),
    Upcoming_3_Weeks_Sum=('Upcoming_3_Weeks_Sum', 'first'),
    Upcoming_3_Weeks_Sum_area = ('Upcoming_3_Weeks_Sum_area', 'first'),
    Upcoming_3_Weeks_Sum_reg = ('Upcoming_3_Weeks_Sum_reg', 'first'),
    Current_Week_Sum=('Current_Week_Sum', 'first'),
    Current_Week_Sum_area = ('Current_Week_Sum_area', 'first'),
    Current_Week_Sum_reg = ('Current_Week_Sum_reg', 'first')
).reset_index()
aggregated_data_tab11_mf = filtered_data_mf.groupby(['Region', 'Area', 'GT_ShopCode__c', 'Shop[Name]', 'CATEGORY','iso_week']).agg(
    OpenHours=('OpenHours', 'sum'),
    TotalHours=('TotalHours', 'sum'),
    AfterSales= ('After-Sales', 'sum'),
    FirstVisit=('First Visit', 'sum'),
    PreSales=('Pre-Sales', 'sum'),
    BlockedHours=('BlockedHours', 'sum'),
    PersonalBlockHours=('PersonalBlockHours', 'sum'),
    BusinessBlockHours = ('BusinessBlockHours', 'sum'),
    AdminBlockHours=('AdminBlockHours', 'sum'),
    WeBlockHours=('WeBlockHours', 'sum'),
    OtherBlockHours=('OtherBlockHours', 'sum'),
    ratioadelanto = ('ratioadelanto', 'first'),
    ratioadelanto_area = ('ratioadelanto_area', 'first'),
    ratioadelanto_reg = ('ratioadelanto_reg', 'first'),
    Upcoming_3_Weeks_Sum=('Upcoming_3_Weeks_Sum', 'first'),
    Upcoming_3_Weeks_Sum_area = ('Upcoming_3_Weeks_Sum_area', 'first'),
    Upcoming_3_Weeks_Sum_reg = ('Upcoming_3_Weeks_Sum_reg', 'first'),
    Current_Week_Sum=('Current_Week_Sum', 'first'),
    Current_Week_Sum_area = ('Current_Week_Sum_area', 'first'),
    Current_Week_Sum_reg = ('Current_Week_Sum_reg', 'first')
).reset_index()
for df in [aggregated_data_tab11, aggregated_data_tab11_yesterday, aggregated_data_tab11_mf]:
    # Ensure Ratioadelanto columns exist before trying to use them
    if 'ratioadelanto' in df.columns:
         df["Ratio Adelanto"] = df["ratioadelanto"].round(2)
    if 'ratioadelanto_reg' in df.columns:
        df["Ratio Adelanto Reg"] = df["ratioadelanto_reg"].round(2) # Assuming you might want this
    if 'ratioadelanto_area' in df.columns:
        df["Ratio Adelanto Area"] = df["ratioadelanto_area"].round(2) # Assuming you might want this


# Rename columns globally
aggregated_data_tab11.rename(columns={"Region": "Region", "Area": "Area", "Shop[Name]": "Shop"}, inplace=True)
aggregated_data_tab11_mf.rename(columns={"Region": "Region", "Area": "Area", "Shop[Name]": "Shop"}, inplace=True)
aggregated_data_tab11_yesterday.rename(columns={"Region": "Region", "Area": "Area", "Shop[Name]": "Shop"}, inplace=True)

openhourdata= filtered_data.groupby(['GT_ShopCode__c', 'Shop[Name]','CATEGORY','iso_week', 'date', 'weekday']).agg(
    ShiftHours=('ShiftDurationHours', 'sum'),
    BlockedHours=('BlockedHours', 'sum'),
    OverlapHours=('OverlapHours', 'sum'),
    OpenHours=('OpenHours', 'sum'),
    OpenHours_MB=('OpenHours_MB', 'sum'),
    BookedHours=('BookedHours', 'sum'),
    AfterSalesSlots=('After-Sales', 'sum'),
    HAsalesunits=('[HA_Sales__Units_]', 'sum'),
    Agendaappts=('[Agenda_Appointments]', 'sum'),
    Agendacancelled=('[Appointments_Cancelled]', 'sum'),
    TMK=('tmk_flag', 'max'),
    extra_pos=('extra_pos', 'max'),
    Last_3M_Sales=('[Last_3M_Sales_Value]', 'sum'),
    Last_3M_Appointments=('[Last_3M_Appointments]', 'sum'),
    Last_3M_Cancelled=('[Last_3M_Cancelled]', 'sum'),
    Slots_30min=('30min_slots', 'sum'),
    Slots_60min=('60min_slots', 'sum')
).reset_index()
openhourdata['App2Sale'] = openhourdata['HAsalesunits']/ (openhourdata['Agendaappts']+ openhourdata['Agendacancelled'])
openhourdata['App2Sale']=openhourdata['App2Sale'].fillna(0)
# Convert the date column to datetime format temporarily for filtering
openhourdata['is_weekend'] = pd.to_datetime(openhourdata['date'], errors='coerce').dt.dayofweek
# Filter out Saturdays (5) and Sundays (6)
openhourdata = openhourdata[
    ~openhourdata['is_weekend'].isin([5, 6])
]
# Drop the helper column after filtering
openhourdata = openhourdata.drop(columns=['is_weekend'])
openhourdata['AfterSalesHours'] = openhourdata['AfterSalesSlots'] * 5 / 60
# Calculate percentages
openhourdata['AfterSalesHours%'] = (openhourdata['AfterSalesHours'] / openhourdata['ShiftHours']) * 100
openhourdata['BlockedHours%'] = (openhourdata['BlockedHours'] / openhourdata['ShiftHours']) * 100
openhourdata['BookedHours%'] = (openhourdata['BookedHours'] / openhourdata['ShiftHours']) * 100
openhourdata['OpenHours%'] = (openhourdata['OpenHours'] / openhourdata['ShiftHours']) * 100

# Replace infinite and NaN values with 0
openhourdata.replace([np.inf, -np.inf], np.nan, inplace=True)


# Ensure date column is of type date
openhourdata['date'] = pd.to_datetime(openhourdata['date'], errors='coerce')
weekly_shift_slots_tab8 = weekly_shift_slots_tab8.rename(columns={
 '[HA_Sales__Units_]': 'HAsalesunits',
    '[Agenda_Appointments]': 'Agendaappts',
    '[Appointments_Cancelled]': 'Agendacancelled',
    'tmk_flag': 'TMK',
    '[Last_3M_Sales_Value]': 'Last_3M_Sales',
    '[Last_3M_Appointments]': 'Last_3M_Appointments',
    '[Last_3M_Cancelled]': 'Last_3M_Cancelled',
    '30min_slots': 'Slots_30min',
    '60min_slots': 'Slots_60min',
})
# 1) Ensure dtypes first
# ---- weekly_shift_slots_tab8: clean, non-fragmenting, warning-free ----
weekly_shift_slots_tab8 = weekly_shift_slots_tab8.copy()
weekly_shift_slots_tab8['date'] = pd.to_datetime(weekly_shift_slots_tab8['date'], errors='coerce')

# numeric-only sanitizing
num_cols = weekly_shift_slots_tab8.select_dtypes(include='number').columns
weekly_shift_slots_tab8[num_cols] = (
    weekly_shift_slots_tab8[num_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
)
# datetimes: use NaT
for c in weekly_shift_slots_tab8.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
    weekly_shift_slots_tab8[c] = weekly_shift_slots_tab8[c].fillna(pd.NaT)

# hours + denominator
weekly_shift_slots_tab8 = weekly_shift_slots_tab8.rename(columns={'ShiftDurationHours': 'ShiftHours'})
weekly_shift_slots_tab8['AfterSalesHours'] = weekly_shift_slots_tab8['After-Sales'] * 5 / 60
den = weekly_shift_slots_tab8['ShiftHours'].replace(0, np.nan)

# add ONLY the % columns once (avoid duplicate AfterSalesHours)
new_cols = pd.DataFrame({
    'AfterSalesHours%': (weekly_shift_slots_tab8['AfterSalesHours'] / den) * 100,
    'BlockedHours%':    (weekly_shift_slots_tab8['BlockedHours']    / den) * 100,
    'BookedHours%':     (weekly_shift_slots_tab8['BookedHours']     / den) * 100,
    'OpenHours%':       (weekly_shift_slots_tab8['OpenHours']       / den) * 100,
}, index=weekly_shift_slots_tab8.index).fillna(0)

weekly_shift_slots_tab8 = pd.concat([weekly_shift_slots_tab8, new_cols], axis=1)

# weekdays only
weekly_shift_slots_tab8 = weekly_shift_slots_tab8[~weekly_shift_slots_tab8['date'].dt.dayofweek.isin([5, 6])]

# optional: defragment after many ops
weekly_shift_slots_tab8 = weekly_shift_slots_tab8.copy()

# Replace infinite and NaN values with 0
pd.set_option('future.no_silent_downcasting', True)

# Ensure date column is of type date
weekly_shift_slots_tab8['date'] = pd.to_datetime(weekly_shift_slots_tab8['date'], errors='coerce')
def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
        processed_data = output.getvalue()
        return processed_data

def convert_saturation_to_excel(df, group_col, is_comparison=False):
    # Build export DataFrame with same headers as AgGrid.
    export_data = pd.DataFrame()
    export_data[group_col] = df[group_col]
    
    # Determine week numbers based on columns that have 'TotalHours_'
    week_nums = sorted({col.split('_')[1] for col in df.columns if 'TotalHours_' in col}, key=lambda x: int(x))
    
    # Dynamically define ratio and upcoming field based on group_col
    if group_col == 'Shop':
        ratio_field = "ratioadelanto"
        upcoming_field = "Upcoming_3_Weeks_Sum"
    elif group_col == 'Area':
        ratio_field = "ratioadelanto_area"
        upcoming_field = "Upcoming_3_Weeks_Sum_area"
    elif group_col == 'Region':
        ratio_field = "ratioadelanto_reg"
        upcoming_field = "Upcoming_3_Weeks_Sum_reg"
    else:
        ratio_field = "ratioadelanto"
        upcoming_field = "Upcoming_3_Weeks_Sum"
    
    fields_order = [
        "FTE", "SaturationPercentage", "FirstVisitPercentage",
        "PreSalesPercentage", "AfterSalesPercentage",
        "PersonalBlocksPercentage", "BusinessBlocksPercentage",
        "AdminBlocksPercentage", "WeBlocksPercentage",
        ratio_field, upcoming_field
    ]
    display_names = {
        "FTE": "FTE",
        "SaturationPercentage": "Saturation %",
        "FirstVisitPercentage": "First Visit %",
        "PreSalesPercentage": "Pre-Sales %",
        "AfterSalesPercentage": "After-Sales %",
        "PersonalBlocksPercentage": "Personal B.%",
        "BusinessBlocksPercentage": "Business B. %",
        "AdminBlocksPercentage": "Admin B.%",
        "WeBlocksPercentage": "WE B.%",
        ratio_field: "Ratio Adelanto",
        upcoming_field: "Appts_Upcoming_3_Weeks_Sum"
    }

    # Build export columns, adjusting SaturationPercentage values by dividing by 100.
    for week in week_nums:
        for field in fields_order:
            col_name = f"{field}_{week}"
            if col_name in df.columns:
                new_col_name = f"Week {week} - {display_names[field]}"
                if field == "SaturationPercentage":
                    # Divide by 100 to convert to Excel decimal format.
                    export_data[new_col_name] = df[col_name] / 100
                else:
                    export_data[new_col_name] = df[col_name]

    # Write export_data to Excel with XlsxWriter and apply number formats.
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_data.to_excel(writer, sheet_name="Data", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Data"]

        # Define cell formats.
        # For FTE and Ratio Adelanto, use "0.0" or "0.0%" depending on comparison mode.
        fte_format_str = "0.0%" if is_comparison else "0.0"
        fte_format = workbook.add_format({"num_format": fte_format_str})
        percentage_format = workbook.add_format({"num_format": "0.0%"})
        
        # Apply formats: any header containing "%" gets the percentage format.
        for idx, col in enumerate(export_data.columns):
            if "%" in col:
                worksheet.set_column(idx, idx, None, percentage_format)
            elif "FTE" in col or "Ratio Adelanto" in col:
                worksheet.set_column(idx, idx, None, fte_format)
            else:
                worksheet.set_column(idx, idx, None)
    return output.getvalue()
    
tab6, tab12, tab4, tab9= st.tabs(["Summary", "Saturation","HCM vs SF", "HRBP Tasks"])

with tab4:
    url1 = "www.linkedin.com/in/alisaaleksanyan"
    url2 = "www.linkedin.com/in/alisaaleksanyan"

    if filtered_hcm_4.empty:
        st.warning("No shops found for the selected filter criteria.")

    st.markdown(f'''
        :green[*Goal: Make sure the contracted hours in the HR system (HCM) match the agenda hours configured in Salesforce (SF).  
        On this tab, you can check the current employee assignments for your area.  
        If you spot any discrepancies that need fixing, please submit a ticket:*]
        - [Open HR Ticket]({url1})
        - [Open Support Ticket]({url2})
        ''')


    filtered_hcm_file = "hcm_sf_file.xlsx"
    column_order = [
    'Region', 'Area', 'Shop Name', 'CATEGORY',  # 'CATEGORY' will be renamed to 'Tipo'
    'Shop Code', 'iso_year', 'iso_week',  
    'Resource Name', 'Personal Number', 
    'Duración SF', 'Duración HCM', 'Diferencia de hcm duración']

    # Rename columns before saving
    hcm_sf_file = filtered_hcm_4[column_order].rename(columns={'CATEGORY': 'Tipo'})
    hcm_sf_file.to_excel(filtered_hcm_file, index=False)

    with open(filtered_hcm_file, "rb") as file:
        st.download_button(
            label="Download as Excel",
            data=file,
            file_name="filtered_hcm_4.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    # Ensure 'iso_year_week' is present in 'filtered_hcm'
    filtered_hcm_4['iso_year_week'] = (
        filtered_hcm_4['iso_year'].astype(str) + '_' +
        filtered_hcm_4['iso_week'].astype(str).str.zfill(2)
    )

    def iso_year_start(iso_year):
        """Return the Gregorian calendar date of the first day of the given ISO year."""
        fourth_jan = datetime(iso_year, 1, 4)
        delta = timedelta(days=fourth_jan.isoweekday() - 1)
        return fourth_jan - delta

    def iso_to_gregorian(iso_year, iso_week, iso_day):
        """Return the Gregorian calendar date for the given ISO year, week, and day."""
        year_start = iso_year_start(iso_year)
        return year_start + timedelta(days=iso_day - 1, weeks=iso_week - 1)

    filtered_hcm_4['week_start_date'] = filtered_hcm_4.apply(
        lambda row: iso_to_gregorian(row['iso_year'], row['iso_week'], 1) if pd.notnull(row['iso_year']) and pd.notnull(row['iso_week']) else pd.NaT, 
        axis=1
    )

    filtered_hcm_4['week_end_date'] = filtered_hcm_4.apply(
        lambda row: iso_to_gregorian(row['iso_year'], row['iso_week'], 7) if pd.notnull(row['iso_year']) and pd.notnull(row['iso_week']) else pd.NaT, 
        axis=1
    )

        # Get current month and year
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    def week_in_current_month(row):
        """Check if the week overlaps with the current month."""
        month_start = datetime(current_year, current_month, 1)
        month_end = datetime(current_year, current_month, calendar.monthrange(current_year, current_month)[1])
        return (row['week_start_date'] <= month_end) and (row['week_end_date'] >= month_start)
    resource_name_options = filtered_hcm_4['Resource Name'].dropna().unique()
    selected_resource_name = st.selectbox(
        "Filter for Employee", 
        options=["All"] + list(resource_name_options),
        index=0
    )
    if selected_resource_name != "All":
            filtered_hcm_4 = filtered_hcm_4[filtered_hcm_4['Resource Name'] == selected_resource_name]

    iso_week_filter = int(iso_week_filter)

    #filtered_hcm_tab4 = filtered_hcm_4[filtered_hcm_4.apply(week_in_current_month, axis=1)]
        # Ensure there is data to process
    if filtered_hcm_4.empty:
        st.warning("No data available for the selected filters in the current month.")
        st.stop()
    # Pivot the table for Tab 4
    pivot_table_tab4 = filtered_hcm_4.pivot_table(
        index=['Resource Name','Personal Number','Employee Assignment[Primary Assignment]','Shop Name'],
        columns='iso_year_week',
        values=['Duración SF', 'Duración HCM', 'Diferencia de hcm duración'],
        aggfunc='sum',
        fill_value=0
    )

    # Flatten the columns for display
    pivot_table_tab4.columns = [
        f"Week_{col[1]}_{'SF' if col[0] == 'Duración SF' else 'HCM' if col[0] == 'Duración HCM' else 'Delta'}"
        for col in pivot_table_tab4.columns.to_flat_index()
    ]

    pivot_table_tab4_reset = pivot_table_tab4.reset_index()
    pivot_table_tab4_reset.columns = [col.replace(' ', '_') for col in pivot_table_tab4_reset.columns]
    if pivot_table_tab4_reset.empty:
            st.warning("No data available to display.")
            st.stop()

    # Format all numeric columns to one decimal point
    numeric_columns_in_pivot_tab4 = pivot_table_tab4_reset.select_dtypes(include=['float64', 'int64']).columns
    pivot_table_tab4_reset[numeric_columns_in_pivot_tab4] = pivot_table_tab4_reset[numeric_columns_in_pivot_tab4].round(1)

    # Create the DataFrame for AgGrid
    df_tab4 = pivot_table_tab4_reset    
    #df_tab4["GroupKey"] = df_tab4["Resource_Name"] + " "+ "(" + df_tab4["Personal_Number"].astype(str)  + ")"
    df_tab4["GroupKey"] = (df_tab4["Resource_Name"] + " ("+ df_tab4["Personal_Number"].map("{:.0f}".format) + ")")

    if df_tab4.empty:
        st.warning("No data available for the selected filters. Adjust your selection and try again.")
        st.stop()
     # JavaScript code for cell styling
    js_code = JsCode("""
        function(params) {
            var deltaField = params.colDef.field.replace('SF', 'Delta')
                                                .replace('HCM', 'Delta');
            var deltaValue;
            if (params.node.rowPinned === 'top') {
                deltaValue = params.data ? params.data[deltaField] : null;
                return {'fontWeight': 'bold', 'backgroundColor': '#e0e0e0', 'color': 'black'};
            } else if (params.node.group) {
                deltaValue = params.node.aggData ? params.node.aggData[deltaField] : null;
            } else {
                deltaValue = params.data ? params.data[deltaField] : null;
            }
            if (deltaValue === 0) {
                return {'backgroundColor': '#95cd41', 'color': 'black', 'fontWeight': 'bold'};
            } else if (deltaValue > 0) {
                return {'backgroundColor': '#f1b84b', 'color': 'black', 'fontWeight': 'bold'};
            } else {
                return {'backgroundColor': '#cc0641', 'color': 'white', 'fontWeight': 'bold'};
            }
        }
    """)

    # Custom CSS for styling the grid
    custom_css = {
        ".ag-header-cell": {
            "background-color": "#cc0641 !important",
            "color": "white !important",
            "font-weight": "bold",
            "padding": "4px"
        },
        ".ag-header-group-cell": {
            "background-color": "#cc0641 !important",
            "color": "white !important",
            "font-weight": "bold",
        },
        ".ag-cell": {
            "padding": "2px",
            "font-size": "12px"
        },
        ".ag-theme-streamlit .ag-row": {
            "max-height": "30px"
        },
        ".ag-theme-streamlit .ag-root-wrapper": {
            "border": "2px solid #cc0641",
            "border-radius": "5px"
        }
    }

    # Define the column definitions
    columnDefs = [
            {
            "headerName": "Resource Name",
            "field": "GroupKey",
            "rowGroup": True,
            "hide": True  # Hide the original column
        },
        {
            "headerName": "Shop Name",
            "field": "Shop_Name",
            "hide": True,
            "width": 0,
            "maxWidth": 0,
            "minWidth": 0,
            "suppressMenu": True,
            "suppressSizeToFit": True,
            "suppressColumnsToolPanel": True
            }
    ]

    unique_iso_year_weeks = sorted(set(
        ['_'.join(col.split('_')[1:3]) for col in df_tab4.columns if col.startswith('Week_')]
    ))
    iso_year_week_to_week_number = {
        iso_year_week: int(iso_year_week.split('_')[1].lstrip('W')) for iso_year_week in unique_iso_year_weeks
    }

    for iso_year_week in unique_iso_year_weeks:
        week_columns = []
        for Métrica in ['SF', 'HCM', 'Delta']:
            field_name = f"Week_{iso_year_week}_{Métrica}"
            week_columns.append({
                "field": field_name,
                "headerName": Métrica,
                "valueFormatter": "x != null ? x.toFixed(1) : ''",
                "resizable": True,
                "flex": 1,
                "aggFunc": "sum",
                "cellStyle": js_code
            })
        week_number = iso_year_week_to_week_number[iso_year_week]
        header_name = f"Week {week_number}"
        columnDefs.append({
            "headerName": header_name,
            "children": week_columns
        })

    week_number = iso_year_week_to_week_number[iso_year_week]
    header_name = f"Week {week_number}"
    # Calculate totals for numeric columns
    total_row_tab4 = {'Resource_Name': 'Total'}
    for col in numeric_columns_in_pivot_tab4:
        if col in df_tab4.columns:
            total_row_tab4[col] = f"{int(df_tab4[col].sum().round(0)):,}"

    # Convert total_row to DataFrame
    total_df_tab4 = pd.DataFrame(total_row_tab4, index=[0])
    df_tab4_with_totals = pd.concat([total_df_tab4, df_tab4], ignore_index=True)
    df_tab4_with_totals = df_tab4_with_totals[df_tab4_with_totals['Resource_Name'] != 'Total']
    pinned_top_row_tab4 = [total_row_tab4]

    # Use df_tab4_with_totals as the grid data
    data_to_display = df_tab4_with_totals

    # Configure GridOptionsBuilder
    gb_tab4 = GridOptionsBuilder.from_dataframe(data_to_display)
    gb_tab4.configure_default_column(
        groupable=True,
        value=True,
        enableRowGroup=True,
        aggFunc='sum',
        editable=False
    )

    # Configure grid options
    gb_tab4.configure_grid_options(
        domLayout='normal',
        autoSizeColumns='allColumns',
        pinnedTopRowData=pinned_top_row_tab4,
        enableRangeSelection=True,
        groupIncludeFooter=False,
        groupIncludeTotalFooter=False,
        groupDefaultExpanded=0,
        suppressAggFuncInHeader=True,
        suppressRowClickSelection=True,
        groupRowAggNodes=JsCode("""
            function(nodes) {
                var result = {};
                nodes.forEach(function(node) {
                    var data = node.data;
                    for (var key in data) {
                        if (typeof data[key] === 'number') {
                            if (!result[key]) {
                                result[key] = 0;
                            }
                            result[key] += data[key];
                        }
                    }
                });
                return result;
            }
        """),
        autoGroupColumnDef={
            "headerName": "Name / Shop",
            "cellRendererParams": {
                "suppressCount": True,
                "innerRenderer": JsCode("""
                    function(params) {
                        if (params.node.rowPinned === 'top') {
                            return params.data.Resource_Name;
                        } else if (params.node.group) {
                            return params.node.key;
                        } else if (params.data && params.data.Shop_Name) {
                            return params.data.Shop_Name;
                        }
                        return null;
                    }
                """)
            },
            "cellStyle": {'fontWeight': 'bold'},
            "flex": 2,
            "minWidth": 150
        },
        suppressRowTransform=JsCode("""
            function(params) {
                // Suppress row rendering if 'Personal_Number' meets specific condition
                return params.data && params.data.Personal_Number; // Adjust condition as needed
            }
        """)
    )

    grid_options_tab4 = gb_tab4.build()
    grid_options_tab4['columnDefs'] = columnDefs

    # Render the grid
    AgGrid(
        data_to_display.reset_index(drop=True).copy(),
        gridOptions=grid_options_tab4,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        height=1000,
        width='100%',
        theme='streamlit',
        custom_css=custom_css
    )

with tab12:
    month_start_date = date.today().replace(day=1)
    formatted_month_start_date = month_start_date.strftime("%B %#d").lstrip('0')

    comparison_options = [
        "Actual Data",  # Default
        f"Comparison with month start ({formatted_month_start_date})",
        "Comparison with yesterday"
    ]
    selected_mode = st.selectbox("Select comparison type:", comparison_options)

    reference_df = None
    comparison_label = ""
    if "month start" in selected_mode:
        reference_df = aggregated_data_tab11_mf
        comparison_label = f"vs. {formatted_month_start_date}"
    elif "yesterday" in selected_mode:
        reference_df = aggregated_data_tab11_yesterday
        comparison_label = "vs. Yesterday"
    def pivot_saturation_data(input_df, group_col):
        # Choose the proper ratio and upcoming fields based on the grouping column
        if group_col == 'Shop':
            ratio_field = 'ratioadelanto'
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'
        elif group_col == 'Area':
            ratio_field = 'ratioadelanto_area'
            upcoming_field = 'Upcoming_3_Weeks_Sum_area'
            current_field = 'Current_Week_Sum_area'
        elif group_col == 'Region':
            ratio_field = 'ratioadelanto_reg'
            upcoming_field = 'Upcoming_3_Weeks_Sum_reg'
            current_field = 'Current_Week_Sum_reg'
        else:
            ratio_field = 'ratioadelanto'  # default
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'

        pivoted = input_df.pivot_table(
            index=[group_col],
            columns=['iso_week'],
            values=[
                'OpenHours', 'TotalHours', 'FirstVisit', 'PreSales', 'AfterSales',
                'PersonalBlockHours', 'BusinessBlockHours', 'AdminBlockHours',
                'WeBlockHours', 'OtherBlockHours', ratio_field, current_field, upcoming_field
            ],
            aggfunc={
                'OpenHours': 'sum',
                'TotalHours': 'sum',
                'FirstVisit': 'sum',
                'PreSales': 'sum',
                'AfterSales': 'sum',
                'PersonalBlockHours': 'sum',
                'BusinessBlockHours': 'sum',
                'AdminBlockHours': 'sum',
                'WeBlockHours': 'sum',
                'OtherBlockHours': 'sum',
                ratio_field: 'first',  # always use the first value
                current_field: 'first',
                upcoming_field: 'first'
            },
            fill_value=0
        )
        pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns.to_flat_index()]
        pivoted.reset_index(inplace=True)

        # Calculate derived fields for each TotalHours column
        for col in pivoted.columns:
            if 'TotalHours' in col:
                # FTE
                pivoted[col.replace('TotalHours', 'FTE')] = pivoted[col] / 40

                # Saturation %
                open_col = col.replace('TotalHours', 'OpenHours')
                if open_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'SaturationPercentage')] = (
                        100 - (pivoted[open_col] * 100 / pivoted[col])
                    ).fillna(0)

                # First Visit %
                first_col = col.replace('TotalHours', 'FirstVisit')
                if first_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'FirstVisitPercentage')] = (
                        (pivoted[first_col] * 5 / 60) / pivoted[col]
                    ).fillna(0)

                # Pre-Sales %
                pre_col = col.replace('TotalHours', 'PreSales')
                if pre_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'PreSalesPercentage')] = (
                        (pivoted[pre_col] * 5 / 60) / pivoted[col]
                    ).fillna(0)

                # After-Sales %
                after_col = col.replace('TotalHours', 'AfterSales')
                if after_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'AfterSalesPercentage')] = (
                        (pivoted[after_col] * 5 / 60) / pivoted[col]
                    ).fillna(0)

                # Personal Block %
                per_blocked_col = col.replace('TotalHours', 'PersonalBlockHours')
                if per_blocked_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'PersonalBlocksPercentage')] = (
                        pivoted[per_blocked_col] / pivoted[col]
                    ).fillna(0)

                # Business Block %
                bus_blocked_col = col.replace('TotalHours', 'BusinessBlockHours')
                if bus_blocked_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'BusinessBlocksPercentage')] = (
                        pivoted[bus_blocked_col] / pivoted[col]
                    ).fillna(0)
                
                # Admin Block %
                admin_blocked_col = col.replace('TotalHours', 'AdminBlockHours')
                if admin_blocked_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'AdminBlocksPercentage')] = (
                        pivoted[admin_blocked_col] / pivoted[col]
                    ).fillna(0)

                # WE Block %
                we_blocked_col = col.replace('TotalHours', 'WeBlockHours')
                if we_blocked_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'WeBlocksPercentage')] = (
                        pivoted[we_blocked_col] / pivoted[col]
                    ).fillna(0)
                        
                # Other Block %
                other_blocked_col = col.replace('TotalHours', 'OtherBlockHours')
                if other_blocked_col in pivoted.columns:
                    pivoted[col.replace('TotalHours', 'OtherBlocksPercentage')] = (
                        pivoted[other_blocked_col] / pivoted[col]
                    ).fillna(0)

        # Now round all columns with the dynamic ratio_field (placed outside the loop)
        for col in pivoted.columns:
            if ratio_field in col:
                pivoted[col] = pivoted[col].round(2)

        pivoted.columns = [c.replace(" ", "_") for c in pivoted.columns]
        return pivoted


    def calculate_percentage_change(pivot_cur, pivot_ref, group_col):
        """
        Returns a new DataFrame pivot_diff with the same columns as pivot_cur
        but each numeric value is replaced by ((cur - ref)/ref)*100.
        """
        pivot_diff = pivot_cur.copy()
        measure_cols = [c for c in pivot_cur.columns if c != group_col]

        # Ensure pivot_ref has those columns (fill missing with 0)
        for col in measure_cols:
            if col not in pivot_ref.columns:
                pivot_ref[col] = 0

        for col in measure_cols:
            ref_vals = pivot_ref[col]
            cur_vals = pivot_cur[col]
            with pd.option_context('mode.use_inf_as_na', True):
                diff = (cur_vals - ref_vals)
                # Adjust condition to capture any ratio fields dynamically:
                if col.startswith("FTE_"):
                    diff = (((cur_vals - ref_vals)/ref_vals) * 100).replace([np.inf, -np.inf], 0).fillna(0)
                pivot_diff[col] = diff

        return pivot_diff


    def add_pinned_total_row(df, group_col):
        """
        Recalculates grand totals for FTE, Saturation%, etc. Returns:
        - final_df (without a total row)
        - pinned_data (the total row as pinned)
        """
        # Define dynamic ratio field based on the group
        if group_col == 'Shop':
            ratio_field = 'ratioadelanto'
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'
        elif group_col == 'Area':
            ratio_field = 'ratioadelanto_area'
            upcoming_field = 'Upcoming_3_Weeks_Sum_area'
            current_field = 'Current_Week_Sum_area'
        elif group_col == 'Region':
            ratio_field = 'ratioadelanto_reg'
            upcoming_field = 'Upcoming_3_Weeks_Sum_reg'
            current_field = 'Current_Week_Sum_reg'
        else:
            ratio_field = 'ratioadelanto'  # default
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'


        df_copy = df.copy().reset_index(drop=True)
        total_row = {group_col: "Total"}

        for col in df_copy.columns:
            if "TotalHours_" in col:
                week_num = col.split('_')[1]
                total_hours_sum = df_copy[col].sum()

                open_col    = f"OpenHours_{week_num}"
                first_col   = f"FirstVisit_{week_num}"
                pre_col     = f"PreSales_{week_num}"
                after_col   = f"AfterSales_{week_num}"
                per_blocked_col = f"PersonalBlockHours_{week_num}"
                bus_blocked_col = f"BusinessBlockHours_{week_num}"
                admin_blocked_col = f"AdminBlockHours_{week_num}"
                we_blocked_col = f"WeBlockHours_{week_num}"
                upcoming_col = f"{upcoming_field}_{week_num}" 
                current_week_col = f"{current_field}_{week_num}" 
                fte_col     = col.replace("TotalHours", "FTE")

                if fte_col in df_copy.columns:
                    total_row[fte_col] = df_copy[fte_col].sum().round(1)

                    # Recompute saturation from grand totals
                    if open_col in df_copy.columns and total_hours_sum != 0:
                        open_sum = df_copy[open_col].sum()
                        sat_col = col.replace("TotalHours","SaturationPercentage")
                        total_row[sat_col] = 100 - (open_sum * 100 / total_hours_sum)

                    # First Visit %
                    if first_col in df_copy.columns and total_hours_sum != 0:
                        first_sum = df_copy[first_col].sum()
                        firstVisit_col = col.replace("TotalHours","FirstVisitPercentage")
                        total_row[firstVisit_col] = (first_sum * 5 / 60) / total_hours_sum

                    # Pre-Sales %
                    if pre_col in df_copy.columns and total_hours_sum != 0:
                        pre_sum = df_copy[pre_col].sum()
                        preSales_col = col.replace("TotalHours","PreSalesPercentage")
                        total_row[preSales_col] = (pre_sum * 5 / 60) / total_hours_sum

                    # After-Sales %
                    if after_col in df_copy.columns and total_hours_sum != 0:
                        after_sum = df_copy[after_col].sum()
                        afterSales_col = col.replace("TotalHours","AfterSalesPercentage")
                        total_row[afterSales_col] = (after_sum * 5 / 60) / total_hours_sum

                    # Personal & Business Blocks
                    if per_blocked_col in df_copy.columns and total_hours_sum != 0:
                        per_blocked_sum = df_copy[per_blocked_col].sum()
                        pers_col = col.replace("TotalHours","PersonalBlocksPercentage")
                        total_row[pers_col] = per_blocked_sum / total_hours_sum

                    if bus_blocked_col in df_copy.columns and total_hours_sum != 0:
                        bus_blocked_sum = df_copy[bus_blocked_col].sum()
                        biz_col  = col.replace("TotalHours","BusinessBlocksPercentage")
                        total_row[biz_col]  = bus_blocked_sum / total_hours_sum

                    if admin_blocked_col in df_copy.columns and total_hours_sum != 0:
                        admin_blocked_sum = df_copy[admin_blocked_col].sum()
                        admin_col  = col.replace("TotalHours","AdminBlocksPercentage")
                        total_row[admin_col]  = admin_blocked_sum / total_hours_sum

                    if we_blocked_col in df_copy.columns and total_hours_sum != 0:
                        we_blocked_sum = df_copy[we_blocked_col].sum()
                        we_col  = col.replace("TotalHours","WeBlocksPercentage")
                        total_row[we_col]  = we_blocked_sum / total_hours_sum

                    # Use dynamic ratio field here
                    ratio_col = f"{ratio_field}_{week_num}"
                    if current_week_col in df_copy.columns and upcoming_col in df_copy.columns:
                        current_week_sum = df_copy[current_week_col].sum()
                        upcoming_3_weeks_sum = df_copy[upcoming_col].sum()
                        if upcoming_3_weeks_sum != 0:
                            total_row[ratio_col] = current_week_sum / upcoming_3_weeks_sum
                        else:
                            total_row[ratio_col] = 0
        pinned_top_row = [total_row]
        return df_copy, pinned_top_row


    def add_pinned_total_row_comparison(pivot_diff, pivot_cur, pivot_ref, group_col):
        # Define dynamic ratio field
        if group_col == 'Shop':
            ratio_field = 'ratioadelanto'
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'
        elif group_col == 'Area':
            ratio_field = 'ratioadelanto_area'
            upcoming_field = 'Upcoming_3_Weeks_Sum_area'
            current_field = 'Current_Week_Sum_area'
        elif group_col == 'Region':
            ratio_field = 'ratioadelanto_reg'
            upcoming_field = 'Upcoming_3_Weeks_Sum_reg'
            current_field = 'Current_Week_Sum_reg'
        else:
            ratio_field = 'ratioadelanto'  # default
            upcoming_field = 'Upcoming_3_Weeks_Sum'
            current_field = 'Current_Week_Sum'
        total_dict = {group_col: "Total"}
        pivot_diff_no_total = pivot_diff.copy()

        for col in pivot_cur.columns:
            if 'TotalHours_' in col:
                week_num = col.split('_')[1]

                # Sums of raw hours
                total_hours_cur = pivot_cur[col].sum()
                total_hours_ref = pivot_ref[col].sum() if col in pivot_ref.columns else 0

                # FTE
                fte_col = col.replace("TotalHours", "FTE")
                fte_cur = pivot_cur[fte_col].sum() if fte_col in pivot_cur.columns else 0
                fte_ref = pivot_ref[fte_col].sum() if fte_col in pivot_ref.columns else 0
                total_dict[fte_col] = ((fte_cur - fte_ref)/fte_ref)*100 if fte_ref != 0 else 0

                # Saturation
                open_col = col.replace("TotalHours", "OpenHours")
                open_cur = pivot_cur[open_col].sum() if open_col in pivot_cur.columns else 0
                open_ref = pivot_ref[open_col].sum() if open_col in pivot_ref.columns else 0
                sat_col = col.replace("TotalHours", "SaturationPercentage")
                sat_cur = 100 - (open_cur * 100 / total_hours_cur) if total_hours_cur != 0 else 0
                sat_ref = 100 - (open_ref * 100 / total_hours_ref) if total_hours_ref != 0 else 0
                total_dict[sat_col] = ((sat_cur - sat_ref))  if sat_ref != 0 else 0

                # First Visit
                first_col = col.replace("TotalHours", "FirstVisit")
                first_cur = pivot_cur[first_col].sum() if first_col in pivot_cur.columns else 0
                first_ref = pivot_ref[first_col].sum() if first_col in pivot_ref.columns else 0
                first_pct_col = col.replace("TotalHours", "FirstVisitPercentage")
                first_pct_cur = (first_cur * 5/60)/total_hours_cur if total_hours_cur != 0 else 0
                first_pct_ref = (first_ref * 5/60)/total_hours_ref if total_hours_ref != 0 else 0
                total_dict[first_pct_col] = ((first_pct_cur - first_pct_ref) if first_pct_ref != 0 else 0)

                # Pre-Sales
                pre_col = col.replace("TotalHours", "PreSales")
                pre_cur = pivot_cur[pre_col].sum() if pre_col in pivot_cur.columns else 0
                pre_ref = pivot_ref[pre_col].sum() if pre_col in pivot_ref.columns else 0
                pre_pct_col = col.replace("TotalHours", "PreSalesPercentage")
                pre_pct_cur = (pre_cur * 5/60)/total_hours_cur if total_hours_cur != 0 else 0
                pre_pct_ref = (pre_ref * 5/60)/total_hours_ref if total_hours_ref != 0 else 0
                total_dict[pre_pct_col] = ((pre_pct_cur - pre_pct_ref) if pre_pct_ref != 0 else 0)

                # After-Sales
                after_col = col.replace("TotalHours", "AfterSales")
                after_cur = pivot_cur[after_col].sum() if after_col in pivot_cur.columns else 0
                after_ref = pivot_ref[after_col].sum() if after_col in pivot_ref.columns else 0
                after_pct_col = col.replace("TotalHours", "AfterSalesPercentage")
                after_pct_cur = (after_cur * 5/60)/total_hours_cur if total_hours_cur != 0 else 0
                after_pct_ref = (after_ref * 5/60)/total_hours_ref if total_hours_ref != 0 else 0
                total_dict[after_pct_col] = ((after_pct_cur - after_pct_ref) if after_pct_ref != 0 else 0)

                # Personal & Business Blocks
                per_blocked_col = col.replace("TotalHours","PersonalBlockHours")
                per_blocked_cur = pivot_cur[per_blocked_col].sum() if per_blocked_col in pivot_cur.columns else 0
                per_blocked_ref = pivot_ref[per_blocked_col].sum() if per_blocked_col in pivot_ref.columns else 0
                pers_col = col.replace("TotalHours","PersonalBlocksPercentage")
                pers_cur = per_blocked_cur / total_hours_cur if total_hours_cur != 0 else 0
                pers_ref = per_blocked_ref / total_hours_ref if total_hours_ref != 0 else 0
                total_dict[pers_col] = (pers_cur - pers_ref) 

                bus_blocked_col = col.replace("TotalHours","BusinessBlockHours")
                bus_blocked_cur = pivot_cur[bus_blocked_col].sum() if bus_blocked_col in pivot_cur.columns else 0
                bus_blocked_ref = pivot_ref[bus_blocked_col].sum() if bus_blocked_col in pivot_ref.columns else 0
                biz_col  = col.replace("TotalHours","BusinessBlocksPercentage")
                bus_cur = bus_blocked_cur / total_hours_cur if total_hours_cur != 0 else 0
                bus_ref = bus_blocked_ref / total_hours_ref if total_hours_ref != 0 else 0
                total_dict[biz_col]  = (bus_cur - bus_ref) 

                admin_blocked_col = col.replace("TotalHours","AdminBlockHours") 
                admin_blocked_cur = pivot_cur[admin_blocked_col].sum() if admin_blocked_col in pivot_cur.columns else 0              
                admin_blocked_ref = pivot_ref[admin_blocked_col].sum() if admin_blocked_col in pivot_ref.columns else 0 
                admin_col  = col.replace("TotalHours","AdminBlocksPercentage")
                admin_cur = admin_blocked_cur / total_hours_cur if total_hours_cur != 0 else 0
                admin_ref = admin_blocked_ref / total_hours_ref if total_hours_ref != 0 else 0  
                total_dict[admin_col]  = (admin_cur - admin_ref) 

                we_blocked_col = col.replace("TotalHours","WeBlockHours")
                we_blocked_cur = pivot_cur[we_blocked_col].sum() if we_blocked_col in pivot_cur.columns else 0
                we_blocked_ref = pivot_ref[we_blocked_col].sum() if we_blocked_col in pivot_ref.columns else 0
                total_dict[we_blocked_col]  = (we_blocked_cur / total_hours_cur if total_hours_cur != 0 else 0) - (we_blocked_ref / total_hours_ref if total_hours_ref != 0 else 0)
                we_col  = col.replace("TotalHours","WeBlocksPercentage")
                we_cur = we_blocked_cur / total_hours_cur if total_hours_cur != 0 else 0
                we_ref = we_blocked_ref / total_hours_ref if total_hours_ref != 0 else 0  
                total_dict[we_col]  = (we_cur - we_ref) 
                
                # Use dynamic ratio field here as well
                ratio_col = col.replace("TotalHours", ratio_field)
                current_week_col = f"{current_field}_{week_num}"
                upcoming_week_col = f"{upcoming_field}_{week_num}"

                current_week_sum_cur = pivot_cur[current_week_col].sum() if current_week_col in pivot_cur.columns else 0
                upcoming_3_weeks_sum_cur = pivot_cur[upcoming_week_col].sum() if upcoming_week_col in pivot_cur.columns else 1
                current_week_sum_ref = pivot_ref[current_week_col].sum() if current_week_col in pivot_ref.columns else 0
                upcoming_3_weeks_sum_ref = pivot_ref[upcoming_week_col].sum() if upcoming_week_col in pivot_ref.columns else 1

                ratio_cur = (current_week_sum_cur / upcoming_3_weeks_sum_cur) if upcoming_3_weeks_sum_cur != 0 else 0
                ratio_ref = (current_week_sum_ref / upcoming_3_weeks_sum_ref) if upcoming_3_weeks_sum_ref != 0 else 0

                ratio_col = f"{ratio_field}_{week_num}"
                total_dict[ratio_col] = ratio_cur - ratio_ref
        pinned_top_row_data = [total_dict]
        return pivot_diff_no_total, pinned_top_row_data


    def render_aggrid_table(df, group_col, pinned_data, is_comparison=False):
        # Determine the dynamic ratio field for the current group
        if group_col == 'Shop':
            dynamic_ratio = 'ratioadelanto'
        elif group_col == 'Area':
            dynamic_ratio = 'ratioadelanto_area'
        elif group_col == 'Region':
            dynamic_ratio = 'ratioadelanto_reg'
        else:
            dynamic_ratio = 'ratioadelanto'

        js_code = JsCode("""
            function(params) {
                // Pinned rows style
                if (params.node.rowPinned) {
                    return {'font-weight': 'bold', 'backgroundColor': '#e0e0e0'};
                }
                var field = params.colDef.field;
                var value = parseFloat(params.value) || 0;
                // After-Sales %: red if > 15%
                if (field.indexOf("AfterSalesPercentage") !== -1) {
                    if (value > 0.15) {
                        return {'backgroundColor': '#cc0641', 'color': 'white'};
                    }
                }
                // Personal/Business Blocks %
                // All Blocks %
                if (
                    field.indexOf("PersonalBlocksPercentage") !== -1 ||
                    field.indexOf("BusinessBlocksPercentage") !== -1 ||
                    field.indexOf("AdminBlocksPercentage") !== -1 ||
                    field.indexOf("WeBlocksPercentage") !== -1
                ) {
                    var parts = field.split('_');
                    var weekNum = parts[parts.length - 1];
                    var personalField = "PersonalBlocksPercentage_" + weekNum;
                    var businessField = "BusinessBlocksPercentage_" + weekNum;
                    var adminField = "AdminBlocksPercentage_" + weekNum;
                    var weField = "WeBlocksPercentage_" + weekNum;
                    var personalVal = parseFloat(params.data[personalField]) || 0;
                    var businessVal = parseFloat(params.data[businessField]) || 0;
                    var adminVal = parseFloat(params.data[adminField]) || 0;
                    var weVal = parseFloat(params.data[weField]) || 0;
                    if ((personalVal + businessVal + adminVal + weVal) > 0.30) {
                        return {'backgroundColor': '#cc0641', 'color': 'white'};
                    }
                }
                // First Visit %
                if (field.indexOf("FirstVisitPercentage") !== -1) {
                    if (value > 0.40) {
                        return {'backgroundColor': '#95cd41'};
                    }
                }
                // Ratio Adelanto styling
                if (field.indexOf(dynamic_ratio) !== -1) {
                    if (value > 2) {
                        return {'backgroundColor': '#95cd41', 'color': 'white'};
                    } else if (value < 0.5) {
                        return {'backgroundColor': '#cc0641', 'color': 'white'};
                    }
                }
                // FTE styling
                if (field.indexOf("FTE") !== -1) {
                    if (value === 0) {
                        return {'backgroundColor': '#cc0641', 'color': 'white'};
                    }
                }
                return null;
            }
        """.replace("dynamic_ratio", f'"{dynamic_ratio}"'))

        custom_css = {
            ".centered-week-header": {"text-align": "center !important"},
            ".ag-header-cell": {
                "background-color": "#cc0641 !important",
                "color": "white !important",
                "font-weight": "bold",
                "font-size": "11px !important",
                "padding": "4px"
            },
            ".ag-header-group-cell": {
                "background-color": "#cc0641 !important",
                "color": "white !important",
                "font-weight": "bold"
            },
            ".ag-cell": {
                "padding": "2px",
                "font-size": "12px"
            },
            ".ag-header": {"height": "35px"},
            ".ag-theme-streamlit .ag-row": {"max-height": "30px"},
            ".ag-theme-streamlit .ag-menu-option-text, .ag-theme-streamlit .ag-filter-body-wrapper, .ag-theme-streamlit .ag-input-wrapper, .ag-theme-streamlit .ag-icon": {
                "font-size": "6px !important"
            },
            ".ag-theme-streamlit .ag-root-wrapper": {
                "border": "2px solid #cc0641",
                "border-radius": "5px"
            }
        }

        # Build column definitions
        columnDefs = [{
            "headerName": group_col,
            "field": group_col,
            "resizable": True,
            "flex": 2,
            "minWidth": 150,
            "filter": 'agTextColumnFilter',
        }]

        # For every "TotalHours_X" column, build a "Week X" group
        for c in df.columns:
            if 'TotalHours_' in c:
                week_num = c.split('_')[1]
                headerName = f"Week {week_num}"
                fte_field = c.replace('TotalHours', 'FTE')
                fte_value_formatter = "x.toFixed(1) + '%'" if is_comparison else "x.toFixed(1)"
                # Use the dynamic_ratio field for the Ratio Adelanto column
                ratio_field_name = c.replace('TotalHours', dynamic_ratio)

                columnDefs.append({
                    "headerName": headerName,
                    "headerClass": "centered-week-header",
                    "children": [
                        {
                            "field": fte_field,
                            "headerName": "FTE",
                            "valueFormatter": fte_value_formatter,
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'SaturationPercentage'),
                            "headerName": "Saturation %",
                            "valueFormatter": "(x).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'FirstVisitPercentage'),
                            "headerName": "First Visit %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'PreSalesPercentage'),
                            "headerName": "Pre-Sales %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'AfterSalesPercentage'),
                            "headerName": "After-Sales %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'PersonalBlocksPercentage'),
                            "headerName": "Personal B. %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'BusinessBlocksPercentage'),
                            "headerName": "Business B. %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'AdminBlocksPercentage'),
                            "headerName": "Admin B. %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": c.replace('TotalHours', 'WeBlocksPercentage'),
                            "headerName": "WE B. %",
                            "valueFormatter": "(x * 100).toFixed(1) + '%'",
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        },
                        {
                            "field": ratio_field_name,
                            "headerName": "Ratio Adelanto",
                            "valueFormatter": fte_value_formatter,
                            "resizable": True,
                            "flex": 1,
                            "cellStyle": js_code
                        }
                    ]
                })

        gb = GridOptionsBuilder.from_dataframe(df)
        for col in df.columns:
            if col != group_col:
                gb.configure_column(col, cellStyle=js_code)

        gb.configure_grid_options(
            pinnedTopRowData=pinned_data,
            domLayout='normal',
            autoSizeColumns='allColumns',
            enableFillHandle=True
        )
        grid_options = gb.build()
        grid_options['columnDefs'] = columnDefs

        AgGrid(
            df,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            height=min(max(len(df) * 52, 180), 1000),
            width='100%',
            theme='streamlit',
            custom_css=custom_css
        )


    def show_one_table(group_col, selected_mode):
        is_comparison = (selected_mode != "Actual Data")
        if not is_comparison:
            # For "Actual Data"
            pivot_cur = pivot_saturation_data(aggregated_data_tab11, group_col)
            final_df, pinned_data = add_pinned_total_row(pivot_cur, group_col)
            render_aggrid_table(final_df, group_col, pinned_data, is_comparison=False)
            excel_data = convert_saturation_to_excel(final_df, group_col, is_comparison=False)
            st.download_button(
                label=f"📥 Download data for table: {group_col}",
                data=excel_data,
                file_name=f"{group_col}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            # For comparison modes
            pivot_cur = pivot_saturation_data(aggregated_data_tab11, group_col)
            pivot_ref = pivot_saturation_data(reference_df, group_col)
            pivot_diff = calculate_percentage_change(pivot_cur, pivot_ref, group_col)
            final_diff_df, pinned_top_row = add_pinned_total_row_comparison(pivot_diff, pivot_cur, pivot_ref, group_col)
            render_aggrid_table(final_diff_df, group_col, pinned_top_row, is_comparison=True)
            excel_data = convert_saturation_to_excel(final_diff_df, group_col, is_comparison=True)
            st.download_button(
                label=f"📥 Download data for table: {group_col}",
                data=excel_data,
                file_name=f"{group_col}_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # Finally, show tables for Region, Area, and Shop
    show_one_table('Region', selected_mode)
    show_one_table('Area', selected_mode)
    show_one_table('Shop', selected_mode)




with tab6:    
    # ====== PERFORMANCE RADAR: KPI tiles + Top/Bottom leaderboards ======

    st.markdown("### Performance Radar")
    # ==== KPI TILES (Global averages) — Baseline: Yesterday ====
    # Build Shop-level pivots
    pivot_cur = pivot_saturation_data(aggregated_data_tab11, 'Shop')
    pivot_ref = pivot_saturation_data(aggregated_data_tab11_yesterday, 'Shop')

    # Find latest common ISO week present in both pivots
    def week_list(df):
        return sorted({int(c.split('_')[-1]) for c in df.columns if c.startswith('SaturationPercentage_')})
    weeks_cur = set(week_list(pivot_cur))
    weeks_ref = set(week_list(pivot_ref))
    weeks_common = sorted(list(weeks_cur & weeks_ref))
    if not weeks_common:
        st.info("No common ISO week found between today and yesterday.")
    else:
        wk = weeks_common[-1]

        # Helper to compute GLOBAL (weighted) % from totals
        def compute_global(df, wk):
            get = lambda name: df.get(f"{name}_{wk}", pd.Series(dtype='float'))
            tot_hours   = get("TotalHours").sum()
            open_hours  = get("OpenHours").sum()
            first_vis   = get("FirstVisit").sum()          # 5-min slots
            after_sales = get("AfterSales").sum()          # 5-min slots
            blocks_hrs  = (
                get("PersonalBlockHours").sum() +
                get("BusinessBlockHours").sum() +
                get("AdminBlockHours").sum() +
                get("WeBlockHours").sum()
            )
            # Safeguard
            if tot_hours == 0:
                return dict(sat=0.0, blocks=0.0, first=0.0, after=0.0)
            sat_pct    = 100 - (open_hours * 100.0 / tot_hours)
            blocks_pct = (blocks_hrs / tot_hours) * 100.0
            first_pct  = ((first_vis * 5.0 / 60.0) / tot_hours) * 100.0
            after_pct  = ((after_sales * 5.0 / 60.0) / tot_hours) * 100.0
            return dict(sat=sat_pct, blocks=blocks_pct, first=first_pct, after=after_pct)

        cur = compute_global(pivot_cur, wk)
        ref = compute_global(pivot_ref, wk)

        def kpi(col, label, cur_val, ref_val, better_is_down=True):
            delta = cur_val - ref_val
            # For st.metric, positive/negative arrows are automatic; we add context in the text.
            st.metric(
                label=label,
                value=f"{cur_val:.1f}%",
                delta=f"{delta:+.1f} pts vs Yesterday"
            )
            # Thin color bar for quick visual
            improve = (delta < 0) if better_is_down else (delta > 0)
            color = "#95cd41" if improve else ("#cc0641" if delta != 0 else "#999999")
            st.markdown(
                f"<div style='height:3px;background:{color};border-radius:2px;margin-top:-8px;margin-bottom:8px'></div>",
                unsafe_allow_html=True
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi(c1, "Saturation %",  cur['sat'],   ref['sat'],   better_is_down=True)
        with c2: kpi(c2, "Blocks %",      cur['blocks'],ref['blocks'],better_is_down=True)
        with c3: kpi(c3, "First-Visit %", cur['first'], ref['first'], better_is_down=False)
        with c4: kpi(c4, "After-Sales %", cur['after'], ref['after'], better_is_down=True)
    # ===== Tasks: summary (place this right before the "SF vs HCM Comparison" subheader) =====
        st.subheader("Tasks: summary", divider="gray")
        
        KEY = "Clave compuesta"
        tasks_repo_path = "output/Tasks.csv"
        
        def _load_tasks_only():
            raw = _download_github_file(REPO_OWNER, REPO_NAME, tasks_repo_path, GITHUB_TOKEN)
            base_cols = [KEY, "Action", "Details", "updated_at"]
            if raw is None:
                return pd.DataFrame(columns=base_cols)
            try:
                df = pd.read_csv(BytesIO(raw))
            except Exception:
                df = pd.read_csv(BytesIO(raw), sep=";")
            # normalize
            for c in base_cols:
                if c not in df.columns:
                    df[c] = "" if c != "updated_at" else pd.NaT
            return df[base_cols]
        
        tasks_df = _load_tasks_only()
        
        if tasks_df.empty:
            st.info("No managed tasks found yet.")
        else:
            # Enrich tasks with Region + iso_week using filtered_hcm (same keys, current filters)
            if KEY in filtered_hcm.columns:
                enrich_cols = [KEY, "Region", "iso_week"]
                tasks_enriched = tasks_df.merge(
                    hcm[enrich_cols].drop_duplicates(subset=[KEY]),
                    on=KEY, how="left"
                )
            else:
                tasks_enriched = tasks_df.copy()
                tasks_enriched["Region"] = np.nan
                tasks_enriched["iso_week"] = np.nan
        
            # Focus on the selected ISO week from sidebar
            tasks_week = tasks_enriched[tasks_enriched["iso_week"] == int(iso_week_filter)].copy()
        
            if tasks_week.empty:
                st.info(f"No task for ISO week {iso_week_filter}.")
            else:
                tasks_week["Region"] = tasks_week["Region"].fillna("Unknown")
        
                # --- Bar chart: Total Tasks vs Solved per Region (from Tasks.csv only) ---
                by_region = (
                    tasks_week.groupby("Region", dropna=False).size().reset_index(name="Total Tasks")
                )
                solved_counts = (
                    tasks_week.assign(is_solved=tasks_week["Action"].eq("Solved"))
                              .groupby("Region")["is_solved"].sum()
                              .reset_index(name="Solved")
                )
                by_region = by_region.merge(solved_counts, on="Region", how="left").fillna(0)
        
                fig_bar = px.bar(
                    by_region.melt(id_vars="Region", var_name="Type", value_name="Count"),
                    x="Region",
                    y="Count",
                    color="Type",
                    barmode="group",
                    text="Count",
                    color_discrete_map={
                        "No action possible": "#5570ff",
                        "Support Ticket": "#f1b84b",
                        "HR Ticket": "#cc0641",
                        "Solved": "#b5a642",
                        "IT Ticket": "#f86b52",
                        "Total Tasks": "#5570ff"  # keep visible if legend mixes in
                    },
                    title=f"Tasks vs Solved per Region (ISO week {iso_week_filter})"
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(yaxis_title="Count", xaxis_title="Region")
        
                # --- Donut chart: Action distribution (from Tasks.csv only) ---
                allowed_actions = ["No action possible", "Support Ticket", "HR Ticket", "IT Ticket", "Solved"]
                dist = (tasks_week[tasks_week["Action"].isin(allowed_actions)]
                        .groupby("Action", dropna=False).size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False))
        
                if dist.empty:
                    fig_pie = px.pie(
                        pd.DataFrame({"Action": ["No data"], "count": [1]}),
                        names="Action", values="count", hole=0.5,
                        title=f"Action Distribution (ISO week {iso_week_filter})"
                    )
                else:
                    fig_pie = px.pie(
                        dist, names="Action", values="count", hole=0.5,
                        title=f"Action Distribution (ISO week {iso_week_filter})",
                        color="Action",
                        color_discrete_map={
                            "No action possible": "#5570ff",
                            "Support Ticket": "#f1b84b",
                            "HR Ticket": "#cc0641",
                            "Solved": "#b5a642",
                            "IT Ticket": "#f86b52"
                        }
                    )
        
                # Layout
                c1, c2 = st.columns([1.25, 1])
                with c1:
                    st.plotly_chart(fig_bar, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("SF vs HCM Comparison", divider="gray")


    # Warning messages for empty data
    if weekly_shift_slots.empty:
        st.warning("No shops found for the selected filter criteria.")
    if weekly_shift_slots_yesterday.empty:
        st.warning("No shops found for the selected filter criteria.")
    
    
    hcm_weekly_diff = filtered_hcm.groupby('iso_week').agg(
        total_diff=('Diferencia de hcm duración', 'sum')
    ).reset_index()
    
    # Create HCM vs SF line chart
    fig_diff = px.line(
        hcm_weekly_diff,
        x='iso_week', 
        y='total_diff',
        labels={'iso_week': 'ISO Week', 'total_diff': 'Total Difference (HCM vs SF)'},
        title="Difference in duration of SF vs HCM over the weeks",
        markers=True
    )
    fig_diff.update_layout(
        xaxis_title="ISO Week",
        yaxis_title="Total Difference",
        hovermode="x unified"
    )
    fig_diff.update_traces(
        hovertemplate='ISO Week: %{x}<br>Total Difference: %{y:.2f}'
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    # Display the table in a full row below the chart and top shops
    st.subheader("SF vs HCM Comparison (Weekly)")
    
    # Get all unique weeks in the dataset
    available_weeks = sorted(filtered_hcm["iso_week"].unique())
    
    table_rows = []
    
    for w in available_weeks:  # Iterate over all unique weeks
        week_data = filtered_hcm[filtered_hcm["iso_week"] == w]
        sf_total = week_data["Duración SF"].sum()
        hcm_total = week_data["Duración HCM"].sum()
    
        delta = sf_total- hcm_total 
        delta_pct = (delta / sf_total * 100) if sf_total != 0 else 0
    
        shop_group = week_data.groupby("Shop Name", dropna=False).agg({
            "Duración SF": "sum",
            "Duración HCM": "sum"
        }).reset_index()
    
        n_hcm_gt_sf = (shop_group["Duración HCM"] > shop_group["Duración SF"]).sum()
        n_sf_gt_hcm = (shop_group["Duración SF"] > shop_group["Duración HCM"]).sum()
        n_sf_eq_hcm = (shop_group["Duración SF"] == shop_group["Duración HCM"]).sum()
    
        table_rows.append({
            "Week": w,
            "SF": sf_total if pd.notna(sf_total) else 0,  # Default to 0 if missing
            "HCM": hcm_total if pd.notna(hcm_total) else 0,
            "Delta": delta if pd.notna(delta) else 0,
            "Delta %": delta_pct if pd.notna(delta_pct) else 0,
            "HCM>SF": n_hcm_gt_sf,
            "SF>HCM": n_sf_gt_hcm,
            "SF=HCM": n_sf_eq_hcm
        })
    
    
    results_df = pd.DataFrame(table_rows)
    # Round numeric columns to 1 decimal place
    results_df = results_df.round(1)
    results_df["SF"] = results_df["SF"].apply(lambda x: f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    results_df["HCM"] = results_df["HCM"].apply(lambda x: f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    results_df["Delta"] = results_df["Delta"].apply(lambda x: f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    results_df["Delta %"] = results_df["Delta %"].apply(lambda x: f"{x:,.0f}%".replace(",", "X").replace(".", ",").replace("X", "."))
    
    # Build AgGrid options
    gb = GridOptionsBuilder.from_dataframe(results_df)
    
    # Define custom CSS for header cells
    custom_css = {
        ".ag-header-cell": {
            "background-color": "#cc0641 !important",
            "color": "white !important",
            "font-weight": "bold",
            "padding": "4px" 
        }
    }
    
        # Configure columns with fixed widths
    columns_config = [
        {"field": "Week", "width": 100, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "SF", "width": 150, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "HCM", "width": 150, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "Delta", "width": 120, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "Delta %", "width": 120, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "HCM>SF", "width": 100, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "SF>HCM", "width": 100, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"},
        {"field": "SF=HCM", "width": 100, "cellClass": "ag-cell-right-aligned", "headerClass": "custom-header-right-align"}
    ]
    
    for col in columns_config:
        gb.configure_column(
            col["field"],
            width=col["width"],
            cellClass=col["cellClass"],
            headerClass=col["headerClass"]
        )
    
    gb.configure_grid_options(
        domLayout='normal',
        suppressSizeToFit=True,  
        enableFillHandle=True
    )
    
    # Build grid options
    grid_options = gb.build()
    
    # Render the AG-Grid in Streamlit
    AgGrid(
        results_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,  # Ensure columns are fitted on load
        height=min(max(len(results_df) * 1, 230), 400),
        width='100%',
        theme='streamlit',
        custom_css=custom_css
    )
    if not results_df.empty:
        st.download_button(
            label="📥 Download Comparison file",
            data=convert_df_to_excel(results_df),
            file_name="sf_vs_hcm_comparacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tab9:
    st.subheader("HRBP Tasks")
    # ===== Tasks persistence (used in HRBP Tasks tab) =====
    # Where to store the actions CSV in your repo
    tasks_repo_path = "output/Tasks.csv"
    
    @st.cache_data(ttl=0, show_spinner=False)
    def load_persisted_tasks():
        """Load Tasks.csv from GitHub if it exists; otherwise return an empty, normalized DataFrame."""
        raw = _download_github_file(REPO_OWNER, REPO_NAME, tasks_repo_path, GITHUB_TOKEN)
        cols = ["Clave compuesta", "Shop Code", "Resource Name", "Action", "Details", "updated_at"]
        if raw is None:
            return pd.DataFrame(columns=cols)
    
        try:
            df = pd.read_csv(BytesIO(raw))
        except Exception:
            # fallback delimiter if someone saved with ';'
            df = pd.read_csv(BytesIO(raw), sep=";")
    
        # Normalize columns
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c != "updated_at" else pd.NaT
        return df[cols]
    
    persisted_df = load_persisted_tasks()

    # --- Key and required columns
    KEY = "Clave compuesta"
    required_cols = [
        KEY, "Shop Code", "Resource Name",
        "Duración SF", "Duración HCM", "Diferencia de hcm duración",
        "iso_year", "iso_week"
    ]
    missing = [c for c in required_cols if c not in filtered_hcm.columns]
    if missing:
        st.error(f"Missing columns in HCM data: {missing}")
        st.stop()

    # Ensure iso_year_week exists as 'YYYY_Www'
    hcm_for_tasks = filtered_hcm.copy()
    if "iso_year_week" not in hcm_for_tasks.columns:
        # robust two-digit week formatting
        iy = pd.to_numeric(hcm_for_tasks["iso_year"], errors="coerce").astype("Int64")
        iw = pd.to_numeric(hcm_for_tasks["iso_week"], errors="coerce").astype("Int64")
        hcm_for_tasks["iso_year_week"] = (
            iy.astype(str) + "_" + iw.apply(lambda x: f"W{int(x):02d}" if pd.notna(x) else "W00")
        )

    # 1) Filter to the ISO week selected in the sidebar
    # (iso_week_filter is already defined from the sidebar selectbox)
    hcm_for_tasks = hcm_for_tasks[hcm_for_tasks["iso_week"] == int(iso_week_filter)]

    # 2) Keep only rows where the difference is not 0
    hcm_for_tasks = hcm_for_tasks[hcm_for_tasks["Diferencia de hcm duración"] != 0]

    # 3) One live row per unique key (your key already encodes year+week)
    live_df = (
        hcm_for_tasks[[KEY, "Shop Code", "Resource Name",
                       "Duración SF", "Duración HCM", "Diferencia de hcm duración",
                       "iso_year_week"]]
        .drop_duplicates(subset=[KEY], keep="last")
        .reset_index(drop=True)
    )

    # Rename to English for display
    live_df = live_df.rename(columns={
        "Duración SF": "SF Duration",
        "Duración HCM": "HCM Duration",
        "Diferencia de hcm duración": "HCM Duration Difference"
    })

    # Merge with persisted actions by key
    persist_cols = [KEY, "Action", "Details"]
    for c in persist_cols:
        if c not in persisted_df.columns:
            persisted_df[c] = ""  # normalize shape if file is empty/new
    
    table_df = live_df.merge(persisted_df[persist_cols], on=KEY, how="left")
    table_df["Action"]  = table_df["Action"].fillna("")
    table_df["Details"] = table_df["Details"].fillna("")
    table_df = table_df.drop_duplicates(subset=[KEY], keep="first").reset_index(drop=True)
    
    # Final display order (include iso_year_week)
    display_cols = [
        KEY, "iso_year_week", "Shop Code", "Resource Name",
        "SF Duration", "HCM Duration", "HCM Duration Difference",
        "Action", "Details"
    ]
    table_df = table_df[display_cols]


    st.caption(
        f"Showing ISO week **{iso_week_filter}**. Edit Action/Details per row. Click **Save** to persist to Git."
    )

    # --- AgGrid with editable Action dropdown + Details free text (unchanged) ---
    gb = GridOptionsBuilder.from_dataframe(table_df)
    
    # 1) Hide the composite key column in the grid
    gb.configure_column(KEY, hide=True)
    
    # 2) Make "Action" editable as a dropdown; "Details" free text
    gb.configure_default_column(editable=False, resizable=True)
    gb.configure_column(
        "Action",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": ["", "No action possible", "Support Ticket", "HR Ticket", "IT Ticket", "Solved"]}
    )
    gb.configure_column("Details", editable=True)
    
    # 3) Row-level styling: turn the whole row green/white if Action is selected
    get_row_style = JsCode("""
    function(params) {
        if (params.data && params.data.Action && params.data.Action.toString().trim() !== "") {
            return {
                'background-color': '#95cd41',
                'color': 'white',
                'font-weight': 'bold'
            };
        }
        return null;
    }
    """)
    
    gb.configure_grid_options(
        rowSelection="single",
        animateRows=True,
        getRowStyle=get_row_style,   # <- apply the row styling
    )
    
    grid_options = gb.build()
    
    # 4) Header style: red background, white text (match other tabs)
    custom_css_tasks = {
        ".ag-header-cell": {
            "background-color": "#cc0641 !important",
            "color": "white !important",
            "font-weight": "bold",
            "padding": "4px"
        },
        ".ag-header-group-cell": {
            "background-color": "#cc0641 !important",
            "color": "white !important",
            "font-weight": "bold"
        },
        ".ag-theme-streamlit .ag-root-wrapper": {
            "border": "2px solid #cc0641",
            "border-radius": "5px"
        }
    }
    
    grid_resp = AgGrid(
        table_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        theme="streamlit",
        height=min(600, 60 + 28*max(5, len(table_df))),
        fit_columns_on_grid_load=True,
        custom_css=custom_css_tasks,  # <- red header + border
    )
    
    edited_df = grid_resp["data"].copy()
    
    
    export_cols = [
        "Shop Code",
        "Resource Name",
        "SF Duration",
        "HCM Duration",
        "HCM Duration Difference",
        "Action",
        "Details",
    ]

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("💾 Save changes", type="primary", use_container_width=True):
            try:
                # 4) Persist: merge back into the *persisted* source-of-truth and commit
                #    - Keep any actions for pairs not currently mismatching (so history isn't lost)
                latest_actions = edited_df[[KEY, "Action", "Details"]].copy()

                # Deduplicate keys (last edit wins)
                latest_actions = latest_actions.drop_duplicates(subset=[KEY], keep="last")

                # Update/insert into persisted_df by key
                base = persisted_df[persist_cols].copy()
                if base.empty:
                    merged_persisted = latest_actions
                else:
                    keep_mask = ~base[KEY].isin(latest_actions[KEY])
                    merged_persisted = pd.concat([base.loc[keep_mask], latest_actions], ignore_index=True)
                

                # Commit to GitHub
                csv_bytes = merged_persisted.to_csv(index=False).encode("utf-8")
                commit_msg = f"chore(tasks): update Tasks.csv via app ({datetime.utcnow().isoformat()}Z)"
                github_upsert_file(
                    REPO_OWNER, REPO_NAME, tasks_repo_path, GITHUB_TOKEN, csv_bytes, commit_msg
                )

                # Clear any Streamlit caches and rerun so everyone sees the new file immediately
                st.cache_data.clear()
                st.success("Tasks saved to GitHub ✅ Reloading …")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save tasks: {e}")

    with c2:
        st.download_button(
            "⬇️ Download current table as CSV",
            edited_df[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="Tasks.csv",
            mime="text/csv",
            use_container_width=True,
        )

















