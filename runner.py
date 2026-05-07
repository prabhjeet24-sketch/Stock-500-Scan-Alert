"""
runner.py — Consolidated Next-Day Stock Scanner (Oversold + Momentum) for GitHub Actions

Features:
- Static universe from sp500_universe.csv (no Wikipedia dependency)
- Oversold + Momentum scoring + explicit Next-Day setup scoring
- Bullish RSI divergence detection
- Earnings proximity guardrail (Finnhub earnings calendar)
- Extreme gap day penalty
- Sector caps (avoid concentration in tech)
- ATR-based stop/target suggestions
- Multi-timeframe confirmation (Daily + 4H via Finnhub 240-minute candles)
- Email (Outlook/Hotmail SMTP) + Slack webhook output
- DST-proof weekday schedule support (use GH Actions + ET time gate)

Environment variables (set via GitHub Actions env + Secrets):
Required:
  FINNHUB_API_KEY
  EMAIL_TO (default set in workflow to prabhjeetsingh@hotmail.com)

If sending email:
  SMTP_HOST = smtp-mail.outlook.com
  SMTP_PORT = 587
  SMTP_USER = prabhjeetsingh@hotmail.com
  SMTP_PASS = <password/app-password>
  EMAIL_FROM = prabhjeetsingh@hotmail.com

Optional:
  SLACK_WEBHOOK_URL

Tuning:
  TOP_N=25
  TOP_OVERSOLD=10
  TOP_MOMENTUM=10
  MAX_PER_SECTOR=5
  EXCLUDE_HEALTHCARE=1
  EXCLUDE_CHINA_ADR=1
  RUN_TZ=America/New_York
  RUN_HHMM=0830
  RUN_WINDOW_MINUTES=10
"""

import os
import time
import json
import math
import smtplib
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dateutil import parser as dateparser

# -----------------------------
# Constants / Defaults
# -----------------------------
FINNHUB_BASE = "https://finnhub.io/api/v1"

CACHE_DIR = ".cache"
CANDLE_CACHE_FILE = os.path.join(CACHE_DIR, "candles_cache.json")
EARNINGS_CACHE_FILE = os.path.join(CACHE_DIR, "earnings_cache.json")

UNIVERSE_CSV = "sp500_universe.csv"

LOOKBACK_DAYS_DAILY = 320
LOOKBACK_DAYS_4H = 90

RS_LOOKBACK = 63
PULLBACK_LOOKBACK = 20

MIN_PRICE = 5.0
MIN_AVG_VOL = 600_000

EARNINGS_GUARD_DAYS = 4
EXTREME_GAP_PCT = 8.0

RSI_OVERSOLD = 33
RSI_DEEP_OVERSOLD = 28

RSI_MOMO_LOW = 50
RSI_MOMO_HIGH = 70

CHINA_ADR_HINTS = set(["BABA","JD","PDD","BIDU","NIO","LI","XPEV","NTES","TME","YMM","ZTO","BEKE"])
HEALTHCARE_SECTOR_NAME = "Health Care"

# Finnhub free-tier safety (conservative)
MAX_CALLS_PER_MIN = 50
SLEEP_BETWEEN_CALLS = 60.0 / MAX_CALLS_PER_MIN

# -----------------------------
# Env helpers
# -----------------------------
def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)).strip())
    except:
        return default

def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)).strip())
    except:
        return default

def env_flag(name, default="1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def utc_now():
    return datetime.now(timezone.utc)

def to_unix(dt: datetime) -> int:
    return int(dt.timestamp())

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

# -----------------------------
# Time gate (DST-proof for dual cron)
# -----------------------------
def should_run_now():
    tz_name = os.getenv("RUN_TZ", "America/New_York")
    hhmm = os.getenv("RUN_HHMM", "0830")
    window_minutes = env_int("RUN_WINDOW_MINUTES", 10)

    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    target_h = int(hhmm[:2])
    target_m = int(hhmm[2:])
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)

    delta_min = abs((now - target).total_seconds()) / 60.0
    return delta_min <= window_minutes

# -----------------------------
# Rate limiter + Finnhub GET
# -----------------------------
class RateLimiter:
    def __init__(self, sleep_between=SLEEP_BETWEEN_CALLS):
        self.sleep_between = sleep_between
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.sleep_between:
            time.sleep(self.sleep_between - elapsed)
        self.last_call = time.time()

rl = RateLimiter()

def finnhub_get(path, params, api_key):
    url = f"{FINNHUB_BASE}{path}"
    params = dict(params)
    params["token"] = api_key

    for attempt in range(3):
        rl.wait()
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(2.0 + attempt * 2.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 + attempt)
    return None

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame):
    if len(df) < 2:
        return pd.Series([np.nan]*len(df))
    prev_close = df["c"].shift(1)
    tr1 = df["h"] - df["l"]
    tr2 = (df["h"] - prev_close).abs()
    tr3 = (df["l"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period=14):
    return true_range(df).rolling(period).mean()

def compute_features(df: pd.DataFrame):
    df = df.copy()
    close = df["c"]
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["rsi14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = signal_line
    df["macd_hist"] = hist

    df["atr14"] = atr(df, 14)
    df["atr14_sma20"] = df["atr14"].rolling(20).mean()
    df["atr_contraction"] = df["atr14"] / df["atr14_sma20"]
    return df

# -----------------------------
# Candles + caching
# -----------------------------
def load_candle_cache():
    ensure_cache_dir()
    return load_json(CANDLE_CACHE_FILE, {})

def save_candle_cache(cache):
    save_json(CANDLE_CACHE_FILE, cache)

def load_earnings_cache():
    ensure_cache_dir()
    return load_json(EARNINGS_CACHE_FILE, {})

def save_earnings_cache(cache):
    save_json(EARNINGS_CACHE_FILE, cache)

def get_candles(symbol, api_key, resolution="D", lookback_days=320, cache=None, cache_ttl_hours=12):
    if cache is None:
        cache = load_candle_cache()

    key = f"{symbol}:{resolution}:{lookback_days}"
    now = utc_now()
    ttl = timedelta(hours=cache_ttl_hours)

    if key in cache:
        try:
            ts = dateparser.parse(cache[key]["ts"])
            if now - ts < ttl:
                data = cache[key]["data"]
                df = pd.DataFrame({
                    "t": data["t"], "o": data["o"], "h": data["h"],
                    "l": data["l"], "c": data["c"], "v": data["v"]
                })
                df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
                return df.sort_values("t").reset_index(drop=True), cache
        except:
            pass

    frm = now - timedelta(days=lookback_days)
    payload = finnhub_get(
        "/stock/candle",
        params={"symbol": symbol, "resolution": resolution, "from": to_unix(frm), "to": to_unix(now)},
        api_key=api_key
    )
    if not payload or payload.get("s") != "ok":
        return None, cache

    df = pd.DataFrame({
        "t": payload["t"], "o": payload["o"], "h": payload["h"],
        "l": payload["l"], "c": payload["c"], "v": payload["v"]
    })
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.sort_values("t").reset_index(drop=True)

    cache[key] = {"ts": now.isoformat(), "data": payload}
    return df, cache

def get_next_earnings(symbol, api_key, cache=None, cache_ttl_hours=24):
    if cache is None:
        cache = load_earnings_cache()

    now = utc_now()
    ttl = timedelta(hours=cache_ttl_hours)

    if symbol in cache:
        try:
            ts = dateparser.parse(cache[symbol]["ts"])
            if now - ts < ttl:
                return cache[symbol].get("earnings_date"), cache
        except:
            pass

    start = now.date().isoformat()
    end = (now + timedelta(days=60)).date().isoformat()
    payload = finnhub_get("/calendar/earnings", params={"from": start, "to": end}, api_key=api_key)

    earnings_date = None
    try:
        if payload and "earningsCalendar" in payload:
            dates = [row["date"] for row in payload["earningsCalendar"]
                     if row.get("symbol") == symbol and row.get("date")]
            if dates:
                earnings_date = sorted(dates)[0]
    except:
        earnings_date = None

    cache[symbol] = {"ts": now.isoformat(), "earnings_date": earnings_date}
    return earnings_date, cache

# -----------------------------
# Setup helpers
# -----------------------------
def avg_volume(df: pd.DataFrame, lookback=20):
    if len(df) < lookback + 1:
        return np.nan
    return float(df["v"].iloc[-lookback:].mean())

def compute_gap_pct(df: pd.DataFrame):
    if len(df) < 2:
        return np.nan
    prev_close = df["c"].iloc[-2]
    today_open = df["o"].iloc[-1]
    if prev_close <= 0:
        return np.nan
    return (today_open / prev_close - 1.0) * 100.0

def close_above_yesterday(df: pd.DataFrame):
    return len(df) >= 2 and df["c"].iloc[-1] > df["c"].iloc[-2]

def breakout_above_yesterday_high(df: pd.DataFrame):
    return len(df) >= 2 and df["c"].iloc[-1] > df["h"].iloc[-2]

def close_location_value(df: pd.DataFrame):
    h = df["h"].iloc[-1]
    l = df["l"].iloc[-1]
    c = df["c"].iloc[-1]
    rng = h - l
    if rng <= 0:
        return 0.0
    return ((c - l) / rng) * 2.0 - 1.0

def close_near_high(df: pd.DataFrame, top_pct=0.25):
    h = df["h"].iloc[-1]
    l = df["l"].iloc[-1]
    c = df["c"].iloc[-1]
    rng = h - l
    if rng <= 0:
        return False
    return (h - c) <= (rng * top_pct)

def volume_surge(df: pd.DataFrame, lookback=20, multiple=1.1):
    if len(df) < lookback + 1:
        return False
    v_today = df["v"].iloc[-1]
    v_avg = df["v"].iloc[-lookback:].mean()
    return v_today >= multiple * v_avg

def pullback_from_high(df: pd.DataFrame, lookback=PULLBACK_LOOKBACK):
    if len(df) < lookback + 1:
        return np.nan
    recent_high = df["h"].iloc[-lookback:].max()
    last_close = df["c"].iloc[-1]
    if recent_high <= 0:
        return np.nan
    return (1.0 - last_close / recent_high) * 100.0

def bullish_rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, window=6):
    if len(df) < window + 2:
        return False
    p = df["c"].iloc[-window:]
    r = rsi_series.iloc[-window:]
    if r.isna().any():
        return False
    price_slope = np.polyfit(np.arange(window), p.values, 1)[0]
    rsi_slope = np.polyfit(np.arange(window), r.values, 1)[0]
    return (price_slope < 0) and (rsi_slope > 0)

def earnings_penalty(earnings_date_str):
    if not earnings_date_str:
        return 0.0, None
    try:
        ed = dateparser.parse(earnings_date_str).date()
        today = utc_now().date()
        delta = (ed - today).days
        if abs(delta) <= EARNINGS_GUARD_DAYS:
            return 12.0, delta
        elif abs(delta) <= 10:
            return 6.0, delta
        return 0.0, delta
    except:
        return 0.0, None

def atr_trade_levels(entry, atr14, stop_atr=1.6, tgt1_atr=2.2, tgt2_atr=3.2):
    if atr14 is None or pd.isna(atr14) or atr14 <= 0 or entry is None or pd.isna(entry):
        return None, None, None
    stop = entry - stop_atr * atr14
    tgt1 = entry + tgt1_atr * atr14
    tgt2 = entry + tgt2_atr * atr14
    return stop, tgt1, tgt2

# -----------------------------
# Scoring
# -----------------------------
def normalize(x, lo, hi):
    if x is None or pd.isna(x):
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def oversold_score(df_feat: pd.DataFrame):
    last = df_feat.iloc[-1]
    c = float(last["c"])
    r = last["rsi14"]
    s20 = last["sma20"]
    s50 = last["sma50"]

    score = 0.0

    if not pd.isna(r):
        score += (1.0 - normalize(r, 20, 55)) * 45.0
        if r <= RSI_OVERSOLD:
            score += 10.0
        if r <= RSI_DEEP_OVERSOLD:
            score += 8.0

    if not pd.isna(s20) and s20 > 0:
        dist20 = (s20 / c - 1.0) * 100.0
        score += normalize(dist20, 0, 12) * 15.0

    if not pd.isna(s50) and s50 > 0:
        dist50 = (s50 / c - 1.0) * 100.0
        score += normalize(dist50, 0, 18) * 10.0

    div = bullish_rsi_divergence(df_feat, df_feat["rsi14"], window=6)
    if div:
        score += 12.0

    if close_above_yesterday(df_feat):
        score = score * 1.08 + 3.0
    else:
        score *= 0.90

    gap = compute_gap_pct(df_feat)
    if not pd.isna(gap) and abs(gap) >= EXTREME_GAP_PCT:
        score -= 12.0

    return score, {"divergence": div, "gap_pct": gap}

def momentum_score(df_feat: pd.DataFrame, spy_feat: pd.DataFrame):
    last = df_feat.iloc[-1]
    c = float(last["c"])
    s50 = last["sma50"]
    s200 = last["sma200"]
    r = last["rsi14"]
    macd_val = last["macd"]
    macd_hist = last["macd_hist"]

    if pd.isna(s50) or pd.isna(s200):
        return 0.0, {"trend_ok": False}

    trend_ok = (c > s50) and (s50 > s200)
    if not trend_ok:
        return 0.0, {"trend_ok": False}

    score = 25.0

    if not pd.isna(r):
        if RSI_MOMO_LOW <= r <= RSI_MOMO_HIGH:
            score += 18.0
        elif r < RSI_MOMO_LOW:
            score += 8.0
        else:
            score += 6.0

    if not pd.isna(macd_val) and macd_val > 0:
        score += 10.0
    if not pd.isna(macd_hist) and macd_hist > 0:
        score += 6.0

    rs_bonus = 0.0
    if spy_feat is not None and len(df_feat) > RS_LOOKBACK and len(spy_feat) > RS_LOOKBACK:
        stock_ret = df_feat["c"].iloc[-1] / df_feat["c"].iloc[-1 - RS_LOOKBACK] - 1.0
        spy_ret = spy_feat["c"].iloc[-1] / spy_feat["c"].iloc[-1 - RS_LOOKBACK] - 1.0
        rs = stock_ret - spy_ret
        rs_bonus = normalize(rs, -0.05, 0.15) * 18.0
        score += rs_bonus

    pb = pullback_from_high(df_feat, lookback=PULLBACK_LOOKBACK)
    pb_score = 0.0
    if not pd.isna(pb):
        if 3.0 <= pb <= 8.0:
            pb_score = 18.0
        elif 1.5 <= pb < 3.0:
            pb_score = 10.0
        elif 8.0 < pb <= 12.0:
            pb_score = 8.0
    score += pb_score

    if close_above_yesterday(df_feat):
        score += 6.0

    gap = compute_gap_pct(df_feat)
    if not pd.isna(r) and r > 75 and not pd.isna(gap) and gap > EXTREME_GAP_PCT:
        score -= 15.0

    return score, {"trend_ok": True, "pullback_pct": pb, "rs_bonus": rs_bonus}

def next_day_setup_score(df_feat: pd.DataFrame, earn_pen: float):
    score = 0.0

    if close_above_yesterday(df_feat):
        score += 8.0
    else:
        score -= 6.0

    if close_near_high(df_feat, top_pct=0.25):
        score += 10.0
    elif close_near_high(df_feat, top_pct=0.40):
        score += 6.0

    clv = close_location_value(df_feat)
    score += max(0.0, clv) * 6.0

    if breakout_above_yesterday_high(df_feat):
        score += 8.0

    if volume_surge(df_feat, lookback=20, multiple=1.1):
        score += 7.0
    elif volume_surge(df_feat, lookback=20, multiple=0.95):
        score += 3.0

    atr_contr = df_feat.iloc[-1].get("atr_contraction", np.nan)
    if not pd.isna(atr_contr):
        if atr_contr < 0.95:
            score += 4.0
        elif atr_contr > 1.15:
            score -= 3.0

    gap = compute_gap_pct(df_feat)
    if not pd.isna(gap) and abs(gap) >= EXTREME_GAP_PCT:
        score -= 10.0

    score -= earn_pen
    return score

# -----------------------------
# Multi-timeframe 4H confirmation
# -----------------------------
def multi_tf_confirm(symbol, api_key, candle_cache):
    df4h, candle_cache = get_candles(
        symbol, api_key, resolution="240", lookback_days=LOOKBACK_DAYS_4H,
        cache=candle_cache, cache_ttl_hours=6
    )
    if df4h is None or len(df4h) < 120:
        return False, {"tf_confirm_4h": False, "4h_reason": "no_4h"}, candle_cache

    feat4 = compute_features(df4h)
    last4 = feat4.iloc[-1]

    rsi_ok = (last4["rsi14"] >= 50) if not pd.isna(last4["rsi14"]) else False
    macd_ok = (last4["macd_hist"] > 0) if not pd.isna(last4["macd_hist"]) else False
    trend_ok = (last4["c"] > last4["sma50"]) if not pd.isna(last4["sma50"]) else False

    confirm = bool(rsi_ok and macd_ok and trend_ok)
    meta = {
        "tf_confirm_4h": confirm,
        "4h_rsi14": float(last4["rsi14"]) if not pd.isna(last4["rsi14"]) else None,
        "4h_macd_hist": float(last4["macd_hist"]) if not pd.isna(last4["macd_hist"]) else None,
        "4h_trend_ok": bool(trend_ok),
        "4h_reason": "ok" if confirm else "no_confirm"
    }
    return confirm, meta, candle_cache

# -----------------------------
# Universe (static CSV)
# -----------------------------
def load_universe_from_csv(path=UNIVERSE_CSV, exclude_healthcare=True, exclude_china=True):
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["name"] = df.get("name", df["ticker"]).fillna(df["ticker"]).astype(str)
    df["sector"] = df.get("sector", "").fillna("").astype(str)

    if exclude_healthcare:
        df = df[df["sector"] != HEALTHCARE_SECTOR_NAME]

    if exclude_china:
        df = df[~df["ticker"].isin(CHINA_ADR_HINTS)]

    return df[["ticker", "name", "sector"]].drop_duplicates("ticker").reset_index(drop=True)

# -----------------------------
# Sector caps
# -----------------------------
def apply_sector_caps(df_ranked, top_n=25, max_per_sector=5):
    picks = []
    counts = defaultdict(int)

    for idx, row in df_ranked.iterrows():
        sector = row.get("sector", "") or "Unknown"
        if counts[sector] >= max_per_sector:
            continue
        picks.append(idx)
        counts[sector] += 1
        if len(picks) >= top_n:
            break

    out = df_ranked.loc[picks].copy().reset_index(drop=True)
    return out, dict(counts)

# -----------------------------
# Notifications (Slack + Email)
# -----------------------------
def send_slack(webhook_url: str, text: str):
    if not webhook_url:
        return
    payload = {"text": text}
    r = requests.post(webhook_url, json=payload, timeout=20)
    r.raise_for_status()

def send_email(subject: str, body: str, attachment_path: str | None = None):
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    to   = os.getenv("EMAIL_TO", "")
    frm  = os.getenv("EMAIL_FROM", user)

    if not (host and user and pwd and to):
        print("Email not sent: missing SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = frm
    msg["To"] = to
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="text", subtype="csv", filename=os.path.basename(attachment_path))

    # Outlook/Hotmail SMTP submission uses STARTTLS on port 587.
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(user, pwd)
        s.send_message(msg)

# -----------------------------
# Formatting for Email/Slack
# -----------------------------
def format_alert_message_multi(df_combined, df_oversold, df_momentum, sector_counts):
    def fmt_row(r):
        tf = "✅" if bool(r.get("tf_confirm_4h", False)) else "❌"
        pb = r.get("pullback_pct", np.nan)
        stop = r.get("stop", np.nan)
        t1 = r.get("target1", np.nan)
        t2 = r.get("target2", np.nan)
        return (
            f"{r['ticker']} | ${r['price']:.2f} | ND {r['next_day_score']:.1f} | "
            f"Comb {r['combined_score']:.1f} | RSI {r['rsi14']:.1f} | "
            f"PB {pb if pd.notna(pb) else float('nan'):.1f}% | 4H {tf} | "
            f"Stop {stop if pd.notna(stop) else float('nan'):.2f} | "
            f"T1 {t1 if pd.notna(t1) else float('nan'):.2f} | "
            f"T2 {t2 if pd.notna(t2) else float('nan'):.2f}"
        )

    lines = []
    lines.append("📈 Next-Day Scanner Picks (Morning Run)")
    lines.append(f"Sector caps (Combined): {sector_counts}")
    lines.append("")

    lines.append("🏆 COMBINED (Top, sector-capped)")
    for _, r in df_combined.iterrows():
        lines.append(fmt_row(r))
    lines.append("")

    lines.append("🧲 OVERSOLD (Top)")
    for _, r in df_oversold.iterrows():
        lines.append(fmt_row(r))
    lines.append("")

    lines.append("🚀 MOMENTUM (Top)")
    for _, r in df_momentum.iterrows():
        lines.append(fmt_row(r))

    return "\n".join(lines)

# -----------------------------
# Main scan function
# -----------------------------
def run_scan(api_key: str):
    ensure_cache_dir()
    candle_cache = load_candle_cache()
    earnings_cache = load_earnings_cache()

    # Env-controlled knobs
    exclude_healthcare = env_flag("EXCLUDE_HEALTHCARE", "1")
    exclude_china = env_flag("EXCLUDE_CHINA_ADR", "1")

    top_n = env_int("TOP_N", 25)
    top_oversold = env_int("TOP_OVERSOLD", 10)
    top_momentum = env_int("TOP_MOMENTUM", 10)
    max_per_sector = env_int("MAX_PER_SECTOR", 5)

    w_oversold = env_float("W_OVERSOLD", 0.50)
    w_momentum = env_float("W_MOMENTUM", 0.50)
    next_day_weight = env_float("W_NEXTDAY", 0.35)

    # Load universe
    univ = load_universe_from_csv(UNIVERSE_CSV, exclude_healthcare=exclude_healthcare, exclude_china=exclude_china)

    # Fetch SPY for relative strength
    spy_df, candle_cache = get_candles("SPY", api_key, resolution="D", lookback_days=LOOKBACK_DAYS_DAILY,
                                       cache=candle_cache, cache_ttl_hours=12)
    if spy_df is None:
        raise RuntimeError("Could not fetch SPY candles (required).")
    spy_feat = compute_features(spy_df)

    rows = []
    for _, u in univ.iterrows():
        sym = u["ticker"]

        df, candle_cache = get_candles(sym, api_key, resolution="D", lookback_days=LOOKBACK_DAYS_DAILY,
                                       cache=candle_cache, cache_ttl_hours=12)
        if df is None or len(df) < 210:
            continue

        price = float(df["c"].iloc[-1])
        if price < MIN_PRICE:
            continue

        av = avg_volume(df, 20)
        if pd.isna(av) or av < MIN_AVG_VOL:
            continue

        feat = compute_features(df)

        # Earnings guard
        ed, earnings_cache = get_next_earnings(sym, api_key, cache=earnings_cache, cache_ttl_hours=24)
        ep, edelta = earnings_penalty(ed)

        # Scores
        o_raw, o_meta = oversold_score(feat)
        m_raw, m_meta = momentum_score(feat, spy_feat)
        nd = next_day_setup_score(feat, ep)

        o = o_raw - ep
        m = m_raw - ep

        # 4H confirm
        tf_ok, tf_meta, candle_cache = multi_tf_confirm(sym, api_key, candle_cache)

        combined = (w_oversold * o) + (w_momentum * m) + (next_day_weight * nd)
        combined += 6.0 if tf_ok else -2.0

        atr14 = float(feat["atr14"].iloc[-1]) if not pd.isna(feat["atr14"].iloc[-1]) else None
        stop, t1, t2 = atr_trade_levels(price, atr14)

        rows.append({
            "ticker": sym,
            "name": u["name"],
            "sector": u["sector"],
            "price": price,
            "avg_vol_20d": float(av),
            "rsi14": float(feat["rsi14"].iloc[-1]) if not pd.isna(feat["rsi14"].iloc[-1]) else np.nan,
            "gap_pct": float(o_meta.get("gap_pct", np.nan)) if not pd.isna(o_meta.get("gap_pct", np.nan)) else np.nan,
            "close_above_yday": bool(close_above_yesterday(feat)),
            "breakout_yday_high": bool(breakout_above_yesterday_high(feat)),
            "bullish_divergence": bool(o_meta.get("divergence", False)),
            "pullback_pct": float(m_meta.get("pullback_pct", np.nan)) if not pd.isna(m_meta.get("pullback_pct", np.nan)) else np.nan,
            "earnings_date": ed,
            "earnings_in_days": edelta,
            "oversold_score": float(o),
            "momentum_score": float(m),
            "next_day_score": float(nd),
            "combined_score": float(combined),
            "atr14": atr14,
            "stop": stop,
            "target1": t1,
            "target2": t2,
            **tf_meta
        })

    save_candle_cache(candle_cache)
    save_earnings_cache(earnings_cache)

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return df_all, None, None, None, None

    # Primary sort: tomorrow setups first
    df_all = df_all.sort_values(["next_day_score", "combined_score"], ascending=False)

    # Combined Top N with sector caps
    df_combined, sector_counts = apply_sector_caps(df_all, top_n=top_n, max_per_sector=max_per_sector)

    # Oversold Top
    df_oversold = df_all.sort_values(["oversold_score", "next_day_score"], ascending=False).head(top_oversold).reset_index(drop=True)

    # Momentum Top
    df_momentum = df_all.sort_values(["momentum_score", "next_day_score"], ascending=False).head(top_momentum).reset_index(drop=True)

    return df_all, df_combined, df_oversold, df_momentum, sector_counts

# -----------------------------
# Entrypoint
# -----------------------------
def main():
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing FINNHUB_API_KEY")

    os.makedirs(CACHE_DIR, exist_ok=True)

    df_all, df_combined, df_oversold, df_momentum, sector_counts = run_scan(api_key)

    if df_all is None or df_all.empty:
        raise SystemExit("Scan returned no results (check universe CSV, filters, or API limits).")

    # Save outputs for artifacts
    out_all = os.path.join(CACHE_DIR, "scan_all.csv")
    out_combined = os.path.join(CACHE_DIR, "next_day_picks_combined.csv")
    out_oversold = os.path.join(CACHE_DIR, "next_day_picks_oversold.csv")
    out_momentum = os.path.join(CACHE_DIR, "next_day_picks_momentum.csv")

    df_all.to_csv(out_all, index=False)
    df_combined.to_csv(out_combined, index=False)
    df_oversold.to_csv(out_oversold, index=False)
    df_momentum.to_csv(out_momentum, index=False)

    msg = format_alert_message_multi(df_combined, df_oversold, df_momentum, sector_counts)

    # Slack optional
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if slack_webhook:
        send_slack(slack_webhook, msg)

    # Email (Outlook/Hotmail SMTP: smtp-mail.outlook.com:587 STARTTLS).
    send_email(
        subject="Next-Day Scanner Picks (8:30am ET) — Combined + Oversold + Momentum",
        body=msg,
        attachment_path=out_combined
    )

    print("✅ Done. Outputs saved in .cache/")

if __name__ == "__main__":
    # If you use dual cron (12:30 + 13:30 UTC), this gate ensures only the true 8:30am ET run proceeds.
    if not should_run_now():
        print("Exiting: not within ET run window.")
        raise SystemExit(0)

    main()
