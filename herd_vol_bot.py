import os
import logging
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime, timezone, timedelta

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

MSK_TZ = timezone(timedelta(hours=3))

BOT_TOKEN = os.getenv("BOT_TOKEN", "")

PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "")
PROXY_USER = os.getenv("PROXY_USER", "")
PROXY_PASS = os.getenv("PROXY_PASS", "")

if PROXY_HOST and PROXY_PORT:
    if PROXY_USER and PROXY_PASS:
        PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
    else:
        PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
else:
    PROXY_URL = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("herd-vol-bot")

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"

MAX_DAYS = 200
MAX_SCAN_SYMBOLS = 200

BASE_CORR_DAYS = 60
HIGH_CORR = 0.7

MIN_TURNOVER = 5_000_000
HEARTBEAT_INTERVAL = 1 * 3600
VOL_PERIOD_DAY = 8

AUTO_STATE: Dict[int, Dict[str, Any]] = {}
CORR_BASE: Dict[str, float] = {}
USER_SETTINGS: Dict[int, Dict[str, Any]] = {}


def normalize_symbol(s: str) -> str:
    s = s.upper().replace(" ", "")
    if not s.endswith("USDT"):
        s += "USDT"
    return s


def build_symbol_links(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    tv_url = f"https://www.tradingview.com/chart/?symbol=BYBIT%3A{symbol}"
    spot_url = f"https://www.bybit.com/trade/spot/{base}/USDT"
    return f'<a href="{tv_url}">TV</a> | <a href="{spot_url}">Bybit</a>'


def fetch_bybit_last_price(symbol: str) -> Optional[float]:
    params = {"category": "spot", "symbol": symbol}
    r = requests.get(BYBIT_TICKERS_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        return None

    lst = data.get("result", {}).get("list", [])
    if not lst:
        return None

    return float(lst[0]["lastPrice"])


def fetch_bybit_daily_candles(symbol: str) -> List[Dict[str, Any]]:
    params = {"category": "spot", "symbol": symbol, "interval": "D", "limit": MAX_DAYS}
    r = requests.get(BYBIT_KLINE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline error: {data.get('retMsg')}")

    lst = data.get("result", {}).get("list", [])
    lst = list(reversed(lst))

    out: List[Dict[str, Any]] = []
    for row in lst:
        dt = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
        out.append(
            {
                "time": dt,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }
        )
    return out


def fetch_symbols_for_scan(max_symbols: int, min_turnover: float) -> List[str]:
    r = requests.get(BYBIT_TICKERS_URL, params={"category": "spot"}, timeout=15)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit tickers error: {data.get('retMsg')}")

    arr = data.get("result", {}).get("list", [])
    out = []

    for row in arr:
        sym = row.get("symbol", "")
        if not sym.endswith("USDT"):
            continue

        try:
            turnover = float(row.get("turnover24h", 0.0))
        except Exception:
            continue

        if turnover < float(min_turnover):
            continue

        out.append({"symbol": sym, "turnover": turnover})

    out.sort(key=lambda x: x["turnover"], reverse=True)
    return [x["symbol"] for x in out[:max_symbols]]


def fetch_bybit_closes(symbol: str, interval: str, limit: int) -> List[float]:
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    r = requests.get(BYBIT_KLINE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline error: {data.get('retMsg')}")

    raw = data.get("result", {}).get("list", [])
    raw = list(reversed(raw))
    return [float(row[4]) for row in raw if len(row) >= 5]


def compute_returns_from_closes(closes: List[float], max_len: int) -> List[float]:
    closes = closes[-(max_len + 1) :]
    out: List[float] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        out.append((cur - prev) / prev if prev else 0.0)
    return out


def compute_daily_returns(candles: List[Dict[str, Any]], max_len: int) -> List[float]:
    closes = [c["close"] for c in candles][- (max_len + 1) :]
    out: List[float] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        out.append((cur - prev) / prev if prev else 0.0)
    return out


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 10:
        return 0.0

    xs = xs[-n:]
    ys = ys[-n:]

    mx = sum(xs) / n
    my = sum(ys) / n

    num = denx = deny = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        num += dx * dy
        denx += dx * dx
        deny += dy * dy

    if denx <= 0 or deny <= 0:
        return 0.0

    return num / ((denx ** 0.5) * (deny ** 0.5))


def compute_volatility_value(candles: List[Dict[str, Any]], period: int) -> float:
    n = len(candles)
    period = min(period, n - 1)
    tot = 0.0
    for i in range(period):
        c = candles[-2 - i]
        tot += c["high"] - c["low"]
    return tot / max(1, period)


def calc_vol_channel(
    candles: List[Dict[str, Any]], period: int = VOL_PERIOD_DAY
) -> Optional[Dict[str, float]]:
    if len(candles) < 2:
        return None

    last = candles[-1]
    avg = compute_volatility_value(candles, period)

    up = last["low"] + avg
    down = last["high"] - avg
    mid = (up + down) / 2
    half = (up - down) / 2

    return {
        "up": float(up),
        "down": float(down),
        "mid": float(mid),
        "half": float(half),
        "close": float(last["close"]),
    }


def get_user_settings(chat_id: int) -> Dict[str, Any]:
    defaults = {
        "min_turnover": MIN_TURNOVER,
        "corr_days": BASE_CORR_DAYS,
    }
    current = USER_SETTINGS.get(chat_id, {})
    merged = {**defaults, **current}
    USER_SETTINGS[chat_id] = merged
    return merged


def make_settings_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    st = get_user_settings(chat_id)

    rows = []

    t_opts = [1_000_000, 3_000_000, 5_000_000, 10_000_000]
    row = []
    for t in t_opts:
        txt = f"{t/1_000_000:.0f}M"
        if abs(float(st["min_turnover"]) - float(t)) < 1:
            txt += " ✅"
        row.append(InlineKeyboardButton(txt, callback_data=f"set_turnover:{t}"))
    rows.append(row)

    corr_opts = [30, 60, 90]
    row = []
    for c in corr_opts:
        txt = f"corr {c}d"
        if int(st["corr_days"]) == int(c):
            txt += " ✅"
        row.append(InlineKeyboardButton(txt, callback_data=f"set_corr:{c}"))
    rows.append(row)

    return InlineKeyboardMarkup(rows)


async def send_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = get_user_settings(chat_id)

    text = (
        "⚙️ <b>Herd volatility bot settings</b>\n\n"
        f"• Min. 24h turnover: <b>{int(st['min_turnover']):,} USDT</b>\n"
        f"• Correlation period: <b>{int(st['corr_days'])} days</b>\n\n"
        "Choose parameters below:"
    )

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=make_settings_keyboard(chat_id),
        disable_web_page_preview=True,
    )


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    chat_id = query.message.chat.id

    st = get_user_settings(chat_id)

    if data.startswith("set_turnover:"):
        st["min_turnover"] = float(data.split(":", 1)[1])
    elif data.startswith("set_corr:"):
        st["corr_days"] = int(data.split(":", 1)[1])

    USER_SETTINGS[chat_id] = st

    text = (
        "⚙️ <b>Settings updated</b>\n\n"
        f"• Min. 24h turnover: <b>{int(st['min_turnover']):,} USDT</b>\n"
        f"• Correlation period: <b>{int(st['corr_days'])} days</b>\n"
    )

    await query.edit_message_text(
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=make_settings_keyboard(chat_id),
        disable_web_page_preview=True,
    )


def get_main_keyboard(chat_id: Optional[int] = None) -> ReplyKeyboardMarkup:
    enabled = False
    if chat_id in AUTO_STATE:
        enabled = AUTO_STATE[chat_id].get("enabled", True)

    auto_btn = "🟢 AUTO" if enabled else "🔴 AUTO"
    kb = [
        [KeyboardButton(auto_btn)],
        [KeyboardButton("SETTINGS")],
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)


def classify_delta_corr(corr_base: float, corr_now: float) -> Optional[str]:
    if corr_base < 0.7:
        return None

    delta = corr_base - corr_now

    if delta < 0.3:
        return None
    elif delta < 0.4:
        return "weak"
    elif delta < 0.6:
        return "medium"
    else:
        return "strong"


async def update_base_correlation(context: ContextTypes.DEFAULT_TYPE):
    global CORR_BASE

    logger.info("Updating base daily correlation vs BTC...")

    try:
        btc_daily = fetch_bybit_daily_candles("BTCUSDT")
        btc_rets = compute_daily_returns(btc_daily, max_len=BASE_CORR_DAYS)
    except Exception as e:
        logger.error(f"Failed to load BTC daily for correlation: {e}")
        return

    try:
        symbols = fetch_symbols_for_scan(MAX_SCAN_SYMBOLS, MIN_TURNOVER)
    except Exception as e:
        logger.error(f"Failed to load symbol list for correlation: {e}")
        return

    new_corr: Dict[str, float] = {}

    for sym in symbols:
        if sym == "BTCUSDT":
            continue
        try:
            daily = fetch_bybit_daily_candles(sym)
            coin_rets = compute_daily_returns(daily, max_len=BASE_CORR_DAYS)
            corr = pearson_corr(btc_rets, coin_rets)
            new_corr[sym] = corr
        except Exception as e:
            logger.warning(f"Correlation error for {sym}: {e}")
            continue

    CORR_BASE = new_corr
    logger.info(f"Base correlation updated for {len(CORR_BASE)} symbols")


async def autoscan_worker(context: ContextTypes.DEFAULT_TYPE):
    cnt_total = cnt_daily_ok = cnt_base_ok = cnt_now_ok = 0
    cnt_strength = cnt_sent = cnt_prev_skip = cnt_err = 0
    t0 = datetime.now(timezone.utc)

    try:
        btc_daily = fetch_bybit_daily_candles("BTCUSDT")
        btc_daily_closes = [c["close"] for c in btc_daily]
        btc_daily_rets = compute_returns_from_closes(
            btc_daily_closes, max_len=BASE_CORR_DAYS
        )
    except Exception as e:
        logger.error(f"[autoscan] failed to load BTC daily: {e}")
        return

    try:
        btc_1h_closes = fetch_bybit_closes("BTCUSDT", interval="60", limit=60)
        btc_1h_rets = compute_returns_from_closes(btc_1h_closes, max_len=60)
    except Exception as e:
        logger.error(f"[autoscan] failed to load BTC 1h: {e}")
        return

    logger.info(f"[autoscan] start: chats={len(AUTO_STATE)}")

    for chat_id, state in list(AUTO_STATE.items()):
        if not state.get("enabled", True):
            continue

        st = get_user_settings(chat_id)
        min_turnover = float(st["min_turnover"])
        corr_days = int(st["corr_days"])

        try:
            symbols = fetch_symbols_for_scan(MAX_SCAN_SYMBOLS, min_turnover)
        except Exception as e:
            logger.error(f"[autoscan] symbol list error (chat={chat_id}): {e}")
            continue

        prev = state.setdefault("prev", {})

        for symbol in symbols:
            if symbol == "BTCUSDT":
                continue

            cnt_total += 1

            try:
                daily = fetch_bybit_daily_candles(symbol)
                if len(daily) < 10:
                    continue
                cnt_daily_ok += 1

                coin_daily_closes = [c["close"] for c in daily]
                coin_daily_rets = compute_returns_from_closes(
                    coin_daily_closes, max_len=corr_days
                )

                corr_base = pearson_corr(btc_daily_rets, coin_daily_rets)
                if corr_base >= HIGH_CORR:
                    cnt_base_ok += 1

                coin_1h_closes = fetch_bybit_closes(symbol, interval="60", limit=60)
                coin_1h_rets = compute_returns_from_closes(coin_1h_closes, max_len=60)
                corr_now = pearson_corr(btc_1h_rets, coin_1h_rets)
                cnt_now_ok += 1

                strength = classify_delta_corr(corr_base, corr_now)
                if strength is None:
                    continue
                cnt_strength += 1

                vol = calc_vol_channel(daily)
                if not vol:
                    continue

                up = vol["up"]
                down = vol["down"]
                close_last = vol["close"]

                mid = (up + down) / 2
                half = (up - down) / 2
                vol_pos_pct = (close_last - mid) / half * 100 if half > 0 else 0.0

                prev_strength = prev.get(symbol)
                if prev_strength == strength:
                    cnt_prev_skip += 1
                    continue

                emoji_map = {"weak": "🟠", "medium": "🟡", "strong": "🟢"}
                emoji = emoji_map.get(strength, "🟡")

                links = build_symbol_links(symbol)

                msg = (
                    f"{emoji} <b>{symbol}</b> — abnormal correlation vs BTC\n\n"
                    f"corr_base: <b>{corr_base:.2f}</b>\n"
                    f"corr_now: <b>{corr_now:.2f}</b>\n"
                    f"Δcorr: <b>{(corr_base - corr_now):.2f}</b>\n\n"
                    f"Position in daily volatility channel: <b>{vol_pos_pct:.1f}%</b>\n"
                    f"(-100% = lower band, +100% = upper band)\n\n"
                    f"{links}"
                )

                await context.bot.send_message(
                    chat_id,
                    msg,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                cnt_sent += 1
                prev[symbol] = strength

            except Exception as e:
                cnt_err += 1
                logger.error(f"[autoscan] error for {symbol} (chat={chat_id}): {e}")
                continue

    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info(
        f"[autoscan] done in {dt:.1f}s: total={cnt_total} daily_ok={cnt_daily_ok} "
        f"base>={HIGH_CORR}={cnt_base_ok} now_ok={cnt_now_ok} strength={cnt_strength} "
        f"sent={cnt_sent} prev_skip={cnt_prev_skip} errors={cnt_err}"
    )


async def toggle_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = AUTO_STATE.setdefault(chat_id, {"enabled": True, "prev": {}})

    if st["enabled"]:
        st["enabled"] = False
        await update.message.reply_text(
            "🔴 Autoscan disabled", reply_markup=get_main_keyboard(chat_id)
        )
    else:
        st["enabled"] = True
        await update.message.reply_text(
            "🟢 Autoscan enabled", reply_markup=get_main_keyboard(chat_id)
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    chat_id = update.effective_chat.id
    upper = text.upper()

    if upper in ("AUTO", "🟢 AUTO", "🔴 AUTO"):
        await toggle_auto(update, context)
        return

    if upper == "SETTINGS":
        await send_settings(update, context)
        return

    symbol = normalize_symbol(text)
    if len(symbol) >= 4 and symbol.endswith("USDT"):
        await show_symbol_info(symbol, update, context)
        return

    await update.message.reply_text(
        "Use the buttons or type a coin ticker (for example: BTC).",
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(chat_id),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    AUTO_STATE.setdefault(chat_id, {"enabled": True, "prev": {}})

    text = (
        "👋 <b>Volatility + BTC-correlation signal bot</b>\n\n"
        "Autoscan is enabled.\n"
        "The bot scans Bybit spot USDT pairs and looks for coins with strong base "
        "correlation to BTC that temporarily decouple on 1h timeframe, showing their "
        "position inside a daily volatility channel.\n\n"
        "Use the keyboard buttons or type a ticker (e.g., BTC)."
    )

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(chat_id),
        disable_web_page_preview=True,
    )


async def heartbeat_job(context: ContextTypes.DEFAULT_TYPE):
    now_msk = datetime.now(MSK_TZ)

    for chat_id in list(AUTO_STATE.keys()):
        try:
            await context.bot.send_message(
                chat_id,
                "🧪 Heartbeat: bot is running\n"
                f"🕒 {now_msk.strftime('%H:%M:%S')} MSK",
            )
        except Exception as e:
            logger.error(f"heartbeat error for {chat_id}: {e}")


async def show_symbol_info(
    symbol: str, update: Update, context: ContextTypes.DEFAULT_TYPE
):
    chat_id = update.effective_chat.id

    await update.message.reply_text(
        f"⏳ Fetching data for {symbol}...",
        reply_markup=get_main_keyboard(chat_id),
        disable_web_page_preview=True,
    )

    try:
        daily = fetch_bybit_daily_candles(symbol)
    except Exception as e:
        await update.message.reply_text(
            f"❌ Failed to load data: {e}",
            reply_markup=get_main_keyboard(chat_id),
        )
        return

    if len(daily) < 2:
        await update.message.reply_text(
            "Not enough data.",
            reply_markup=get_main_keyboard(chat_id),
        )
        return

    vol = calc_vol_channel(daily)
    if not vol:
        await update.message.reply_text(
            "Failed to calculate volatility channel.",
            reply_markup=get_main_keyboard(chat_id),
        )
        return

    up = vol["up"]
    down = vol["down"]
    mid = vol["mid"]
    half = vol["half"]
    close = vol["close"]

    pos_pct = (close - mid) / half * 100 if half > 0 else 0.0
    links = build_symbol_links(symbol)

    msg = (
        f"📊 <b>{symbol}</b>\n"
        f"Last price: <b>{close:.4f}</b>\n"
        f"Upper band: <b>{up:.4f}</b>\n"
        f"Lower band: <b>{down:.4f}</b>\n"
        f"Position in daily channel: <b>{pos_pct:.1f}%</b>\n\n"
        f"(-100% = lower band, +100% = upper band)\n\n"
        f"{links}"
    )

    await update.message.reply_text(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(chat_id),
        disable_web_page_preview=True,
    )


def main():
    AUTO_STATE.clear()

    if not BOT_TOKEN:
        raise RuntimeError(
            "Set BOT_TOKEN environment variable with your Telegram bot token."
        )

    builder = Application.builder().token(BOT_TOKEN)
    if PROXY_URL:
        builder = builder.proxy(PROXY_URL)
    app = builder.build()

    app.job_queue.run_repeating(
        heartbeat_job,
        interval=HEARTBEAT_INTERVAL,
        first=HEARTBEAT_INTERVAL
        - (datetime.now(MSK_TZ).minute * 60 + datetime.now(MSK_TZ).second),
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Herd volatility bot started")

    app.job_queue.run_repeating(autoscan_worker, interval=300, first=10)
    app.job_queue.run_repeating(update_base_correlation, interval=86400, first=5)
    app.job_queue.run_once(update_base_correlation, when=3)

    app.run_polling()


if __name__ == "__main__":
    main()
