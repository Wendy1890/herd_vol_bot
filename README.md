# Herd Volatility & BTC Correlation Bot (Bybit Spot)

Telegram bot that scans **Bybit spot USDT pairs** and looks for coins that:

- Have **high base correlation to BTC** on daily timeframe
- Temporarily **decouple from BTC on 1h timeframe**
- Show their position inside a **daily volatility channel**

Signals include a correlation breakdown + position inside the volatility channel and links to TradingView and Bybit spot.

## Features

- 🔍 Data source:
  - Bybit spot (`/USDT` pairs)
- 📈 Signal logic:
  - Base correlation vs BTC from daily closes (configurable period)
  - Current correlation vs BTC from 1h closes
  - Δcorr = corr_base − corr_now
  - Only coins with sufficiently large correlation drop are signaled
- 📊 Volatility channel:
  - Daily candles
  - Simple high–low based volatility band
  - Position in channel: -100% (lower band) … +100% (upper band)
- ⚙️ Per-chat settings:
  - Minimum 24h turnover (USDT)
  - Correlation lookback period (days)
- 🔁 Autoscan:
  - Global scan every 5 minutes
  - Per-chat autoscan ON/OFF via reply keyboard
- 🫀 Heartbeat:
  - Message every hour (MSK) confirming the bot is alive
