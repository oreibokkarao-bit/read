"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  🔱⚡ GOD OF GODS SQUEEZE SCANNER v5.0 - RESEARCH EDITION                                 ║
║  ════════════════════════════════════════════════════════════════════════════════════════║
║  MAJOR UPGRADES FROM merged_squeeze.md RESEARCH:                                          ║
║                                                                                           ║
║  1.  BTC REGIME FILTER - Global Veto (BTC>50SMA + BTC.D Down = RISK_ON)                  ║
║  2.  SPOT-LED vs PERP-LED TRAP - Perp/Spot Vol >4x = VETO (2.3x higher expectancy)       ║
║  3.  TRUE CVD ENGINE - Cumulative Volume Delta with Slope & Acceleration                  ║
║  4.  OFI Z-SCORE - Order Flow Imbalance with rolling normalization                        ║
║  5.  LIQUIDATION PROXIMITY BONUS - ≤25bp cluster = full bonus, decay to 60bp             ║
║  6.  PULLBACK-RETEST ENTRY - Wait for POC retest (2.9 R:R vs 0.9 breakout)               ║
║  7.  ATR STOP-LOSS 2.3x - Proven optimal, cuts whipsaws by 37%                           ║
║  8.  GOD CANDLE DETECTION - >12% in 5m + Volume >10x baseline                            ║
║  9.  FUNDING FLIP BONUS - Negative→Positive + Rising OI = +7 score (58% PPV)             ║
║  10. CROWDED FUNDING VETO - Funding >0.1% (10bps) = HARD REJECT                          ║
║  11. WATERFALL GATE SYSTEM - Sequential: Viability→Compression→Ignition→Confirmation     ║
║  12. RVOL THRESHOLDS - Quiet (<1.0), Alert (>3.0), Extreme (>5.0)                        ║
║  13. BB WIDTH PERCENTILE GATE - <10th percentile = SLEEP mode                            ║
║  14. CHANDELIER EXIT - Trail at 3x ATR below highest high                                ║
║  15. OI ACCELERATION - 2nd derivative for early momentum detection                        ║
║  16. SPREAD VETO - Spread >1.5% = IMMEDIATE REJECT                                       ║
║  17. ARCHETYPE CLASSIFICATION - zec/mel/anime/turbo/moca patterns                        ║
║  18. 7-METRIC SHORT SQUEEZE TP ENGINE - See TP_Prediction_Research.md                    ║
║                                                                                           ║
║  ═══════════════════════════════════════════════════════════════════════════════════════ ║
║  7-METRIC SHORT SQUEEZE TP SYSTEM (NEW):                                                  ║
║  ───────────────────────────────────────                                                  ║
║  Research: "SHORT SQUEEZE TPs are determined by liquidation clusters, funding rate       ║
║            normalization, and OI exhaustion, NOT by fixed sigma levels."                 ║
║                                                                                           ║
║  Metric 1: LIQUIDATION HEATMAP - TPs placed just BELOW major liquidation clusters        ║
║            (78% of successful squeezes reverse near these walls)                         ║
║  Metric 2: FUNDING RATE NORMALIZATION - Exit when FR returns to ~0% from extreme neg     ║
║  Metric 3: OI EXHAUSTION PATTERN - Exit when OI peaks then declines 10%+                 ║
║  Metric 4: VPVR HIGH VOLUME NODES - TPs at HVNs (price magnets), fast through LVNs       ║
║  Metric 5: RSI EXHAUSTION - Exit zone when RSI > 85-90 on 4H/12H timeframes             ║
║  Metric 6: CVD MOMENTUM DECAY - Exit when CVD growth rate slows (2nd deriv negative)     ║
║  Metric 7: VOLUME CLIMAX PATTERN - Volume > 3x avg + Shooting Star = PANIC EXIT         ║
║                                                                                           ║
║  TP LADDER ALLOCATION:                                                                    ║
║  • TP1: 40% at First HVN (funding still < -0.30%)                                        ║
║  • TP2: 35% at Second HVN (OI peaks OR funding < -0.15%)                                 ║
║  • TP3: 15% on Exhaustion (RSI > 85 OR funding → 0% OR Volume Climax)                   ║
║  • RUNNER: 10% trailed with Chandelier Exit (3x ATR from high)                           ║
║                                                                                           ║
║  Research shows these upgrades provide:                                                   ║
║  • 62% win rate in RISK_ON vs 27% in RISK_OFF                                           ║
║  • Max drawdown reduced from -23.4% to -11.6% with regime filter                        ║
║  • 2.9 R:R with pullback entry vs 0.9 with breakout entry                               ║
║  • 37% reduction in stop-outs with 2.3x ATR stops                                       ║
║  • +95-115% avg gains on squeezes vs +10% with fixed TP                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import aiohttp
import aiosqlite
import numpy as np
import time
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# Try uvloop for performance (Linux/Mac)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_ENABLED = True
except ImportError:
    UVLOOP_ENABLED = False

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.logging import RichHandler

from scipy.stats import linregress

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - v5.0 RESEARCH-BACKED SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GodOfGodsConfigV5:
    """Master configuration for God of Gods Scanner v5.0 - Research Edition"""
    
    # === EXCHANGE ENDPOINTS ===
    BINANCE_FUTURES_API: str = "https://fapi.binance.com"
    BINANCE_SPOT_API: str = "https://api.binance.com"
    BYBIT_API: str = "https://api.bybit.com"
    OKX_API: str = "https://www.okx.com"
    
    # === SCAN SETTINGS ===
    SCAN_INTERVAL_SECONDS: float = 60.0  # 1 min refresh for swing trading (4-12hr holds)
    KLINE_TIMEFRAME: str = "15m"  # 15m timeframe for swing trades
    KLINE_LIMIT: int = 200  # 200 bars = 50 hours of data
    KLINE_LIMIT_5M: int = 100  # For 5m timeframe analysis
    TOP_CANDIDATES: int = 20
    
    # === UNIVERSE SETTINGS (SCAN ALL COINS!) ===
    # Research: "Universe: Scan ALL USDT-perp + spot (microcaps included)"
    # Binance has ~350 USDT perps, Bybit has ~300+
    SCAN_ALL_SYMBOLS: bool = True  # Scan entire universe
    MAX_SYMBOLS_TO_SCAN: int = 999  # Effectively unlimited
    
    # === EXCHANGE SELECTION ===
    USE_BINANCE: bool = True
    USE_BYBIT: bool = True
    USE_OKX: bool = False  # Can enable later
    
    # === PARALLEL PROCESSING ===
    MAX_CONCURRENT_REQUESTS: int = 15  # Increased for more symbols
    API_DELAY_SECONDS: float = 0.05  # Faster batching
    BATCH_DELAY_SECONDS: float = 0.2  # Delay between batches to avoid rate limits
    
    # === BTC REGIME FILTER (Research: 62% win rate when RISK_ON) ===
    BTC_SYMBOL: str = "BTCUSDT"
    BTCDOM_SYMBOL: str = "BTCDOMUSDT"  # BTC Dominance
    BTC_SMA_PERIOD: int = 50
    REGIME_ENABLED: bool = True
    REGIME_HARD_VETO: bool = False  # If False, just reduces score instead of blocking
    
    # === SPOT-LED TRAP FILTER (NEW - Research: 2.3x higher expectancy) ===
    PERP_SPOT_RATIO_MAX: float = 4.0  # VETO if perp/spot volume > 4x
    SPOT_LED_BONUS: float = 15.0  # Bonus for spot-led moves
    
    # === CVD ENGINE (NEW - True CVD with slope and acceleration) ===
    CVD_LOOKBACK: int = 50  # Bars for CVD calculation
    CVD_SLOPE_PERIOD: int = 15  # Bars for slope calculation
    CVD_DIVERGENCE_THRESHOLD: float = 0.3  # CVD vs Price divergence
    
    # === OFI Z-SCORE (NEW - Order Flow Imbalance) ===
    OFI_LOOKBACK: int = 60  # Rolling window for z-score
    OFI_ZSCORE_THRESHOLD: float = 1.5  # Entry trigger threshold
    
    # === LIQUIDATION PROXIMITY (NEW - Research: ≤25bp = highest priority) ===
    LIQ_CLUSTER_NEAR_BP: float = 25.0  # Basis points for full bonus
    LIQ_CLUSTER_FAR_BP: float = 60.0   # Bonus decays to zero here
    LIQ_PROXIMITY_MAX_BONUS: float = 15.0
    
    # === ATR STOP-LOSS (Swing Trading: 2.5x for 4-12hr holds) ===
    ATR_STOP_MULTIPLIER: float = 2.5  # Slightly wider for swing trades
    ATR_PERIOD: int = 14
    
    # === GOD CANDLE DETECTION (Swing Trading: adjusted for 15m) ===
    GOD_CANDLE_PCT_5M: float = 8.0   # >8% move in 15 minutes (swing threshold)
    GOD_CANDLE_VOL_MULT: float = 6.0  # Volume > 6x baseline (adjusted for 15m)
    GOD_CANDLE_BONUS: float = 25.0
    
    # === FUNDING FILTERS (UPGRADED) ===
    FUNDING_FLIP_BONUS: float = 7.0    # Research: 58% PPV
    FUNDING_CROWDED_THRESHOLD: float = 0.001  # 0.1% = HARD VETO
    FUNDING_NEGATIVE_THRESHOLD: float = -0.0001
    
    # === RVOL THRESHOLDS (NEW - Research specifications) ===
    RVOL_QUIET: float = 1.0   # Calm before storm
    RVOL_ALERT: float = 3.0   # Ignition starting  
    RVOL_EXTREME: float = 5.0 # God candle territory
    RVOL_LOOKBACK: int = 20   # Baseline calculation period
    
    # === BB WIDTH PERCENTILE GATE (Research: very low = SLEEP) ===
    BB_WIDTH_PERCENTILE_GATE: float = 5.0  # More lenient than research's 10th
    BB_WIDTH_LOOKBACK: int = 192  # 192 bars on 15m = 48 hours lookback
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    KC_PERIOD: int = 20
    KC_ATR_MULT: float = 1.5
    BB_SLEEP_VETO_ENABLED: bool = False  # Disable sleeping veto
    
    # === SPREAD VETO (NEW - Research: >1.5% = REJECT) ===
    SPREAD_MAX_PCT: float = 1.5
    SPREAD_VETO_ENABLED: bool = True
    
    # === WATERFALL GATES (NEW - Sequential filtering) ===
    GATE_A_MIN_VOLUME_24H: float = 500_000   # Viability gate
    GATE_B_BB_SQUEEZE_REQUIRED: bool = True  # Compression gate
    GATE_C_IGNITION_PCT: float = 3.0         # 3% move = early warning
    GATE_D_OI_DELTA_MIN: float = 0.0         # OI must be increasing
    
    # === MICROCAP SETTINGS ===
    MICROCAP_MAX_VOLUME_24H: float = 50_000_000
    MICROCAP_MIN_VOLUME_24H: float = 100_000
    MICROCAP_BONUS_SCORE: float = 20.0
    
    # === VPIN SETTINGS ===
    VPIN_WINDOW: int = 50
    VPIN_TOXICITY_THRESHOLD: float = 0.35
    VPIN_EXHAUSTION_THRESHOLD: float = 0.20
    
    # === HURST SETTINGS (Swing Trading: relaxed thresholds) ===
    HURST_WINDOW: int = 100
    HURST_TRENDING_THRESHOLD: float = 0.51  # Slightly relaxed for swing
    HURST_MEAN_REVERT_THRESHOLD: float = 0.47
    
    # === SIGNAL THRESHOLDS ===
    MIN_DISPLAY_SCORE: float = 40.0
    MIN_IGNITION_SCORE: float = 70.0
    EXTREME_SCORE_THRESHOLD: float = 90.0
    HIGH_SCORE_THRESHOLD: float = 80.0
    
    # === CHANDELIER EXIT (Swing Trading: 3x ATR trailing) ===
    CHANDELIER_ATR_MULT: float = 3.0
    CHANDELIER_LOOKBACK: int = 48  # 48 bars on 15m = 12 hours lookback
    
    # === SHORT SQUEEZE TP ENGINE (Research: 7-Metric System) ===
    # Research: "SHORT SQUEEZE TPs are determined by liquidation cluster locations,
    #            funding rate normalization, and OI exhaustion, NOT by fixed sigma levels"
    
    # TP Position Allocation (Research-backed ladder)
    TP1_POSITION_PCT: float = 0.40  # 40% at first HVN
    TP2_POSITION_PCT: float = 0.35  # 35% at second HVN / OI peak
    TP3_POSITION_PCT: float = 0.15  # 15% on exhaustion signals
    RUNNER_POSITION_PCT: float = 0.10  # 10% trailed with Chandelier
    
    # Metric 1: Liquidation Cluster Thresholds
    # Research: "78% of successful squeezes reverse near liquidation walls"
    LIQ_CLUSTER_TP_BUFFER_PCT: float = 0.5  # Place TP just inside cluster
    
    # Metric 2: Funding Rate Normalization Thresholds
    # Research: "Exit when FR returns to ~0% from extreme negative"
    FR_TP1_THRESHOLD: float = -0.0030  # Still has squeeze fuel (-0.30%)
    FR_TP2_THRESHOLD: float = -0.0015  # Fuel depleting (-0.15%)
    FR_TP3_THRESHOLD: float = -0.0010  # Near empty (-0.10%)
    FR_NORMALIZED_THRESHOLD: float = -0.0005  # Squeeze exhausted
    
    # Metric 3: OI Exhaustion Pattern
    # Research: "Exit when OI peaks then declines 10%+"
    OI_PEAK_DECLINE_TP2_PCT: float = -5.0  # OI down 5% from peak = TP2 signal
    OI_PEAK_DECLINE_TP3_PCT: float = -10.0  # OI down 10% from peak = TP3 signal
    OI_HEALTHY_GROWTH_PCT: float = 2.0  # OI growing = HOLD
    
    # Metric 4: VPVR High Volume Nodes
    # Research: "TPs at HVNs (price magnets), through LVNs (fast)"
    # Note: HVN detection requires VPVR data - we estimate from volume profile
    VPVR_HVN_VOL_THRESHOLD: float = 1.5  # Volume > 1.5x avg = HVN
    VPVR_LVN_VOL_THRESHOLD: float = 0.5  # Volume < 0.5x avg = LVN
    
    # Metric 5: RSI Exhaustion (Multi-timeframe)
    # Research: "Exit zone when RSI > 85-90 on 4H/12H"
    RSI_PERIOD: int = 14
    RSI_TP2_THRESHOLD: float = 75.0  # Momentum extended
    RSI_TP3_THRESHOLD: float = 85.0  # Exhaustion zone
    RSI_PANIC_EXIT_THRESHOLD: float = 90.0  # CLIMAX - immediate exit
    
    # Metric 6: CVD Momentum Decay
    # Research: "Exit when CVD growth rate slows (second derivative turns negative)"
    CVD_ACCEL_TP2_THRESHOLD: float = 0.0  # Acceleration turning negative = TP2
    CVD_SLOPE_TP3_THRESHOLD: float = -0.001  # Slope negative = TP3
    
    # Metric 7: Volume Climax Pattern (Reversal Guard)
    # Research: "Volume > 3x avg + Shooting Star = PANIC SELL"
    VOLUME_CLIMAX_MULT: float = 3.0  # Volume > 3x 14-period average
    VOLUME_CLIMAX_RSI_THRESHOLD: float = 90.0  # RSI > 90
    SHOOTING_STAR_WICK_RATIO: float = 2.0  # Upper wick > 2x body
    
    # Fallback Percentage TPs (SWING TRADING: 4-12hr hold targets)
    # Research: "If no VPVR/liquidation data, use these as fallbacks"
    FALLBACK_TP1_PCT: float = 0.08  # +8% from entry (4-6hr target)
    FALLBACK_TP2_PCT: float = 0.15  # +15% from entry (6-8hr target)
    FALLBACK_TP3_PCT: float = 0.25  # +25% from entry (8-12hr target)
    FALLBACK_TP4_PCT: float = 0.40  # +40% (runner for extended moves)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # === MAGNETIC TP SYSTEM (Research: Price Magnets & Confluence) ===
    # ═══════════════════════════════════════════════════════════════════════════
    
    # --- Fibonacci Extension Targets (Research: "key percentages...identify S/R") ---
    FIB_EXT_1: float = 1.272    # Conservative first target
    FIB_EXT_2: float = 1.618    # Golden ratio - primary magnet
    FIB_EXT_3: float = 2.0      # Full extension
    FIB_EXT_4: float = 2.618    # Extended target for runners
    FIB_ENABLED: bool = True    # Use Fibonacci extensions
    
    # --- ATR-Dynamic TP Scaling (Research: "volatility-based...adapts to noise") ---
    ATR_TP1_MULT: float = 3.0   # TP1 at 3× ATR from entry
    ATR_TP2_MULT: float = 5.0   # TP2 at 5× ATR from entry
    ATR_TP3_MULT: float = 8.0   # TP3 at 8× ATR from entry
    ATR_TP4_MULT: float = 12.0  # TP4 at 12× ATR (runner)
    ATR_TP_ENABLED: bool = True # Use ATR-scaled TPs
    
    # --- Swing Structure Detection (Research: "major swing high/low") ---
    SWING_LOOKBACK: int = 50    # Bars to look back for swing highs
    SWING_STRENGTH: int = 5     # Bars on each side to confirm swing
    SWING_ENABLED: bool = True  # Use swing highs as targets
    
    # --- VWAP Deviation Bands (Research: "AVWAP bands...outer bands") ---
    VWAP_LOOKBACK: int = 100    # Bars for VWAP calculation
    VWAP_BAND_1: float = 1.0    # 1σ band - first target
    VWAP_BAND_2: float = 2.0    # 2σ band - overextension target
    VWAP_BAND_3: float = 3.0    # 3σ band - extreme target
    VWAP_ENABLED: bool = True   # Use VWAP bands as targets
    
    # --- Value Area (Research: "Volume Profile...High/Low Volume Nodes, POC, Value Area") ---
    VALUE_AREA_PCT: float = 0.70   # 70% of volume = Value Area (standard)
    POC_ENABLED: bool = True       # Use Point of Control
    VAH_VAL_ENABLED: bool = True   # Use Value Area High/Low
    
    # --- Confluence Scoring (Research: "confluence of multiple...indicators") ---
    CONFLUENCE_MIN_SIGNALS: int = 2     # Minimum signals for "magnetic" TP
    CONFLUENCE_STRONG_SIGNALS: int = 3  # Signals for "super magnetic" TP
    CONFLUENCE_CLUSTER_PCT: float = 0.015  # 1.5% = same zone (cluster TPs together)
    
    # === RISK MANAGEMENT (Swing Trading: 4-12hr holds) ===
    ACCOUNT_BALANCE: float = 10000.0
    MAX_RISK_PER_TRADE_PCT: float = 0.015  # 1.5% risk per swing trade
    MAX_CONCURRENT_TRADES: int = 5  # More concurrent positions for swings
    
    # === DEBUG ===
    DEBUG_MODE: bool = True
    DB_PATH: str = "god_of_gods_v5_trades.db"


CONFIG = GodOfGodsConfigV5()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

log_level = logging.DEBUG if CONFIG.DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
log = logging.getLogger("GodOfGodsV5")
console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES - v5.0 Enhanced
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRegime(Enum):
    RISK_ON = "RISK_ON"      # BTC > 50SMA AND BTC.D Down
    RISK_OFF = "RISK_OFF"    # BTC < 50SMA OR BTC.D Up
    SLEEP = "SLEEP"          # Low volatility regime
    UNKNOWN = "UNKNOWN"


class Archetype(Enum):
    """Signal archetypes - descriptive scenario names"""
    FRESH_BREAK = "FRESH BREAK"       # Fresh breakout with strong volume + CVD/OFI
    COIL_SQUEEZE = "COIL SQUEEZE"     # BB/KC volatility squeeze coiling
    HYPE_PUMP = "HYPE PUMP"           # Leverage-driven pump with OI surge
    EXTREME_VOL = "EXTREME VOL"       # Extreme volume explosion (5x+ RVOL)
    DIP_BOUNCE = "DIP BOUNCE"         # Bounce recovery after deep dip
    MICRO_MOVE = "MICRO MOVE"         # Fast micro-cap move
    TREND_CONT = "TREND CONT"         # Trend continuation pattern
    RANGE_COIL = "RANGE COIL"         # Range coil pre-breakout
    NONE = "NONE"


class GateStatus(Enum):
    """Waterfall gate status"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


@dataclass
class BTCRegimeResult:
    """BTC Regime Filter result - NEW in v5.0"""
    regime: MarketRegime
    btc_price: float
    btc_sma50: float
    btc_above_sma: bool
    btcdom_trend: str  # "UP", "DOWN", "FLAT"
    btcdom_change_1h: float
    should_trade: bool
    reason: str


@dataclass
class SpotLedResult:
    """Spot-Led vs Perp-Led analysis - NEW in v5.0"""
    perp_volume: float
    spot_volume: float
    ratio: float
    is_spot_led: bool
    is_perp_led_trap: bool  # VETO condition
    bonus: float


@dataclass
class CVDResult:
    """True CVD analysis - NEW in v5.0"""
    cvd_value: float
    cvd_slope: float
    cvd_acceleration: float  # 2nd derivative
    price_slope: float
    has_bullish_divergence: bool  # Price down, CVD up
    has_bearish_divergence: bool  # Price up, CVD down
    cvd_zscore: float


@dataclass
class OFIResult:
    """Order Flow Imbalance - NEW in v5.0"""
    ofi_raw: float
    ofi_zscore: float
    is_buy_pressure: bool
    is_sell_pressure: bool
    strength: str  # "STRONG", "MODERATE", "WEAK"


@dataclass
class GodCandleResult:
    """God Candle Detection - NEW in v5.0"""
    is_god_candle: bool
    price_change_pct: float
    volume_multiple: float
    timeframe: str
    direction: str  # "LONG" or "SHORT"


@dataclass
class RVOLResult:
    """Relative Volume analysis - NEW in v5.0"""
    rvol: float
    baseline_volume: float
    current_volume: float
    regime: str  # "QUIET", "ALERT", "EXTREME"
    is_squeeze_setup: bool


@dataclass
class WaterfallGates:
    """Waterfall Gate System - NEW in v5.0"""
    gate_a_viability: GateStatus
    gate_a_reason: str
    gate_b_compression: GateStatus
    gate_b_reason: str
    gate_c_ignition: GateStatus
    gate_c_reason: str
    gate_d_confirmation: GateStatus
    gate_d_reason: str
    all_passed: bool


@dataclass
class VPINResult:
    value: float
    buy_volume: float
    sell_volume: float
    imbalance: float
    is_toxic: bool
    is_exhausted: bool


@dataclass
class HurstResult:
    value: float
    regime: str
    is_valid_for_squeeze: bool
    confidence: float


@dataclass 
class BBKCSqueezeResult:
    is_in_squeeze: bool
    squeeze_duration: int
    bb_upper: float
    bb_lower: float
    bb_width: float
    bb_width_percentile: float  # NEW in v5.0
    kc_upper: float
    kc_lower: float
    momentum: float
    squeeze_fired: bool
    squeeze_direction: str
    is_sleeping: bool  # NEW: BB width < 10th percentile


@dataclass
class EnhancedTPLadder:
    """
    Enhanced TP Ladder with 7-Metric Short Squeeze System - v5.0
    
    Research: "SHORT SQUEEZE TPs are fundamentally different from normal breakout TPs—
    they're determined by liquidation cluster locations, funding rate normalization,
    and OI exhaustion, NOT by fixed sigma levels or percentage targets."
    """
    # Core prices (required)
    entry_price: float
    stop_loss: float  # 2.3x ATR stop (Research: cuts whipsaws by 37%)
    chandelier_stop: float  # Trailing stop: High - 3x ATR
    
    # Structural TP targets (NOT percentage-based) - required
    tp1_price: float  # First HVN above entry
    tp2_price: float  # Second HVN or liquidation cluster
    tp3_price: float  # Major liquidation cluster / historical resistance
    tp4_price: float  # Runner target (for extreme squeezes)
    
    # Risk metrics (required)
    risk_reward_ratio: float
    atr: float
    atr_stop_distance: float
    market_tier: str
    
    # Position allocation per TP level (Research ladder) - with defaults
    tp1_position_pct: float = 0.40  # 40% at TP1
    tp2_position_pct: float = 0.35  # 35% at TP2
    tp3_position_pct: float = 0.15  # 15% at TP3
    runner_position_pct: float = 0.10  # 10% runner with trail
    
    # Structural context (for real-time TP decisions)
    vpvr_hvn_levels: List[float] = field(default_factory=list)  # High Volume Nodes
    liquidation_clusters: List[float] = field(default_factory=list)  # From heatmap
    
    # Exit condition triggers (to be evaluated in real-time)
    tp1_funding_threshold: float = -0.0030  # Still has squeeze fuel
    tp2_funding_threshold: float = -0.0015  # Fuel depleting
    tp3_funding_threshold: float = -0.0010  # Near empty
    
    # Tracking state (for position management)
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    initial_funding_rate: float = 0.0  # FR at entry
    peak_oi: float = 0.0  # Track OI peak for exhaustion
    highest_high_since_entry: float = 0.0  # For chandelier exit
    
    # Classification
    is_structural_tp: bool = True  # True if TPs based on structure, False if fallback %
    tp_source: str = "STRUCTURAL"  # "STRUCTURAL", "VPVR", "LIQUIDATION", "FALLBACK"
    
    # === MAGNETIC TP SYSTEM (NEW) ===
    magnetic_levels: List[Dict] = field(default_factory=list)  # All magnetic price levels
    confluence_score: Dict[str, int] = field(default_factory=dict)  # Signals per TP level
    fib_levels: List[float] = field(default_factory=list)  # Fibonacci extensions
    swing_highs: List[float] = field(default_factory=list)  # Prior resistance pivots
    vwap_bands: Dict[str, float] = field(default_factory=dict)  # VWAP + deviation bands
    value_area: Dict[str, float] = field(default_factory=dict)  # POC, VAH, VAL


@dataclass
class MagneticLevel:
    """
    A price level with magnetic properties - where price is drawn to.
    Research: "Price is drawn to areas of high liquidity"
    """
    price: float
    source: str  # "FIB", "SWING", "HVN", "VWAP", "VAH", "LIQ", "ATR"
    strength: int  # 1-5 based on confluence
    confluence_sources: List[str] = field(default_factory=list)  # All sources at this level
    distance_pct: float = 0.0  # Distance from entry as %
    is_super_magnetic: bool = False  # 3+ signals = super magnetic


@dataclass
class ValueAreaResult:
    """
    Volume Profile Value Area - where 70% of volume traded.
    Research: "Point of Control (POC), Value Area High/Low (VAH/VAL)"
    """
    poc: float  # Point of Control - highest volume price
    vah: float  # Value Area High
    val: float  # Value Area Low
    total_volume: float
    volume_profile: Dict[float, float] = field(default_factory=dict)


@dataclass
class VWAPBandsResult:
    """
    Anchored VWAP with standard deviation bands.
    Research: "Target VWAP on reversions or outer bands on overextensions"
    """
    vwap: float
    upper_1: float  # +1σ
    upper_2: float  # +2σ
    upper_3: float  # +3σ
    lower_1: float  # -1σ
    lower_2: float  # -2σ
    lower_3: float  # -3σ
    std_dev: float


@dataclass
class FibonacciResult:
    """
    Fibonacci extension levels from swing low to entry.
    Research: "key percentages (23.6%, 38.2%, 50%, 61.8%)...identify S/R"
    """
    swing_low: float
    swing_high: float
    ext_1272: float  # 127.2% extension
    ext_1618: float  # 161.8% extension (golden ratio)
    ext_200: float   # 200% extension
    ext_2618: float  # 261.8% extension
    retracement_382: float  # 38.2% retracement (support)
    retracement_618: float  # 61.8% retracement (support)


@dataclass
class SwingHighResult:
    """
    Prior swing highs as resistance/target levels.
    Research: "SL placed beyond major swing high/low"
    """
    swing_highs: List[float]  # Sorted ascending
    nearest_above: float  # First swing high above entry
    major_resistance: float  # Strongest swing high
    swing_count: int


@dataclass
class SqueezeSignalV5:
    """Enhanced signal with all v5.0 features + 7-Metric TP System"""
    symbol: str
    exchange: str  # NEW: binance or bybit
    timestamp: datetime
    price: float
    ignition_score: float
    
    # Core analysis
    vpin: VPINResult
    hurst: HurstResult
    bb_kc_squeeze: BBKCSqueezeResult
    tp_ladder: EnhancedTPLadder
    
    # NEW v5.0 analysis
    btc_regime: BTCRegimeResult
    spot_led: SpotLedResult
    cvd: CVDResult
    ofi: OFIResult
    god_candle: GodCandleResult
    rvol: RVOLResult
    gates: WaterfallGates
    archetype: Archetype
    
    # Market data
    funding_rate: float
    funding_flip_detected: bool  # NEW
    open_interest: float
    oi_change_pct: float
    oi_acceleration: float  # NEW: 2nd derivative
    volume_24h: float
    long_short_ratio: float
    spread_pct: float  # NEW
    
    # Flags (required - no defaults)
    is_microcap: bool
    
    # NEW: 7-Metric TP System Data (with defaults)
    rsi_values: Dict[str, float] = field(default_factory=dict)  # Multi-timeframe RSI
    volume_climax_detected: bool = False  # Reversal Guard signal
    volume_climax_reasons: List[str] = field(default_factory=list)  # Why climax triggered
    tp_source: str = "FALLBACK"  # "STRUCTURAL", "VPVR", "LIQUIDATION", "FALLBACK"
    
    # Flags (with defaults)
    is_veto: bool = False  # Hard reject flag
    veto_reasons: List[str] = field(default_factory=list)
    conviction: str = "LOW"
    should_enter: bool = False
    reasons: List[str] = field(default_factory=list)  # Why signal fired


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: BTC REGIME FILTER (NEW - Research: 62% win rate when RISK_ON)
# ═══════════════════════════════════════════════════════════════════════════════

class BTCRegimeEngine:
    """
    Global regime filter based on BTC and BTC Dominance.
    Research shows:
    - RISK_ON (BTC > 50SMA + BTC.D Down): 62% win rate
    - RISK_OFF: 27% win rate
    - Slashes max drawdown from -23.4% to -11.6%
    """
    
    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return 0.0
        return float(np.mean(prices[-period:]))
    
    @staticmethod
    def calculate_trend(prices: np.ndarray, lookback: int = 20) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        if len(prices) < lookback:
            return "FLAT", 0.0
        
        recent = prices[-lookback:]
        change = (recent[-1] - recent[0]) / recent[0] * 100 if recent[0] > 0 else 0
        
        if change > 0.5:
            return "UP", change
        elif change < -0.5:
            return "DOWN", change
        else:
            return "FLAT", change
    
    @classmethod
    def analyze(
        cls,
        btc_prices: np.ndarray,
        btcdom_prices: Optional[np.ndarray] = None,
        sma_period: int = CONFIG.BTC_SMA_PERIOD
    ) -> BTCRegimeResult:
        """Determine market regime based on BTC and dominance"""
        
        if len(btc_prices) < sma_period:
            return BTCRegimeResult(
                regime=MarketRegime.UNKNOWN,
                btc_price=btc_prices[-1] if len(btc_prices) > 0 else 0,
                btc_sma50=0,
                btc_above_sma=False,
                btcdom_trend="UNKNOWN",
                btcdom_change_1h=0,
                should_trade=True,  # Allow trading if no data
                reason="Insufficient BTC data"
            )
        
        btc_price = float(btc_prices[-1])
        btc_sma = cls.calculate_sma(btc_prices, sma_period)
        btc_above_sma = btc_price > btc_sma
        
        # BTC Dominance trend
        btcdom_trend = "UNKNOWN"
        btcdom_change = 0.0
        if btcdom_prices is not None and len(btcdom_prices) >= 20:
            btcdom_trend, btcdom_change = cls.calculate_trend(btcdom_prices, 20)
        
        # Determine regime
        # RISK_ON: BTC above 50SMA AND BTC.D trending down
        if btc_above_sma and btcdom_trend == "DOWN":
            regime = MarketRegime.RISK_ON
            should_trade = True
            reason = f"BTC ${btc_price:.0f} > SMA50 ${btc_sma:.0f}, BTC.D falling {btcdom_change:.2f}%"
        elif not btc_above_sma:
            regime = MarketRegime.RISK_OFF
            should_trade = False
            reason = f"BTC ${btc_price:.0f} < SMA50 ${btc_sma:.0f}"
        elif btcdom_trend == "UP":
            regime = MarketRegime.RISK_OFF
            should_trade = False
            reason = f"BTC.D rising {btcdom_change:.2f}% - flight to safety"
        else:
            regime = MarketRegime.RISK_ON
            should_trade = True
            reason = f"BTC ${btc_price:.0f} > SMA50, BTC.D {btcdom_trend}"
        
        return BTCRegimeResult(
            regime=regime,
            btc_price=btc_price,
            btc_sma50=btc_sma,
            btc_above_sma=btc_above_sma,
            btcdom_trend=btcdom_trend,
            btcdom_change_1h=btcdom_change,
            should_trade=should_trade,
            reason=reason
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: SPOT-LED vs PERP-LED TRAP FILTER (NEW - Research: 2.3x expectancy)
# ═══════════════════════════════════════════════════════════════════════════════

class SpotLedEngine:
    """
    Detects spot-led vs perp-led moves.
    Research shows:
    - Spot-led moves have 2.3x higher expectancy
    - Perp/Spot ratio > 4x is a VETO (fake pump, likely to mean-revert)
    """
    
    @staticmethod
    def analyze(
        perp_volume: float,
        spot_volume: float,
        max_ratio: float = CONFIG.PERP_SPOT_RATIO_MAX
    ) -> SpotLedResult:
        """Analyze if move is spot-led or perp-led"""
        
        # Avoid division by zero
        if spot_volume < 1:
            spot_volume = 1
        
        ratio = perp_volume / spot_volume
        
        is_spot_led = ratio < 1.5  # Spot volume dominates
        is_perp_trap = ratio > max_ratio  # VETO condition
        
        # Calculate bonus for spot-led moves
        bonus = 0.0
        if is_spot_led:
            bonus = CONFIG.SPOT_LED_BONUS
        elif ratio < 2.0:
            bonus = CONFIG.SPOT_LED_BONUS * 0.5
        
        return SpotLedResult(
            perp_volume=perp_volume,
            spot_volume=spot_volume,
            ratio=ratio,
            is_spot_led=is_spot_led,
            is_perp_led_trap=is_perp_trap,
            bonus=bonus
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: TRUE CVD ENGINE (NEW - Research: Slope and Acceleration)
# ═══════════════════════════════════════════════════════════════════════════════

class CVDEngine:
    """
    True Cumulative Volume Delta calculation.
    Research shows:
    - CVD Divergence (Price up, CVD down) = EXIT signal
    - CVD Acceleration (2nd derivative) for early detection
    - CVD Slope for momentum confirmation
    """
    
    @staticmethod
    def calculate(
        taker_buy_vol: np.ndarray,
        taker_sell_vol: np.ndarray,
        closes: np.ndarray,
        lookback: int = CONFIG.CVD_LOOKBACK,
        slope_period: int = CONFIG.CVD_SLOPE_PERIOD
    ) -> CVDResult:
        """Calculate CVD with slope and acceleration"""
        
        min_len = min(len(taker_buy_vol), len(taker_sell_vol), len(closes))
        if min_len < slope_period + 5:
            return CVDResult(
                cvd_value=0, cvd_slope=0, cvd_acceleration=0,
                price_slope=0, has_bullish_divergence=False,
                has_bearish_divergence=False, cvd_zscore=0
            )
        
        # Calculate raw CVD (cumulative buy - sell)
        buy = taker_buy_vol[-lookback:] if len(taker_buy_vol) >= lookback else taker_buy_vol
        sell = taker_sell_vol[-lookback:] if len(taker_sell_vol) >= lookback else taker_sell_vol
        
        delta = buy - sell  # Per-bar delta
        cvd = np.cumsum(delta)  # Cumulative
        cvd_value = float(cvd[-1])
        
        # CVD Slope (first derivative) - normalized
        cvd_recent = cvd[-slope_period:]
        x = np.arange(len(cvd_recent))
        try:
            cvd_slope_raw, _, _, _, _ = linregress(x, cvd_recent)
            # Normalize by average volume
            avg_vol = np.mean(buy + sell) + 1e-10
            cvd_slope = cvd_slope_raw / avg_vol
        except:
            cvd_slope = 0.0
        
        # CVD Acceleration (second derivative)
        if len(cvd) >= slope_period * 2:
            cvd_slope_prev = 0.0
            cvd_prev = cvd[-slope_period*2:-slope_period]
            x_prev = np.arange(len(cvd_prev))
            try:
                cvd_slope_prev, _, _, _, _ = linregress(x_prev, cvd_prev)
                cvd_slope_prev = cvd_slope_prev / avg_vol
            except:
                pass
            cvd_acceleration = cvd_slope - cvd_slope_prev
        else:
            cvd_acceleration = 0.0
        
        # Price slope for divergence detection
        prices = closes[-slope_period:]
        x_p = np.arange(len(prices))
        try:
            price_slope_raw, _, _, _, _ = linregress(x_p, prices)
            price_slope = price_slope_raw / (np.mean(prices) + 1e-10)
        except:
            price_slope = 0.0
        
        # Detect divergences
        # Bullish: Price falling but CVD rising (buying into weakness)
        has_bullish_div = price_slope < -0.001 and cvd_slope > 0.001
        # Bearish: Price rising but CVD falling (distribution)
        has_bearish_div = price_slope > 0.001 and cvd_slope < -0.001
        
        # CVD Z-score for strength measurement
        if len(cvd) >= 60:
            cvd_mean = np.mean(cvd[-60:])
            cvd_std = np.std(cvd[-60:]) + 1e-10
            cvd_zscore = (cvd_value - cvd_mean) / cvd_std
        else:
            cvd_zscore = 0.0
        
        return CVDResult(
            cvd_value=cvd_value,
            cvd_slope=cvd_slope,
            cvd_acceleration=cvd_acceleration,
            price_slope=price_slope,
            has_bullish_divergence=has_bullish_div,
            has_bearish_divergence=has_bearish_div,
            cvd_zscore=cvd_zscore
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: OFI Z-SCORE ENGINE (NEW - Research: Order Flow Imbalance)
# ═══════════════════════════════════════════════════════════════════════════════

class OFIEngine:
    """
    Order Flow Imbalance calculation.
    Research shows:
    - OFI Z-score > 1.5 = significant buy pressure
    - OFI Z-score < -1.5 = significant sell pressure
    """
    
    @staticmethod
    def calculate(
        taker_buy_vol: np.ndarray,
        taker_sell_vol: np.ndarray,
        lookback: int = CONFIG.OFI_LOOKBACK
    ) -> OFIResult:
        """Calculate OFI with rolling z-score normalization"""
        
        min_len = min(len(taker_buy_vol), len(taker_sell_vol))
        if min_len < 20:
            return OFIResult(
                ofi_raw=0, ofi_zscore=0,
                is_buy_pressure=False, is_sell_pressure=False,
                strength="WEAK"
            )
        
        # Raw OFI: difference between taker buy and sell
        buy = taker_buy_vol[-lookback:] if len(taker_buy_vol) >= lookback else taker_buy_vol
        sell = taker_sell_vol[-lookback:] if len(taker_sell_vol) >= lookback else taker_sell_vol
        
        ofi_per_bar = buy - sell
        
        # Current OFI (sum of recent bars)
        ofi_window = min(5, len(ofi_per_bar))
        ofi_raw = float(np.sum(ofi_per_bar[-ofi_window:]))
        
        # Z-score normalization
        ofi_mean = np.mean(ofi_per_bar)
        ofi_std = np.std(ofi_per_bar) + 1e-10
        ofi_zscore = (ofi_raw / ofi_window - ofi_mean) / ofi_std
        
        # Determine pressure
        threshold = CONFIG.OFI_ZSCORE_THRESHOLD
        is_buy = ofi_zscore > threshold
        is_sell = ofi_zscore < -threshold
        
        if abs(ofi_zscore) > 2.5:
            strength = "STRONG"
        elif abs(ofi_zscore) > threshold:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return OFIResult(
            ofi_raw=ofi_raw,
            ofi_zscore=ofi_zscore,
            is_buy_pressure=is_buy,
            is_sell_pressure=is_sell,
            strength=strength
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: GOD CANDLE DETECTOR (NEW - Research: >12% in 5m, Vol >10x)
# ═══════════════════════════════════════════════════════════════════════════════

class GodCandleEngine:
    """
    Detects "God Candle" events.
    Research definition:
    - 5-minute candle closing >12% higher
    - Volume > 10x baseline
    """
    
    @staticmethod
    def calculate(
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        lookback_baseline: int = 20
    ) -> GodCandleResult:
        """Detect god candle in recent price action"""
        
        if len(closes) < lookback_baseline + 1:
            return GodCandleResult(
                is_god_candle=False, price_change_pct=0,
                volume_multiple=0, timeframe="1m", direction="NONE"
            )
        
        # Check last 5 bars (simulate 5m on 1m chart)
        for i in range(1, min(6, len(closes))):
            idx = -i
            if idx - lookback_baseline < -len(closes):
                continue
            
            # Price change
            open_price = opens[idx]
            close_price = closes[idx]
            if open_price <= 0:
                continue
            
            pct_change = (close_price - open_price) / open_price * 100
            
            # Volume multiple
            baseline_vol = np.mean(volumes[idx-lookback_baseline:idx]) + 1e-10
            vol_mult = volumes[idx] / baseline_vol
            
            # Check god candle criteria
            is_god = (
                abs(pct_change) >= CONFIG.GOD_CANDLE_PCT_5M and
                vol_mult >= CONFIG.GOD_CANDLE_VOL_MULT
            )
            
            if is_god:
                direction = "LONG" if pct_change > 0 else "SHORT"
                return GodCandleResult(
                    is_god_candle=True,
                    price_change_pct=pct_change,
                    volume_multiple=vol_mult,
                    timeframe="1m",
                    direction=direction
                )
        
        # No god candle found
        return GodCandleResult(
            is_god_candle=False,
            price_change_pct=0,
            volume_multiple=0,
            timeframe="1m",
            direction="NONE"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: RVOL ENGINE (NEW - Research: Quiet/Alert/Extreme thresholds)
# ═══════════════════════════════════════════════════════════════════════════════

class RVOLEngine:
    """
    Relative Volume calculation.
    Research thresholds:
    - RVOL < 1.0: Quiet (calm before storm, squeeze building)
    - RVOL > 3.0: Alert (ignition starting)
    - RVOL > 5.0: Extreme (god candle territory)
    """
    
    @staticmethod
    def calculate(
        volumes: np.ndarray,
        lookback: int = CONFIG.RVOL_LOOKBACK
    ) -> RVOLResult:
        """Calculate relative volume vs baseline"""
        
        if len(volumes) < lookback + 1:
            return RVOLResult(
                rvol=1.0, baseline_volume=0, current_volume=0,
                regime="UNKNOWN", is_squeeze_setup=False
            )
        
        baseline = np.mean(volumes[-lookback-1:-1]) + 1e-10
        current = float(volumes[-1])
        rvol = current / baseline
        
        # Determine regime
        if rvol < CONFIG.RVOL_QUIET:
            regime = "QUIET"
            is_squeeze = True  # Low volume = potential squeeze building
        elif rvol >= CONFIG.RVOL_EXTREME:
            regime = "EXTREME"
            is_squeeze = False
        elif rvol >= CONFIG.RVOL_ALERT:
            regime = "ALERT"
            is_squeeze = False
        else:
            regime = "NORMAL"
            is_squeeze = False
        
        return RVOLResult(
            rvol=rvol,
            baseline_volume=baseline,
            current_volume=current,
            regime=regime,
            is_squeeze_setup=is_squeeze
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: WATERFALL GATES (NEW - Research: Sequential Filtering)
# ═══════════════════════════════════════════════════════════════════════════════

class WaterfallGateEngine:
    """
    Waterfall Gate System from research.
    A coin must pass Gate A to reach Gate B, and so on.
    
    GATE A (Viability): Liquidity > $500k, Spread < 1.5%
    GATE B (Compression): BB squeeze, RVOL < 1.0
    GATE C (Ignition): Price Delta > 3%, RVOL > 5.0, Taker ratio > 1.5
    GATE D (Confirmation): OI Delta > 0, Spot CVD > Perp CVD
    """
    
    @staticmethod
    def evaluate(
        volume_24h: float,
        spread_pct: float,
        is_in_squeeze: bool,
        rvol: float,
        price_change_5m_pct: float,
        taker_buy_ratio: float,
        oi_delta: float,
        spot_led_result: SpotLedResult
    ) -> WaterfallGates:
        """Evaluate all waterfall gates"""
        
        # GATE A: Viability
        gate_a_passed = True
        gate_a_reasons = []
        
        if volume_24h < CONFIG.GATE_A_MIN_VOLUME_24H:
            gate_a_passed = False
            gate_a_reasons.append(f"Vol ${volume_24h/1000:.0f}K < $500K")
        
        if spread_pct > CONFIG.SPREAD_MAX_PCT:
            gate_a_passed = False
            gate_a_reasons.append(f"Spread {spread_pct:.2f}% > 1.5%")
        
        gate_a_status = GateStatus.PASSED if gate_a_passed else GateStatus.FAILED
        gate_a_reason = ", ".join(gate_a_reasons) if gate_a_reasons else "Liquidity OK"
        
        # GATE B: Compression (only if Gate A passed)
        gate_b_passed = False
        gate_b_reason = "Gate A not passed"
        
        if gate_a_passed:
            if is_in_squeeze and rvol < CONFIG.RVOL_QUIET:
                gate_b_passed = True
                gate_b_reason = f"Squeeze active, RVOL {rvol:.2f} (quiet)"
            elif is_in_squeeze:
                gate_b_passed = True
                gate_b_reason = f"Squeeze active, RVOL {rvol:.2f}"
            else:
                gate_b_reason = "Not in BB-KC squeeze"
        
        gate_b_status = GateStatus.PASSED if gate_b_passed else GateStatus.FAILED
        
        # GATE C: Ignition (only if Gate B passed)
        gate_c_passed = False
        gate_c_reason = "Gate B not passed"
        
        if gate_b_passed:
            ignition_checks = []
            
            if abs(price_change_5m_pct) >= CONFIG.GATE_C_IGNITION_PCT:
                ignition_checks.append(f"Δ{price_change_5m_pct:.1f}%")
            
            if rvol >= CONFIG.RVOL_EXTREME:
                ignition_checks.append(f"RVOL {rvol:.1f}x")
            
            if taker_buy_ratio > 1.5:
                ignition_checks.append(f"Taker {taker_buy_ratio:.1f}x")
            
            if len(ignition_checks) >= 2:
                gate_c_passed = True
                gate_c_reason = " + ".join(ignition_checks)
            else:
                gate_c_reason = f"Need 2+ ignition signals, got {len(ignition_checks)}"
        
        gate_c_status = GateStatus.PASSED if gate_c_passed else GateStatus.FAILED
        
        # GATE D: Confirmation (only if Gate C passed)
        gate_d_passed = False
        gate_d_reason = "Gate C not passed"
        
        if gate_c_passed:
            confirm_checks = []
            
            if oi_delta > CONFIG.GATE_D_OI_DELTA_MIN:
                confirm_checks.append(f"OI +{oi_delta:.1f}%")
            
            if spot_led_result.is_spot_led:
                confirm_checks.append("Spot-led")
            
            if len(confirm_checks) >= 1:
                gate_d_passed = True
                gate_d_reason = " + ".join(confirm_checks)
            else:
                gate_d_reason = "No OI growth or spot confirmation"
        
        gate_d_status = GateStatus.PASSED if gate_d_passed else GateStatus.FAILED
        
        return WaterfallGates(
            gate_a_viability=gate_a_status,
            gate_a_reason=gate_a_reason,
            gate_b_compression=gate_b_status,
            gate_b_reason=gate_b_reason,
            gate_c_ignition=gate_c_status,
            gate_c_reason=gate_c_reason,
            gate_d_confirmation=gate_d_status,
            gate_d_reason=gate_d_reason,
            all_passed=gate_d_passed
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: ARCHETYPE CLASSIFIER (NEW - Research: Pattern Classification)
# ═══════════════════════════════════════════════════════════════════════════════

class ArchetypeEngine:
    """
    Classifies signals into archetypes based on research patterns.
    """
    
    @staticmethod
    def classify(
        rvol: RVOLResult,
        cvd: CVDResult,
        bb_kc: BBKCSqueezeResult,
        ofi: OFIResult,
        price_change_pct: float,
        oi_change_pct: float
    ) -> Tuple[Archetype, str]:
        """Classify the setup into an archetype with reason"""
        
        scores = {}
        reasons = {}
        
        vol_regime = rvol.regime
        
        # FRESH BREAK: Fresh breakout with strong volume
        if vol_regime in ["ALERT", "EXTREME"] and price_change_pct > 2:
            score = 0
            r = []
            if cvd.cvd_slope > 0.001:
                score += 1
                r.append("CVD supports")
            if ofi.is_buy_pressure:
                score += 1
                r.append("OFI buy pressure")
            if rvol.rvol > 3:
                score += 1
                r.append(f"RVOL {rvol.rvol:.1f}x")
            if score >= 2:
                scores["FRESH BREAK"] = score
                reasons["FRESH BREAK"] = ", ".join(r)
        
        # COIL SQUEEZE: Volatility squeeze (coiling)
        if bb_kc.is_in_squeeze and vol_regime == "QUIET":
            score = 2
            r = [f"Squeeze {bb_kc.squeeze_duration}bars"]
            if abs(price_change_pct) < 1:
                score += 1
                r.append("Price coiling")
            if oi_change_pct > 0:
                score += 1
                r.append("OI building")
            scores["COIL SQUEEZE"] = score
            reasons["COIL SQUEEZE"] = ", ".join(r)
        
        # HYPE PUMP: Leverage-driven pump with OI surge
        if vol_regime in ["ALERT", "EXTREME"] and oi_change_pct > 2:
            score = 0
            r = []
            if abs(price_change_pct) > 3:
                score += 2
                r.append(f"Strong move {price_change_pct:.1f}%")
            if cvd.cvd_acceleration > 0:
                score += 1
                r.append("CVD accelerating")
            if score >= 2:
                scores["HYPE PUMP"] = score
                reasons["HYPE PUMP"] = ", ".join(r)
        
        # EXTREME VOL: Extreme volume explosion
        if vol_regime == "EXTREME" and abs(price_change_pct) > 5:
            score = 3
            r = [f"Extreme RVOL {rvol.rvol:.1f}x", f"Move {price_change_pct:.1f}%"]
            if cvd.cvd_slope > 0.002:
                score += 1
                r.append("Strong CVD")
            scores["EXTREME VOL"] = score
            reasons["EXTREME VOL"] = ", ".join(r)
        
        # DIP BOUNCE: Bounce after deep dip
        if price_change_pct < -3 and ofi.is_buy_pressure:
            score = 2
            r = [f"Dip {price_change_pct:.1f}%", "Bid support"]
            if cvd.cvd_slope > 0:
                score += 1
                r.append("CVD recovering")
            scores["DIP BOUNCE"] = score
            reasons["DIP BOUNCE"] = ", ".join(r)
        
        # RANGE COIL: Range coil pre-breakout
        if bb_kc.is_in_squeeze and vol_regime in ["NORMAL", "QUIET"]:
            if abs(price_change_pct) < 1 and abs(oi_change_pct) < 1:
                score = 2
                r = ["Range coil", f"Squeeze {bb_kc.squeeze_duration}bars"]
                scores["RANGE COIL"] = score
                reasons["RANGE COIL"] = ", ".join(r)
        
        # Select best archetype
        if scores:
            best = max(scores, key=scores.get)
            return Archetype(best), reasons[best]
        
        return Archetype.NONE, "No pattern match"


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: ENHANCED BB-KC SQUEEZE (v5.0 - with percentile gate)
# ═══════════════════════════════════════════════════════════════════════════════

class BBKCSqueezeEngineV5:
    """Enhanced BB-KC with BB Width Percentile gate"""
    
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr_list.append(max(hl, hc, lc))
        
        if len(tr_list) < period:
            return np.mean(tr_list) if tr_list else 0.0
        
        return float(np.mean(tr_list[-period:]))
    
    @classmethod
    def calculate(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> BBKCSqueezeResult:
        """Calculate BB-KC squeeze with percentile gate"""
        
        bb_period = CONFIG.BB_PERIOD
        bb_std = CONFIG.BB_STD
        kc_period = CONFIG.KC_PERIOD
        kc_atr_mult = CONFIG.KC_ATR_MULT
        
        if len(closes) < max(bb_period, kc_period, CONFIG.BB_WIDTH_LOOKBACK) + 14:
            return BBKCSqueezeResult(
                is_in_squeeze=False, squeeze_duration=0,
                bb_upper=0, bb_lower=0, bb_width=0, bb_width_percentile=50,
                kc_upper=0, kc_lower=0, momentum=0,
                squeeze_fired=False, squeeze_direction="", is_sleeping=False
            )
        
        # Calculate Bollinger Bands
        bb_sma = np.mean(closes[-bb_period:])
        bb_std_val = np.std(closes[-bb_period:])
        bb_upper = bb_sma + (bb_std * bb_std_val)
        bb_lower = bb_sma - (bb_std * bb_std_val)
        bb_width = (bb_upper - bb_lower) / bb_sma if bb_sma > 0 else 0
        
        # Calculate BB Width Percentile (NEW)
        bb_widths = []
        lookback = min(CONFIG.BB_WIDTH_LOOKBACK, len(closes) - bb_period)
        for i in range(lookback):
            idx = -(i + 1)
            if abs(idx) + bb_period > len(closes):
                break
            hist_sma = np.mean(closes[idx-bb_period:idx])
            hist_std = np.std(closes[idx-bb_period:idx])
            hist_width = (2 * bb_std * hist_std) / hist_sma if hist_sma > 0 else 0
            bb_widths.append(hist_width)
        
        if bb_widths:
            bb_width_percentile = float(np.sum(np.array(bb_widths) < bb_width) / len(bb_widths) * 100)
        else:
            bb_width_percentile = 50.0
        
        # Determine if sleeping (low volatility regime)
        is_sleeping = bb_width_percentile < CONFIG.BB_WIDTH_PERCENTILE_GATE
        
        # Calculate Keltner Channels
        kc_ema = np.mean(closes[-kc_period:])
        atr = cls.calculate_atr(highs, lows, closes, 14)
        kc_upper = kc_ema + (kc_atr_mult * atr)
        kc_lower = kc_ema - (kc_atr_mult * atr)
        
        # Detect squeeze state
        current_squeeze = (bb_lower > kc_lower) and (bb_upper < kc_upper)
        
        # Count squeeze duration
        squeeze_duration = 0
        if current_squeeze:
            squeeze_duration = 1
            for i in range(2, min(len(closes) - bb_period, 50)):
                idx = -i
                hist_bb_sma = np.mean(closes[idx-bb_period:idx])
                hist_bb_std = np.std(closes[idx-bb_period:idx])
                hist_bb_upper = hist_bb_sma + (bb_std * hist_bb_std)
                hist_bb_lower = hist_bb_sma - (bb_std * hist_bb_std)
                
                hist_kc_ema = np.mean(closes[idx-kc_period:idx])
                hist_atr = cls.calculate_atr(highs[:idx], lows[:idx], closes[:idx], 14)
                hist_kc_upper = hist_kc_ema + (kc_atr_mult * hist_atr)
                hist_kc_lower = hist_kc_ema - (kc_atr_mult * hist_atr)
                
                if (hist_bb_lower > hist_kc_lower) and (hist_bb_upper < hist_kc_upper):
                    squeeze_duration += 1
                else:
                    break
        
        # Momentum
        momentum = closes[-1] - bb_sma
        
        # Squeeze fired detection
        squeeze_fired = False
        squeeze_direction = ""
        if not current_squeeze and squeeze_duration == 0:
            # Check if squeeze just released
            if len(closes) >= bb_period + 2:
                prev_squeeze = False
                # Quick check of previous bar
                prev_bb_sma = np.mean(closes[-bb_period-1:-1])
                prev_bb_std = np.std(closes[-bb_period-1:-1])
                prev_bb_upper = prev_bb_sma + (bb_std * prev_bb_std)
                prev_bb_lower = prev_bb_sma - (bb_std * prev_bb_std)
                
                prev_kc_ema = np.mean(closes[-kc_period-1:-1])
                prev_atr = cls.calculate_atr(highs[:-1], lows[:-1], closes[:-1], 14)
                prev_kc_upper = prev_kc_ema + (kc_atr_mult * prev_atr)
                prev_kc_lower = prev_kc_ema - (kc_atr_mult * prev_atr)
                
                prev_squeeze = (prev_bb_lower > prev_kc_lower) and (prev_bb_upper < prev_kc_upper)
                
                if prev_squeeze:
                    squeeze_fired = True
                    squeeze_direction = "LONG" if momentum > 0 else "SHORT"
        
        return BBKCSqueezeResult(
            is_in_squeeze=current_squeeze,
            squeeze_duration=squeeze_duration,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_width=bb_width,
            bb_width_percentile=bb_width_percentile,
            kc_upper=kc_upper,
            kc_lower=kc_lower,
            momentum=momentum,
            squeeze_fired=squeeze_fired,
            squeeze_direction=squeeze_direction,
            is_sleeping=is_sleeping
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: RSI ENGINE (NEW - Required for TP Exhaustion Signals)
# ═══════════════════════════════════════════════════════════════════════════════

class RSIEngine:
    """
    RSI calculation for exhaustion detection.
    Research: "Exit zone when RSI > 85-90 on 4H/12H"
    """
    
    @staticmethod
    def calculate(closes: np.ndarray, period: int = CONFIG.RSI_PERIOD) -> float:
        """Calculate RSI value"""
        if len(closes) < period + 1:
            return 50.0  # Neutral default
        
        # Calculate price changes
        deltas = np.diff(closes[-(period + 1):])
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gain/loss
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_multi_timeframe(
        closes_1m: np.ndarray,
        closes_5m: Optional[np.ndarray] = None,
        closes_1h: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate RSI across multiple timeframes"""
        result = {
            "1m": RSIEngine.calculate(closes_1m),
            "5m": 50.0,
            "1h": 50.0,
            "4h": 50.0,  # Estimated from 1h
            "12h": 50.0,  # Estimated from 1h
        }
        
        if closes_5m is not None and len(closes_5m) >= 15:
            result["5m"] = RSIEngine.calculate(closes_5m)
        
        if closes_1h is not None and len(closes_1h) >= 15:
            result["1h"] = RSIEngine.calculate(closes_1h)
            # Estimate higher timeframes from 1h data
            # 4h = last 4 hours of 1h data aggregated
            if len(closes_1h) >= 60:  # Need at least 60 1h candles for 4h estimate
                result["4h"] = RSIEngine.calculate(closes_1h)  # Approximation
                result["12h"] = RSIEngine.calculate(closes_1h)  # Approximation
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: SHORT SQUEEZE TP ENGINE (Research: 7-Metric System)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TPDecision:
    """Result of TP decision evaluation"""
    should_take: bool
    tp_level: str  # "TP1", "TP2", "TP3", "PANIC", "RUNNER", "HOLD"
    position_to_sell_pct: float
    reasons: List[str]
    new_stop_loss: Optional[float] = None  # Updated SL after TP
    is_panic_exit: bool = False


@dataclass
class VolumeClimaxResult:
    """Volume climax pattern detection (Reversal Guard)"""
    is_climax: bool
    volume_multiple: float
    is_shooting_star: bool
    rsi_extreme: bool
    reasons: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: MAGNETIC TP ENGINE (Research: "Price Magnets & Confluence")
# ═══════════════════════════════════════════════════════════════════════════════

class MagneticTPEngine:
    """
    Magnetic TP Engine - Identifies where price is DRAWN to.
    
    Research Insights:
    - "Price is drawn to areas of high liquidity"
    - "Gamma walls can suppress volatility and 'pin' the price"
    - "78% of successful squeezes reverse near liquidation walls"
    - "Target HVNs (price magnets), fast through LVNs"
    - "Use confluence of multiple, complementary indicators"
    
    Magnetic Sources:
    1. Fibonacci Extensions (1.272, 1.618, 2.0, 2.618)
    2. Swing Highs (prior resistance)
    3. Volume Profile HVNs (high volume nodes)
    4. VWAP Deviation Bands (+1σ, +2σ, +3σ)
    5. Value Area (POC, VAH)
    6. ATR-Scaled Levels (volatility-adaptive)
    7. Liquidation Clusters (if available)
    """
    
    @classmethod
    def calculate_fibonacci_extensions(
        cls,
        closes: np.ndarray,
        entry_price: float,
        lookback: int = 50
    ) -> FibonacciResult:
        """
        Calculate Fibonacci extension levels from swing low to entry.
        
        Research: "Fibonacci Retracement identifies potential S/R zones
        based on key percentages (23.6%, 38.2%, 50%, 61.8%)"
        """
        if len(closes) < lookback:
            lookback = len(closes)
        
        recent = closes[-lookback:]
        swing_low = float(np.min(recent))
        swing_high = float(np.max(recent))
        
        # Use entry price as high if it's above swing high
        if entry_price > swing_high:
            swing_high = entry_price
        
        # Calculate the range
        price_range = swing_high - swing_low
        
        # Extension levels (project above swing high)
        ext_1272 = swing_high + (price_range * 0.272)
        ext_1618 = swing_high + (price_range * 0.618)
        ext_200 = swing_high + (price_range * 1.0)
        ext_2618 = swing_high + (price_range * 1.618)
        
        # Retracement levels (support zones)
        retracement_382 = swing_high - (price_range * 0.382)
        retracement_618 = swing_high - (price_range * 0.618)
        
        return FibonacciResult(
            swing_low=swing_low,
            swing_high=swing_high,
            ext_1272=ext_1272,
            ext_1618=ext_1618,
            ext_200=ext_200,
            ext_2618=ext_2618,
            retracement_382=retracement_382,
            retracement_618=retracement_618
        )
    
    @classmethod
    def detect_swing_highs(
        cls,
        highs: np.ndarray,
        entry_price: float,
        lookback: int = 50,
        strength: int = 5
    ) -> SwingHighResult:
        """
        Detect prior swing highs as resistance/magnetic levels.
        
        Research: "SL placed beyond major swing high/low"
        A swing high has lower highs on both sides for 'strength' bars.
        """
        if len(highs) < lookback:
            lookback = len(highs)
        
        recent = highs[-lookback:]
        swing_highs = []
        
        # Find swing highs (local maxima)
        for i in range(strength, len(recent) - strength):
            is_swing = True
            center = recent[i]
            
            # Check bars on left
            for j in range(1, strength + 1):
                if recent[i - j] >= center:
                    is_swing = False
                    break
            
            # Check bars on right
            if is_swing:
                for j in range(1, strength + 1):
                    if recent[i + j] >= center:
                        is_swing = False
                        break
            
            if is_swing:
                swing_highs.append(float(center))
        
        # Sort and filter to unique levels (cluster within 1%)
        swing_highs = sorted(set(swing_highs))
        
        # Cluster nearby swing highs
        clustered = []
        for sh in swing_highs:
            if not clustered or (sh - clustered[-1]) / clustered[-1] > 0.01:
                clustered.append(sh)
            else:
                # Keep the higher one
                clustered[-1] = max(clustered[-1], sh)
        
        # Find swings above entry
        above_entry = [s for s in clustered if s > entry_price]
        nearest = above_entry[0] if above_entry else entry_price * 1.10
        major = max(clustered) if clustered else entry_price * 1.15
        
        return SwingHighResult(
            swing_highs=clustered,
            nearest_above=nearest,
            major_resistance=major,
            swing_count=len(clustered)
        )
    
    @classmethod
    def calculate_vwap_bands(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        lookback: int = 100
    ) -> VWAPBandsResult:
        """
        Calculate anchored VWAP with standard deviation bands.
        
        Research: "Volume-weighted average price from user-defined point.
        Target VWAP on reversions or outer bands on overextensions."
        """
        if len(closes) < lookback:
            lookback = len(closes)
        
        # Typical price = (H + L + C) / 3
        tp = (highs[-lookback:] + lows[-lookback:] + closes[-lookback:]) / 3
        vol = volumes[-lookback:]
        
        # VWAP = sum(TP * Volume) / sum(Volume)
        cumulative_tp_vol = np.cumsum(tp * vol)
        cumulative_vol = np.cumsum(vol)
        
        # Avoid division by zero
        cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
        
        vwap_series = cumulative_tp_vol / cumulative_vol
        vwap = float(vwap_series[-1])
        
        # Standard deviation from VWAP
        squared_diff = (tp - vwap_series) ** 2
        variance = np.cumsum(squared_diff * vol) / cumulative_vol
        std_series = np.sqrt(variance)
        std_dev = float(std_series[-1])
        
        # Prevent zero std dev
        if std_dev == 0:
            std_dev = vwap * 0.02  # 2% default
        
        return VWAPBandsResult(
            vwap=vwap,
            upper_1=vwap + std_dev * CONFIG.VWAP_BAND_1,
            upper_2=vwap + std_dev * CONFIG.VWAP_BAND_2,
            upper_3=vwap + std_dev * CONFIG.VWAP_BAND_3,
            lower_1=vwap - std_dev * CONFIG.VWAP_BAND_1,
            lower_2=vwap - std_dev * CONFIG.VWAP_BAND_2,
            lower_3=vwap - std_dev * CONFIG.VWAP_BAND_3,
            std_dev=std_dev
        )
    
    @classmethod
    def calculate_value_area(
        cls,
        closes: np.ndarray,
        volumes: np.ndarray,
        lookback: int = 100,
        value_area_pct: float = 0.70
    ) -> ValueAreaResult:
        """
        Calculate Volume Profile Value Area (POC, VAH, VAL).
        
        Research: "Volume Profile displays volume at specific price levels.
        Key components are High/Low Volume Nodes (HVN/LVN), 
        Point of Control (POC), and Value Area (VAH/VAL)."
        """
        if len(closes) < lookback:
            lookback = len(closes)
        
        closes = closes[-lookback:]
        volumes = volumes[-lookback:]
        
        # Create price buckets
        price_min = float(np.min(closes))
        price_max = float(np.max(closes))
        n_buckets = 30
        bucket_size = (price_max - price_min) / n_buckets if price_max > price_min else 1
        
        # Accumulate volume in each bucket
        volume_profile: Dict[float, float] = {}
        for price, vol in zip(closes, volumes):
            bucket_idx = int((price - price_min) / bucket_size) if bucket_size > 0 else 0
            bucket_idx = min(bucket_idx, n_buckets - 1)
            bucket_price = price_min + (bucket_idx + 0.5) * bucket_size
            volume_profile[bucket_price] = volume_profile.get(bucket_price, 0) + vol
        
        if not volume_profile:
            return ValueAreaResult(
                poc=closes[-1], vah=closes[-1] * 1.02, val=closes[-1] * 0.98,
                total_volume=0, volume_profile={}
            )
        
        # Find POC (highest volume price)
        poc = max(volume_profile.keys(), key=lambda k: volume_profile[k])
        total_vol = sum(volume_profile.values())
        
        # Calculate Value Area (70% of volume centered on POC)
        target_vol = total_vol * value_area_pct
        
        # Sort buckets by price
        sorted_buckets = sorted(volume_profile.keys())
        poc_idx = sorted_buckets.index(poc)
        
        # Expand outward from POC until we capture target volume
        val_idx = poc_idx
        vah_idx = poc_idx
        current_vol = volume_profile[poc]
        
        while current_vol < target_vol and (val_idx > 0 or vah_idx < len(sorted_buckets) - 1):
            # Check which direction to expand
            vol_below = volume_profile.get(sorted_buckets[val_idx - 1], 0) if val_idx > 0 else 0
            vol_above = volume_profile.get(sorted_buckets[vah_idx + 1], 0) if vah_idx < len(sorted_buckets) - 1 else 0
            
            if vol_below >= vol_above and val_idx > 0:
                val_idx -= 1
                current_vol += volume_profile[sorted_buckets[val_idx]]
            elif vah_idx < len(sorted_buckets) - 1:
                vah_idx += 1
                current_vol += volume_profile[sorted_buckets[vah_idx]]
            else:
                break
        
        val = sorted_buckets[val_idx]
        vah = sorted_buckets[vah_idx]
        
        return ValueAreaResult(
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_vol,
            volume_profile=volume_profile
        )
    
    @classmethod
    def calculate_atr_levels(
        cls,
        entry_price: float,
        atr: float
    ) -> Dict[str, float]:
        """
        Calculate ATR-scaled TP levels.
        
        Research: "Volatility-based stops...widening the stop in volatile conditions."
        """
        return {
            'atr_tp1': entry_price + (atr * CONFIG.ATR_TP1_MULT),
            'atr_tp2': entry_price + (atr * CONFIG.ATR_TP2_MULT),
            'atr_tp3': entry_price + (atr * CONFIG.ATR_TP3_MULT),
            'atr_tp4': entry_price + (atr * CONFIG.ATR_TP4_MULT),
        }
    
    @classmethod
    def cluster_magnetic_levels(
        cls,
        all_levels: List[Dict],
        entry_price: float,
        cluster_pct: float = 0.015
    ) -> List[MagneticLevel]:
        """
        Cluster nearby price levels and calculate confluence.
        
        Research: "Use confluence of multiple, complementary indicators"
        
        Levels within cluster_pct% are considered the same zone.
        """
        if not all_levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(all_levels, key=lambda x: x['price'])
        
        # Only keep levels above entry
        above_entry = [l for l in sorted_levels if l['price'] > entry_price]
        
        if not above_entry:
            return []
        
        # Cluster nearby levels
        clusters: List[List[Dict]] = []
        current_cluster = [above_entry[0]]
        
        for level in above_entry[1:]:
            last_price = current_cluster[-1]['price']
            if (level['price'] - last_price) / last_price <= cluster_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        
        clusters.append(current_cluster)
        
        # Create MagneticLevel for each cluster
        magnetic_levels = []
        for cluster in clusters:
            # Average price in cluster
            avg_price = np.mean([l['price'] for l in cluster])
            sources = [l['source'] for l in cluster]
            unique_sources = list(set(sources))
            strength = len(unique_sources)
            
            distance_pct = (avg_price - entry_price) / entry_price * 100
            
            magnetic = MagneticLevel(
                price=float(avg_price),
                source=unique_sources[0] if len(unique_sources) == 1 else "CONFLUENCE",
                strength=strength,
                confluence_sources=unique_sources,
                distance_pct=distance_pct,
                is_super_magnetic=strength >= CONFIG.CONFLUENCE_STRONG_SIGNALS
            )
            magnetic_levels.append(magnetic)
        
        # Sort by distance from entry
        magnetic_levels.sort(key=lambda x: x.price)
        
        return magnetic_levels
    
    @classmethod
    def generate_all_magnetic_levels(
        cls,
        entry_price: float,
        atr: float,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        hvn_levels: Optional[List[float]] = None,
        liquidation_clusters: Optional[List[float]] = None
    ) -> Tuple[List[MagneticLevel], Dict]:
        """
        Generate all magnetic price levels from multiple sources.
        Returns clustered levels ranked by confluence strength.
        """
        all_levels = []
        magnetic_data = {}
        
        # 1. Fibonacci Extensions
        if CONFIG.FIB_ENABLED:
            fib = cls.calculate_fibonacci_extensions(closes, entry_price, CONFIG.SWING_LOOKBACK)
            magnetic_data['fibonacci'] = fib
            
            all_levels.append({'price': fib.ext_1272, 'source': 'FIB_1.272'})
            all_levels.append({'price': fib.ext_1618, 'source': 'FIB_1.618'})
            all_levels.append({'price': fib.ext_200, 'source': 'FIB_2.0'})
            all_levels.append({'price': fib.ext_2618, 'source': 'FIB_2.618'})
        
        # 2. Swing Highs
        if CONFIG.SWING_ENABLED:
            swings = cls.detect_swing_highs(highs, entry_price, CONFIG.SWING_LOOKBACK, CONFIG.SWING_STRENGTH)
            magnetic_data['swings'] = swings
            
            for sh in swings.swing_highs:
                if sh > entry_price:
                    all_levels.append({'price': sh, 'source': 'SWING'})
        
        # 3. VWAP Bands
        if CONFIG.VWAP_ENABLED:
            vwap = cls.calculate_vwap_bands(highs, lows, closes, volumes, CONFIG.VWAP_LOOKBACK)
            magnetic_data['vwap'] = vwap
            
            if vwap.upper_1 > entry_price:
                all_levels.append({'price': vwap.upper_1, 'source': 'VWAP_1σ'})
            if vwap.upper_2 > entry_price:
                all_levels.append({'price': vwap.upper_2, 'source': 'VWAP_2σ'})
            if vwap.upper_3 > entry_price:
                all_levels.append({'price': vwap.upper_3, 'source': 'VWAP_3σ'})
        
        # 4. Value Area
        if CONFIG.VAH_VAL_ENABLED:
            va = cls.calculate_value_area(closes, volumes)
            magnetic_data['value_area'] = va
            
            if va.vah > entry_price:
                all_levels.append({'price': va.vah, 'source': 'VAH'})
            if CONFIG.POC_ENABLED and va.poc > entry_price:
                all_levels.append({'price': va.poc, 'source': 'POC'})
        
        # 5. ATR-Scaled Levels
        if CONFIG.ATR_TP_ENABLED:
            atr_levels = cls.calculate_atr_levels(entry_price, atr)
            magnetic_data['atr_levels'] = atr_levels
            
            for key, price in atr_levels.items():
                all_levels.append({'price': price, 'source': key.upper()})
        
        # 6. HVN Levels (from existing VPVR estimation)
        if hvn_levels:
            for hvn in hvn_levels:
                if hvn > entry_price:
                    all_levels.append({'price': hvn, 'source': 'HVN'})
        
        # 7. Liquidation Clusters
        if liquidation_clusters:
            for liq in liquidation_clusters:
                if liq > entry_price:
                    all_levels.append({'price': liq, 'source': 'LIQ'})
        
        # Cluster and rank by confluence
        magnetic_levels = cls.cluster_magnetic_levels(
            all_levels, 
            entry_price, 
            CONFIG.CONFLUENCE_CLUSTER_PCT
        )
        
        return magnetic_levels, magnetic_data
    
    @classmethod
    def select_magnetic_tps(
        cls,
        magnetic_levels: List[MagneticLevel],
        entry_price: float,
        fallback_tp1_pct: float,
        fallback_tp2_pct: float,
        fallback_tp3_pct: float,
        fallback_tp4_pct: float
    ) -> Tuple[float, float, float, float, str]:
        """
        Select TP1-TP4 from magnetic levels, preferring high-confluence zones.
        
        Research: "Confluence of multiple, complementary indicators to validate exit points"
        """
        # Filter to only levels with minimum confluence
        valid_levels = [l for l in magnetic_levels if l.strength >= 1]
        
        # Sort by distance (closest first)
        valid_levels.sort(key=lambda x: x.price)
        
        # Prefer super-magnetic levels (3+ signals)
        super_magnetic = [l for l in valid_levels if l.is_super_magnetic]
        
        # Default fallbacks
        tp1 = entry_price * (1 + fallback_tp1_pct)
        tp2 = entry_price * (1 + fallback_tp2_pct)
        tp3 = entry_price * (1 + fallback_tp3_pct)
        tp4 = entry_price * (1 + fallback_tp4_pct)
        tp_source = "FALLBACK"
        
        if len(valid_levels) >= 1:
            # TP1: First magnetic level (or super-magnetic if close)
            first_super = super_magnetic[0] if super_magnetic else None
            if first_super and first_super.distance_pct < 15:  # Within 15% = use super magnetic
                tp1 = first_super.price
            else:
                tp1 = valid_levels[0].price
            tp_source = "MAGNETIC"
        
        if len(valid_levels) >= 2:
            # TP2: Second level or next super-magnetic
            remaining = [l for l in valid_levels if l.price > tp1]
            super_remaining = [l for l in remaining if l.is_super_magnetic]
            
            if super_remaining and super_remaining[0].distance_pct < 25:
                tp2 = super_remaining[0].price
            elif remaining:
                tp2 = remaining[0].price
        
        if len(valid_levels) >= 3:
            # TP3: Third level
            remaining = [l for l in valid_levels if l.price > tp2]
            if remaining:
                # Prefer highest confluence remaining
                remaining.sort(key=lambda x: (-x.strength, x.price))
                tp3 = remaining[0].price
        
        if len(valid_levels) >= 4:
            # TP4: Extended target (runner)
            remaining = [l for l in valid_levels if l.price > tp3]
            if remaining:
                tp4 = remaining[0].price
        
        return tp1, tp2, tp3, tp4, tp_source


class ShortSqueezeTPEngine:
    """
    Evidence-based TP engine for short squeezes.
    
    Research: "Uses 7 metrics instead of fixed percentages.
    The key insight is that SHORT SQUEEZE TPs are fundamentally different 
    from normal breakout TPs—they're determined by liquidation cluster locations,
    funding rate normalization, and OI exhaustion."
    
    The 7 Metrics:
    1. Liquidation Heatmap - TPs placed just BELOW major liquidation clusters
    2. Funding Rate Normalization - Exit when FR returns to ~0% from extreme negative
    3. OI Exhaustion Pattern - Exit when OI peaks then declines 10%+
    4. VPVR High Volume Nodes - TPs at HVNs (price magnets)
    5. RSI Exhaustion - Exit zone when RSI > 85-90
    6. CVD Momentum Decay - Exit when CVD growth rate slows
    7. Volume Climax Pattern - Volume > 3x avg + shooting star = PANIC SELL
    """
    
    # Market tier squeeze magnitudes (from original research)
    SQUEEZE_MAGNITUDE = {
        'micro':  {'min': 0.15, 'typical': 0.35, 'extreme': 1.00},
        'small':  {'min': 0.12, 'typical': 0.25, 'extreme': 0.60},
        'medium': {'min': 0.08, 'typical': 0.18, 'extreme': 0.40},
        'large':  {'min': 0.05, 'typical': 0.12, 'extreme': 0.25},
    }
    
    @classmethod
    def get_market_tier(cls, volume_24h: float) -> str:
        """Determine market tier based on 24h volume"""
        if volume_24h < 10_000_000:
            return 'micro'
        elif volume_24h < 50_000_000:
            return 'small'
        elif volume_24h < 200_000_000:
            return 'medium'
        else:
            return 'large'
    
    @classmethod
    def estimate_hvn_levels(
        cls,
        closes: np.ndarray,
        volumes: np.ndarray,
        entry_price: float,
        lookback: int = 100
    ) -> List[float]:
        """
        Estimate High Volume Nodes from price/volume data.
        Research: "HVNs show price stability, strong S/R"
        
        This is an approximation - real VPVR requires tick data.
        """
        if len(closes) < lookback:
            return []
        
        closes = closes[-lookback:]
        volumes = volumes[-lookback:]
        
        # Create price buckets
        price_min = np.min(closes)
        price_max = np.max(closes)
        n_buckets = 20
        bucket_size = (price_max - price_min) / n_buckets if price_max > price_min else 1
        
        # Accumulate volume in each bucket
        volume_profile = {}
        for i, (price, vol) in enumerate(zip(closes, volumes)):
            bucket = int((price - price_min) / bucket_size) if bucket_size > 0 else 0
            bucket = min(bucket, n_buckets - 1)  # Clamp to valid range
            bucket_price = price_min + (bucket + 0.5) * bucket_size
            volume_profile[bucket_price] = volume_profile.get(bucket_price, 0) + vol
        
        if not volume_profile:
            return []
        
        # Find HVNs (buckets with volume > 1.5x average)
        avg_vol = np.mean(list(volume_profile.values()))
        hvn_threshold = avg_vol * CONFIG.VPVR_HVN_VOL_THRESHOLD
        
        hvns = [
            price for price, vol in volume_profile.items()
            if vol >= hvn_threshold and price > entry_price
        ]
        
        # Sort by price
        hvns.sort()
        
        return hvns
    
    @classmethod
    def detect_volume_climax(
        cls,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        rsi: float,
        lookback: int = 14
    ) -> VolumeClimaxResult:
        """
        Detect Volume Climax Pattern (Reversal Guard).
        
        Research: "The 'Reversal Guard' sells instantly if it detects:
        1. Volume Spike: Volume > 3x the 14-period average
        2. Stalling Price: Price makes new high but closes lower (Shooting Star) OR RSI > 90"
        """
        if len(volumes) < lookback + 1:
            return VolumeClimaxResult(
                is_climax=False, volume_multiple=0,
                is_shooting_star=False, rsi_extreme=False, reasons=[]
            )
        
        # Check volume spike
        avg_vol = np.mean(volumes[-lookback-1:-1])
        current_vol = volumes[-1]
        vol_mult = current_vol / avg_vol if avg_vol > 0 else 0
        is_vol_spike = vol_mult >= CONFIG.VOLUME_CLIMAX_MULT
        
        # Check for shooting star (long upper wick, small body)
        open_price = opens[-1]
        high = highs[-1]
        low = lows[-1]
        close = closes[-1]
        
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        is_shooting_star = (
            upper_wick > body * CONFIG.SHOOTING_STAR_WICK_RATIO and
            close < open_price and  # Red candle
            body > 0  # Has a body
        )
        
        # Check RSI extreme
        rsi_extreme = rsi >= CONFIG.VOLUME_CLIMAX_RSI_THRESHOLD
        
        # Build reasons
        reasons = []
        if is_vol_spike:
            reasons.append(f"Vol {vol_mult:.1f}x > 3x avg")
        if is_shooting_star:
            reasons.append("Shooting star candle")
        if rsi_extreme:
            reasons.append(f"RSI {rsi:.0f} > 90")
        
        # Climax requires volume spike + (shooting star OR RSI extreme)
        is_climax = is_vol_spike and (is_shooting_star or rsi_extreme)
        
        return VolumeClimaxResult(
            is_climax=is_climax,
            volume_multiple=vol_mult,
            is_shooting_star=is_shooting_star,
            rsi_extreme=rsi_extreme,
            reasons=reasons
        )
    
    @classmethod
    def calculate_structural_tps(
        cls,
        entry_price: float,
        atr: float,
        highest_high: float,
        volume_24h: float,
        closes: np.ndarray,
        volumes: np.ndarray,
        initial_funding_rate: float,
        vpvr_hvns: Optional[List[float]] = None,
        liquidation_clusters: Optional[List[float]] = None,
        conviction: str = "MEDIUM",
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> EnhancedTPLadder:
        """
        Calculate TP ladder based on MAGNETIC LEVELS with confluence scoring.
        
        Research: "TPs Are Structural, Not Percentage-Based"
        - Uses Fibonacci extensions, Swing highs, VWAP bands, HVNs, Value Area
        - Clusters nearby levels and ranks by confluence
        - Prefers "super magnetic" zones (3+ signals agree)
        """
        tier = cls.get_market_tier(volume_24h)
        magnitude = cls.SQUEEZE_MAGNITUDE[tier]
        
        # ATR Stop = 2.5x (Research: proven optimal for swing trades)
        atr_stop_distance = CONFIG.ATR_STOP_MULTIPLIER * atr
        stop_loss = entry_price - atr_stop_distance
        
        # Chandelier Exit (3x ATR below highest high)
        chandelier_stop = highest_high - (CONFIG.CHANDELIER_ATR_MULT * atr)
        
        # Get structural levels if not provided
        if vpvr_hvns is None or len(vpvr_hvns) == 0:
            vpvr_hvns = cls.estimate_hvn_levels(closes, volumes, entry_price)
        
        # === MAGNETIC TP SYSTEM ===
        # Generate highs/lows from closes if not provided
        if highs is None:
            highs = closes  # Approximation
        if lows is None:
            lows = closes  # Approximation
        
        # Generate all magnetic levels with confluence scoring
        magnetic_levels, magnetic_data = MagneticTPEngine.generate_all_magnetic_levels(
            entry_price=entry_price,
            atr=atr,
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            hvn_levels=vpvr_hvns,
            liquidation_clusters=liquidation_clusters
        )
        
        # Tier and conviction multipliers for fallbacks
        tier_mult = {'micro': 1.4, 'small': 1.2, 'medium': 1.0, 'large': 0.8}[tier]
        conviction_mult = {'EXTREME': 2.0, 'HIGH': 1.5, 'MEDIUM': 1.2, 'LOW': 1.0}.get(conviction, 1.0)
        total_mult = tier_mult * conviction_mult
        
        # Select TPs from magnetic levels
        tp1, tp2, tp3, tp4, tp_source = MagneticTPEngine.select_magnetic_tps(
            magnetic_levels=magnetic_levels,
            entry_price=entry_price,
            fallback_tp1_pct=CONFIG.FALLBACK_TP1_PCT * total_mult,
            fallback_tp2_pct=CONFIG.FALLBACK_TP2_PCT * total_mult,
            fallback_tp3_pct=CONFIG.FALLBACK_TP3_PCT * total_mult,
            fallback_tp4_pct=CONFIG.FALLBACK_TP4_PCT * total_mult
        )
        
        is_structural = tp_source != "FALLBACK"
        
        # R:R calculation
        risk = entry_price - stop_loss
        reward = tp2 - entry_price  # Using TP2 for R:R as it's the main target
        rr = reward / risk if risk > 0 else 1.0
        
        # Extract data for ladder storage
        fib_levels = []
        swing_highs = []
        vwap_bands = {}
        value_area = {}
        confluence_score = {}
        
        if 'fibonacci' in magnetic_data:
            fib = magnetic_data['fibonacci']
            fib_levels = [fib.ext_1272, fib.ext_1618, fib.ext_200, fib.ext_2618]
        
        if 'swings' in magnetic_data:
            swing_highs = magnetic_data['swings'].swing_highs
        
        if 'vwap' in magnetic_data:
            vwap = magnetic_data['vwap']
            vwap_bands = {
                'vwap': vwap.vwap,
                'upper_1': vwap.upper_1,
                'upper_2': vwap.upper_2,
                'upper_3': vwap.upper_3
            }
        
        if 'value_area' in magnetic_data:
            va = magnetic_data['value_area']
            value_area = {'poc': va.poc, 'vah': va.vah, 'val': va.val}
        
        # Build confluence score dict
        for level in magnetic_levels:
            key = f"${level.price:.4f}"
            confluence_score[key] = level.strength
        
        # Convert magnetic levels to dict for storage
        magnetic_levels_dict = [
            {
                'price': l.price,
                'source': l.source,
                'strength': l.strength,
                'confluence': l.confluence_sources,
                'distance_pct': l.distance_pct,
                'super_magnetic': l.is_super_magnetic
            }
            for l in magnetic_levels[:10]  # Top 10 levels
        ]
        
        return EnhancedTPLadder(
            entry_price=entry_price,
            stop_loss=stop_loss,
            chandelier_stop=chandelier_stop,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            tp4_price=tp4,
            tp1_position_pct=CONFIG.TP1_POSITION_PCT,
            tp2_position_pct=CONFIG.TP2_POSITION_PCT,
            tp3_position_pct=CONFIG.TP3_POSITION_PCT,
            runner_position_pct=CONFIG.RUNNER_POSITION_PCT,
            risk_reward_ratio=rr,
            atr=atr,
            atr_stop_distance=atr_stop_distance,
            market_tier=tier,
            vpvr_hvn_levels=vpvr_hvns or [],
            liquidation_clusters=liquidation_clusters or [],
            tp1_funding_threshold=CONFIG.FR_TP1_THRESHOLD,
            tp2_funding_threshold=CONFIG.FR_TP2_THRESHOLD,
            tp3_funding_threshold=CONFIG.FR_TP3_THRESHOLD,
            initial_funding_rate=initial_funding_rate,
            is_structural_tp=is_structural,
            tp_source=tp_source,
            # NEW: Magnetic TP data
            magnetic_levels=magnetic_levels_dict,
            confluence_score=confluence_score,
            fib_levels=fib_levels,
            swing_highs=swing_highs,
            vwap_bands=vwap_bands,
            value_area=value_area
        )
    
    @classmethod
    def evaluate_tp_decision(
        cls,
        tp_ladder: EnhancedTPLadder,
        current_price: float,
        current_funding_rate: float,
        oi_current: float,
        oi_change_from_peak_pct: float,
        rsi_values: Dict[str, float],
        cvd_result: CVDResult,
        volume_climax: VolumeClimaxResult
    ) -> TPDecision:
        """
        Real-time TP decision based on 7-metric system.
        
        Research EXIT_SIGNAL formula:
        EXIT_SIGNAL = (
            (RSI_12H > 85) OR
            (Funding_Rate > -0.10%) OR
            (OI_Change_From_Peak < -5%) OR
            (Volume_ZScore > 3.0 AND Candle_Has_Long_Upper_Wick)
        )
        """
        reasons = []
        
        # ==================== PANIC EXIT CHECK (Metric 7: Volume Climax) ====================
        # Research: "sells instantly if it detects a Climax Pattern"
        if volume_climax.is_climax:
            return TPDecision(
                should_take=True,
                tp_level="PANIC",
                position_to_sell_pct=1.0,  # Sell everything
                reasons=["REVERSAL GUARD: " + ", ".join(volume_climax.reasons)],
                is_panic_exit=True
            )
        
        # ==================== TP3 CHECK (Exhaustion Signals) ====================
        # Research: "TP3: 15% exit on exhaustion signals - Exit on ANY of these"
        if not tp_ladder.tp3_hit and current_price >= tp_ladder.tp3_price:
            tp3_signals = []
            
            # Metric 5: RSI Exhaustion
            rsi_4h = rsi_values.get("4h", 50)
            rsi_12h = rsi_values.get("12h", 50)
            if rsi_12h > CONFIG.RSI_TP3_THRESHOLD:
                tp3_signals.append(f"RSI 12H {rsi_12h:.0f} > 85")
            elif rsi_4h > CONFIG.RSI_TP3_THRESHOLD:
                tp3_signals.append(f"RSI 4H {rsi_4h:.0f} > 85")
            
            # Metric 2: Funding nearly normalized
            if current_funding_rate > CONFIG.FR_TP3_THRESHOLD:
                tp3_signals.append(f"FR {current_funding_rate*100:.2f}% normalizing")
            
            # Metric 3: OI Exhaustion
            if oi_change_from_peak_pct < CONFIG.OI_PEAK_DECLINE_TP3_PCT:
                tp3_signals.append(f"OI {oi_change_from_peak_pct:.1f}% from peak")
            
            # Metric 6: CVD turning negative
            if cvd_result.cvd_slope < CONFIG.CVD_SLOPE_TP3_THRESHOLD:
                tp3_signals.append(f"CVD slope negative")
            
            # Price at TP3 level counts as a signal too
            if current_price >= tp_ladder.tp3_price:
                tp3_signals.append(f"Price at TP3 ${tp_ladder.tp3_price:.4f}")
            
            # ANY exhaustion signal = TP3
            if len(tp3_signals) >= 1:
                return TPDecision(
                    should_take=True,
                    tp_level="TP3",
                    position_to_sell_pct=tp_ladder.tp3_position_pct,
                    reasons=tp3_signals,
                    new_stop_loss=tp_ladder.tp2_price  # Move SL to TP2
                )
        
        # ==================== TP2 CHECK (OI Exhaustion Beginning) ====================
        # Research: "TP2: 35% exit when OI exhaustion begins - 2 of 3 conditions"
        if not tp_ladder.tp2_hit and not tp_ladder.tp1_hit:
            # Can't hit TP2 before TP1
            pass
        elif not tp_ladder.tp2_hit and current_price >= tp_ladder.tp2_price:
            tp2_conditions_met = 0
            tp2_reasons = []
            
            # Condition 1: Price at TP2 level
            if current_price >= tp_ladder.tp2_price:
                tp2_conditions_met += 1
                tp2_reasons.append(f"Price at TP2 ${tp_ladder.tp2_price:.4f}")
            
            # Condition 2: Funding depleting OR OI peaked
            if current_funding_rate > CONFIG.FR_TP2_THRESHOLD:
                tp2_conditions_met += 1
                tp2_reasons.append(f"FR {current_funding_rate*100:.2f}% depleting")
            elif oi_change_from_peak_pct < CONFIG.OI_PEAK_DECLINE_TP2_PCT:
                tp2_conditions_met += 1
                tp2_reasons.append(f"OI peaked ({oi_change_from_peak_pct:.1f}% from peak)")
            
            # Condition 3: Momentum extended (RSI > 75)
            rsi_4h = rsi_values.get("4h", 50)
            if rsi_4h > CONFIG.RSI_TP2_THRESHOLD:
                tp2_conditions_met += 1
                tp2_reasons.append(f"RSI 4H {rsi_4h:.0f} extended")
            
            # Condition 4: CVD acceleration slowing
            if cvd_result.cvd_acceleration < CONFIG.CVD_ACCEL_TP2_THRESHOLD:
                tp2_conditions_met += 1
                tp2_reasons.append("CVD acceleration slowing")
            
            # Research: "2 of 3 conditions"
            if tp2_conditions_met >= 2:
                return TPDecision(
                    should_take=True,
                    tp_level="TP2",
                    position_to_sell_pct=tp_ladder.tp2_position_pct,
                    reasons=tp2_reasons,
                    new_stop_loss=tp_ladder.entry_price  # Move SL to breakeven
                )
        
        # ==================== TP1 CHECK (First HVN Hit) ====================
        # Research: "TP1: 40% exit when first HVN hit + funding still has fuel"
        if not tp_ladder.tp1_hit and current_price >= tp_ladder.tp1_price:
            tp1_conditions = []
            
            # Price at TP1 level
            tp1_conditions.append(f"Price at TP1 ${tp_ladder.tp1_price:.4f}")
            
            # Funding still has squeeze fuel
            if current_funding_rate < CONFIG.FR_TP1_THRESHOLD:
                tp1_conditions.append(f"FR {current_funding_rate*100:.2f}% still negative")
            
            # OI still building
            if oi_change_from_peak_pct > 0:
                tp1_conditions.append(f"OI still building +{oi_change_from_peak_pct:.1f}%")
            
            return TPDecision(
                should_take=True,
                tp_level="TP1",
                position_to_sell_pct=tp_ladder.tp1_position_pct,
                reasons=tp1_conditions,
                new_stop_loss=tp_ladder.entry_price  # Move SL to breakeven
            )
        
        # ==================== RUNNER MANAGEMENT (Chandelier Exit) ====================
        # If TP3 hit, manage runner with trailing stop
        if tp_ladder.tp3_hit:
            # Update chandelier stop based on highest high
            new_chandelier = current_price - (CONFIG.CHANDELIER_ATR_MULT * tp_ladder.atr)
            
            if current_price < tp_ladder.chandelier_stop:
                return TPDecision(
                    should_take=True,
                    tp_level="RUNNER",
                    position_to_sell_pct=tp_ladder.runner_position_pct,
                    reasons=[f"Chandelier Exit triggered at ${tp_ladder.chandelier_stop:.4f}"]
                )
        
        # ==================== HOLD ====================
        return TPDecision(
            should_take=False,
            tp_level="HOLD",
            position_to_sell_pct=0.0,
            reasons=["No TP conditions met, continue holding"]
        )
    
    @classmethod
    def calculate_chandelier_exit(
        cls,
        highest_high: float,
        atr: float,
        multiplier: float = CONFIG.CHANDELIER_ATR_MULT
    ) -> float:
        """
        Trailing stop for runner position.
        Research: "Chandelier Exit = High - (3 × ATR)"
        """
        return highest_high - (atr * multiplier)


# Backward compatibility: Keep EnhancedTPEngineV5 as alias
class EnhancedTPEngineV5:
    """
    Enhanced TP Engine v5.0 - Now using ShortSqueezeTPEngine.
    Kept for backward compatibility.
    """
    
    SQUEEZE_MAGNITUDE = ShortSqueezeTPEngine.SQUEEZE_MAGNITUDE
    
    @classmethod
    def get_market_tier(cls, volume_24h: float) -> str:
        return ShortSqueezeTPEngine.get_market_tier(volume_24h)
    
    @classmethod
    def calculate(
        cls,
        entry_price: float,
        atr: float,
        highest_high: float,
        volume_24h: float,
        conviction: str = "MEDIUM",
        closes: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        funding_rate: float = 0.0,
        vpvr_hvns: Optional[List[float]] = None,
        liquidation_clusters: Optional[List[float]] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> EnhancedTPLadder:
        """
        Calculate TP ladder using the new Short Squeeze TP Engine with Magnetic TPs.
        """
        # Provide empty arrays if not given
        if closes is None:
            closes = np.array([entry_price])
        if volumes is None:
            volumes = np.array([0.0])
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes
        
        return ShortSqueezeTPEngine.calculate_structural_tps(
            entry_price=entry_price,
            atr=atr,
            highest_high=highest_high,
            volume_24h=volume_24h,
            closes=closes,
            volumes=volumes,
            initial_funding_rate=funding_rate,
            vpvr_hvns=vpvr_hvns,
            liquidation_clusters=liquidation_clusters,
            conviction=conviction,
            highs=highs,
            lows=lows
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: VPIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VPINEngine:
    @staticmethod
    def calculate(
        taker_buy_vol: np.ndarray,
        taker_sell_vol: np.ndarray,
        window: int = CONFIG.VPIN_WINDOW
    ) -> VPINResult:
        if len(taker_buy_vol) < window:
            return VPINResult(0.5, 0, 0, 0, False, False)
        
        buy_vol = taker_buy_vol[-window:]
        sell_vol = taker_sell_vol[-window:]
        
        total_buy = np.sum(buy_vol)
        total_sell = np.sum(sell_vol)
        total_vol = total_buy + total_sell
        
        if total_vol == 0:
            return VPINResult(0.5, 0, 0, 0, False, False)
        
        imbalance = np.sum(np.abs(buy_vol - sell_vol))
        vpin = imbalance / total_vol
        
        return VPINResult(
            value=vpin,
            buy_volume=total_buy,
            sell_volume=total_sell,
            imbalance=imbalance,
            is_toxic=vpin > CONFIG.VPIN_TOXICITY_THRESHOLD,
            is_exhausted=vpin < CONFIG.VPIN_EXHAUSTION_THRESHOLD
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: HURST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HurstEngine:
    @staticmethod
    def calculate(prices: np.ndarray, window: int = CONFIG.HURST_WINDOW) -> HurstResult:
        if len(prices) < window:
            return HurstResult(0.5, "RANDOM", True, 0.0)
        
        try:
            series = prices[-window:]
            returns = np.diff(np.log(series))
            
            if len(returns) < 20:
                return HurstResult(0.5, "RANDOM", True, 0.0)
            
            lags = list(range(2, min(20, len(returns) // 4)))
            rs_values = []
            
            for lag in lags:
                n_chunks = len(returns) // lag
                if n_chunks < 2:
                    continue
                
                rs_chunk = []
                for i in range(n_chunks):
                    chunk = returns[i * lag:(i + 1) * lag]
                    if len(chunk) < lag:
                        continue
                    
                    mean_chunk = np.mean(chunk)
                    cum_dev = np.cumsum(chunk - mean_chunk)
                    R = np.max(cum_dev) - np.min(cum_dev)
                    S = np.std(chunk, ddof=1)
                    
                    if S > 0:
                        rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) < 3:
                return HurstResult(0.5, "RANDOM", True, 0.0)
            
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_rs)
            
            hurst = slope
            confidence = r_value ** 2
            
            if hurst < CONFIG.HURST_MEAN_REVERT_THRESHOLD:
                regime = "MEAN_REVERTING"
                valid = False
            elif hurst > CONFIG.HURST_TRENDING_THRESHOLD:
                regime = "TRENDING"
                valid = True
            else:
                regime = "RANDOM"
                valid = True
            
            return HurstResult(value=hurst, regime=regime, is_valid_for_squeeze=valid, confidence=confidence)
            
        except Exception as e:
            log.debug(f"Hurst error: {e}")
            return HurstResult(0.5, "RANDOM", True, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: DATA INGESTION (v5.0 - with Spot data for Spot-Led detection)
# ═══════════════════════════════════════════════════════════════════════════════

class DataIngestorV5:
    """
    Multi-exchange data ingestor.
    Fetches from Binance AND Bybit to scan the FULL perpetual universe.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.error_count = 0
        
        # Cache for BTC regime data
        self.btc_closes: np.ndarray = np.array([])
        self.btcdom_closes: np.ndarray = np.array([])
        self.btc_regime: Optional[BTCRegimeResult] = None
        self.last_btc_update: datetime = datetime.min
        
        # Symbol cache
        self.binance_symbols: List[Dict] = []
        self.bybit_symbols: List[Dict] = []
        self.last_symbol_refresh: datetime = datetime.min
    
    async def start(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=15)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    # ================== BINANCE API ==================
    
    async def fetch_binance_tickers(self) -> List[Dict]:
        """Fetch ALL Binance futures tickers"""
        try:
            url = f"{CONFIG.BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
            self.request_count += 1
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Add exchange tag
                    for d in data:
                        d['exchange'] = 'binance'
                    log.debug(f"Binance: {len(data)} tickers")
                    return data
        except Exception as e:
            log.debug(f"Binance ticker error: {e}")
            self.error_count += 1
        return []
    
    async def fetch_binance_klines(self, symbol: str, interval: str = "1m", limit: int = 200) -> Dict:
        """Fetch Binance futures klines"""
        try:
            url = f"{CONFIG.BINANCE_FUTURES_API}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    if not raw:
                        return {}
                    
                    return {
                        'exchange': 'binance',
                        'timestamps': np.array([k[0] for k in raw], dtype=np.float64),
                        'opens': np.array([float(k[1]) for k in raw]),
                        'highs': np.array([float(k[2]) for k in raw]),
                        'lows': np.array([float(k[3]) for k in raw]),
                        'closes': np.array([float(k[4]) for k in raw]),
                        'volumes': np.array([float(k[5]) for k in raw]),
                        'quote_volume': np.array([float(k[7]) for k in raw]),
                        'taker_buy_vol': np.array([float(k[9]) for k in raw]),
                        'taker_sell_vol': np.array([float(k[5]) for k in raw]) - np.array([float(k[9]) for k in raw]),
                    }
                elif resp.status == 429:
                    log.warning("Binance rate limited")
                    await asyncio.sleep(1)
        except Exception as e:
            log.debug(f"Binance klines error {symbol}: {e}")
            self.error_count += 1
        return {}
    
    async def fetch_binance_spot_klines(self, symbol: str, limit: int = 50) -> Dict:
        """Fetch Binance spot klines for spot-led detection"""
        try:
            url = f"{CONFIG.BINANCE_SPOT_API}/api/v3/klines"
            params = {"symbol": symbol, "interval": "1m", "limit": limit}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    if not raw:
                        return {}
                    return {
                        'volumes': np.array([float(k[5]) for k in raw]),
                    }
        except:
            pass
        return {}
    
    async def fetch_binance_funding(self, symbol: str) -> Tuple[float, float]:
        """Fetch Binance funding rates"""
        try:
            url = f"{CONFIG.BINANCE_FUTURES_API}/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": 2}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if len(data) >= 2:
                        return float(data[-1].get('fundingRate', 0)), float(data[-2].get('fundingRate', 0))
                    elif len(data) == 1:
                        return float(data[0].get('fundingRate', 0)), 0.0
        except:
            self.error_count += 1
        return 0.0, 0.0
    
    async def fetch_binance_oi(self, symbol: str) -> float:
        """Fetch Binance open interest"""
        try:
            url = f"{CONFIG.BINANCE_FUTURES_API}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('openInterest', 0))
        except:
            self.error_count += 1
        return 0.0
    
    # ================== BYBIT API ==================
    
    async def fetch_bybit_tickers(self) -> List[Dict]:
        """Fetch ALL Bybit linear perpetual tickers"""
        try:
            url = f"{CONFIG.BYBIT_API}/v5/market/tickers"
            params = {"category": "linear"}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('retCode') == 0:
                        tickers = data.get('result', {}).get('list', [])
                        # Convert to unified format
                        unified = []
                        for t in tickers:
                            if t.get('symbol', '').endswith('USDT'):
                                unified.append({
                                    'symbol': t['symbol'],
                                    'exchange': 'bybit',
                                    'lastPrice': t.get('lastPrice', '0'),
                                    'priceChangePercent': t.get('price24hPcnt', '0'),
                                    'quoteVolume': t.get('turnover24h', '0'),
                                    'bidPrice': t.get('bid1Price', '0'),
                                    'askPrice': t.get('ask1Price', '0'),
                                })
                        log.debug(f"Bybit: {len(unified)} tickers")
                        return unified
        except Exception as e:
            log.debug(f"Bybit ticker error: {e}")
            self.error_count += 1
        return []
    
    async def fetch_bybit_klines(self, symbol: str, interval: str = "1", limit: int = 200) -> Dict:
        """Fetch Bybit klines"""
        try:
            url = f"{CONFIG.BYBIT_API}/v5/market/kline"
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('retCode') == 0:
                        raw = data.get('result', {}).get('list', [])
                        if not raw:
                            return {}
                        
                        # Bybit returns newest first, need to reverse
                        raw = list(reversed(raw))
                        
                        return {
                            'exchange': 'bybit',
                            'timestamps': np.array([float(k[0]) for k in raw]),
                            'opens': np.array([float(k[1]) for k in raw]),
                            'highs': np.array([float(k[2]) for k in raw]),
                            'lows': np.array([float(k[3]) for k in raw]),
                            'closes': np.array([float(k[4]) for k in raw]),
                            'volumes': np.array([float(k[5]) for k in raw]),
                            'quote_volume': np.array([float(k[6]) for k in raw]),
                            # Bybit doesn't provide taker buy/sell split in klines
                            # We estimate 50/50 or use turnover
                            'taker_buy_vol': np.array([float(k[5]) for k in raw]) * 0.5,
                            'taker_sell_vol': np.array([float(k[5]) for k in raw]) * 0.5,
                        }
        except Exception as e:
            log.debug(f"Bybit klines error {symbol}: {e}")
            self.error_count += 1
        return {}
    
    async def fetch_bybit_funding(self, symbol: str) -> Tuple[float, float]:
        """Fetch Bybit funding rate"""
        try:
            url = f"{CONFIG.BYBIT_API}/v5/market/funding/history"
            params = {"category": "linear", "symbol": symbol, "limit": 2}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('retCode') == 0:
                        rates = data.get('result', {}).get('list', [])
                        if len(rates) >= 2:
                            return float(rates[0].get('fundingRate', 0)), float(rates[1].get('fundingRate', 0))
                        elif len(rates) == 1:
                            return float(rates[0].get('fundingRate', 0)), 0.0
        except:
            self.error_count += 1
        return 0.0, 0.0
    
    async def fetch_bybit_oi(self, symbol: str) -> float:
        """Fetch Bybit open interest"""
        try:
            url = f"{CONFIG.BYBIT_API}/v5/market/open-interest"
            params = {"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 1}
            self.request_count += 1
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('retCode') == 0:
                        oi_list = data.get('result', {}).get('list', [])
                        if oi_list:
                            return float(oi_list[0].get('openInterest', 0))
        except:
            self.error_count += 1
        return 0.0
    
    # ================== UNIFIED METHODS ==================
    
    async def fetch_all_tickers(self) -> List[Dict]:
        """Fetch tickers from ALL enabled exchanges"""
        all_tickers = []
        
        tasks = []
        if CONFIG.USE_BINANCE:
            tasks.append(self.fetch_binance_tickers())
        if CONFIG.USE_BYBIT:
            tasks.append(self.fetch_bybit_tickers())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_tickers.extend(result)
        
        log.info(f"Total universe: {len(all_tickers)} perpetual contracts")
        return all_tickers
    
    async def update_btc_regime(self):
        """Update BTC regime filter data"""
        now = datetime.now()
        if (now - self.last_btc_update).seconds < 60:
            return
        
        try:
            # Fetch BTC data from Binance (most liquid)
            btc_data = await self.fetch_binance_klines(CONFIG.BTC_SYMBOL, "1h", 100)
            if btc_data:
                self.btc_closes = btc_data['closes']
            
            # Fetch BTC Dominance
            try:
                btcdom_data = await self.fetch_binance_klines(CONFIG.BTCDOM_SYMBOL, "1h", 100)
                if btcdom_data:
                    self.btcdom_closes = btcdom_data['closes']
            except:
                self.btcdom_closes = np.array([])
            
            self.btc_regime = BTCRegimeEngine.analyze(
                self.btc_closes,
                self.btcdom_closes if len(self.btcdom_closes) > 0 else None
            )
            
            self.last_btc_update = now
            log.debug(f"BTC Regime: {self.btc_regime.regime.value}")
            
        except Exception as e:
            log.debug(f"BTC regime error: {e}")
    
    async def fetch_symbol_data(self, symbol: str, ticker: Dict) -> Dict:
        """Fetch all data for a symbol from appropriate exchange"""
        try:
            exchange = ticker.get('exchange', 'binance')
            
            if exchange == 'binance':
                # Parallel fetch for Binance
                tasks = [
                    self.fetch_binance_klines(symbol, "1m", CONFIG.KLINE_LIMIT),
                    self.fetch_binance_spot_klines(symbol, 50),
                    self.fetch_binance_funding(symbol),
                    self.fetch_binance_oi(symbol),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                perp_klines = results[0] if not isinstance(results[0], Exception) else {}
                spot_klines = results[1] if not isinstance(results[1], Exception) else {}
                funding_data = results[2] if not isinstance(results[2], Exception) else (0.0, 0.0)
                oi = results[3] if not isinstance(results[3], Exception) else 0.0
                
            else:  # bybit
                tasks = [
                    self.fetch_bybit_klines(symbol, "1", CONFIG.KLINE_LIMIT),
                    self.fetch_bybit_funding(symbol),
                    self.fetch_bybit_oi(symbol),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                perp_klines = results[0] if not isinstance(results[0], Exception) else {}
                spot_klines = {}  # Bybit spot requires different symbol format
                funding_data = results[1] if not isinstance(results[1], Exception) else (0.0, 0.0)
                oi = results[2] if not isinstance(results[2], Exception) else 0.0
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'ticker': ticker,
                'perp_klines': perp_klines,
                'spot_klines': spot_klines,
                'funding_current': funding_data[0],
                'funding_previous': funding_data[1],
                'open_interest': oi,
                'oi_change_pct': 0.0,  # Would need historical
                'ls_ratio': 1.0,  # Skip for speed
            }
        except Exception as e:
            log.debug(f"Symbol data error {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE: DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class TradeDatabaseV5:
    def __init__(self, db_path: str = CONFIG.DB_PATH):
        self.db_path = db_path
        self.db = None
    
    async def initialize(self):
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("PRAGMA journal_mode=WAL;")
        await self._create_tables()
        await self.db.commit()
    
    async def _create_tables(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS signals_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                score REAL NOT NULL,
                archetype TEXT,
                btc_regime TEXT,
                is_spot_led INTEGER,
                cvd_slope REAL,
                ofi_zscore REAL,
                rvol REAL,
                gates_passed INTEGER,
                veto_reasons TEXT,
                conviction TEXT,
                reasons TEXT
            )
        """)
        
        # Trades tracking table with SL/TP outcomes
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS trades_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_score REAL NOT NULL,
                archetype TEXT,
                stop_loss REAL,
                tp1_price REAL,
                tp2_price REAL,
                tp3_price REAL,
                status TEXT DEFAULT 'ACTIVE',
                hit_sl INTEGER DEFAULT 0,
                hit_tp1 INTEGER DEFAULT 0,
                hit_tp2 INTEGER DEFAULT 0,
                hit_tp3 INTEGER DEFAULT 0,
                exit_price REAL,
                exit_timestamp TEXT,
                pnl_pct REAL,
                expectancy REAL
            )
        """)
    
    async def log_signal(self, signal: SqueezeSignalV5):
        await self.db.execute("""
            INSERT INTO signals_v5 
            (timestamp, symbol, exchange, score, archetype, btc_regime, is_spot_led, 
             cvd_slope, ofi_zscore, rvol, gates_passed, veto_reasons, conviction, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(),
            signal.symbol,
            signal.exchange,
            signal.ignition_score,
            signal.archetype.value,
            signal.btc_regime.regime.value,
            1 if signal.spot_led.is_spot_led else 0,
            signal.cvd.cvd_slope,
            signal.ofi.ofi_zscore,
            signal.rvol.rvol,
            1 if signal.gates.all_passed else 0,
            ",".join(signal.veto_reasons),
            signal.conviction,
            ",".join(signal.reasons)
        ))
        await self.db.commit()
    
    async def log_trade(self, signal: SqueezeSignalV5) -> int:
        """Log a new trade entry and return trade ID"""
        cursor = await self.db.execute("""
            INSERT INTO trades_v5 
            (timestamp, symbol, exchange, entry_price, entry_score, archetype,
             stop_loss, tp1_price, tp2_price, tp3_price, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
        """, (
            signal.timestamp.isoformat(),
            signal.symbol,
            signal.exchange,
            signal.price,
            signal.ignition_score,
            signal.archetype.value,
            signal.tp_ladder.stop_loss,
            signal.tp_ladder.tp1_price,
            signal.tp_ladder.tp2_price,
            signal.tp_ladder.tp3_price
        ))
        await self.db.commit()
        return cursor.lastrowid
    
    async def update_trade_outcome(self, trade_id: int, hit_type: str, exit_price: float, pnl_pct: float):
        """Update trade with hit type: 'SL', 'TP1', 'TP2', 'TP3'"""
        hit_col = f"hit_{hit_type.lower()}"
        if hit_col not in ['hit_sl', 'hit_tp1', 'hit_tp2', 'hit_tp3']:
            return
        
        status = 'CLOSED_SL' if hit_type == 'SL' else f'CLOSED_{hit_type}'
        await self.db.execute(f"""
            UPDATE trades_v5 SET {hit_col} = 1, status = ?, exit_price = ?, 
            exit_timestamp = ?, pnl_pct = ? WHERE id = ?
        """, (status, exit_price, datetime.now().isoformat(), pnl_pct, trade_id))
        await self.db.commit()
    
    async def get_trade_stats(self) -> Dict[str, Any]:
        """Get comprehensive trade statistics"""
        cursor = await self.db.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(hit_sl) as sl_count,
                SUM(hit_tp1) as tp1_count,
                SUM(hit_tp2) as tp2_count,
                SUM(hit_tp3) as tp3_count,
                SUM(CASE WHEN hit_tp1 = 1 OR hit_tp2 = 1 OR hit_tp3 = 1 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl,
                SUM(CASE WHEN status LIKE 'CLOSED%' THEN 1 ELSE 0 END) as closed_trades
            FROM trades_v5
        """)
        row = await cursor.fetchone()
        
        if row and row[0] > 0:
            total = row[0]
            sl = row[1] or 0
            tp1 = row[2] or 0
            tp2 = row[3] or 0
            tp3 = row[4] or 0
            wins = row[5] or 0
            avg_pnl = row[6] or 0.0
            closed = row[7] or 0
            
            win_rate = (wins / closed * 100) if closed > 0 else 0.0
            # Expectancy = (Win% x Avg Win) - (Loss% x Avg Loss)
            # Simplified: avg_pnl is our expectancy proxy
            expectancy = avg_pnl if avg_pnl else 0.0
            
            return {
                'total': total,
                'active': total - closed,
                'closed': closed,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'wins': wins,
                'losses': sl,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'expectancy': expectancy
            }
        return {'total': 0, 'active': 0, 'closed': 0, 'sl': 0, 'tp1': 0, 'tp2': 0, 'tp3': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'expectancy': 0.0}
    
    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades for display"""
        cursor = await self.db.execute("""
            SELECT id, timestamp, symbol, exchange, entry_price, entry_score, archetype,
                   stop_loss, tp1_price, tp2_price, tp3_price, status,
                   hit_tp1, hit_tp2, hit_tp3
            FROM trades_v5 
            WHERE status = 'ACTIVE' OR status LIKE 'PARTIAL%'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        rows = await cursor.fetchall()
        trades = []
        for row in rows:
            trades.append({
                'id': row[0],
                'timestamp': row[1],
                'symbol': row[2],
                'exchange': row[3],
                'entry_price': row[4],
                'entry_score': row[5],
                'archetype': row[6],
                'stop_loss': row[7],
                'tp1_price': row[8],
                'tp2_price': row[9],
                'tp3_price': row[10],
                'status': row[11],
                'hit_tp1': row[12],
                'hit_tp2': row[13],
                'hit_tp3': row[14]
            })
        return trades
    
    async def close(self):
        if self.db:
            await self.db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE: GOD OF GODS v5.0
# ═══════════════════════════════════════════════════════════════════════════════

class GodOfGodsV5:
    """
    God of Gods Squeeze Scanner v5.0 - Research Edition
    
    Implements all 18 upgrades from merged_squeeze.md research:
    - BTC Regime Filter
    - Spot-Led Detection
    - True CVD with Acceleration
    - OFI Z-Score
    - Waterfall Gates
    - ATR 2.3x Stop
    - And more...
    """
    
    def __init__(self):
        self.ingestor = DataIngestorV5()
        self.database = TradeDatabaseV5()
        self.top_signals: List[SqueezeSignalV5] = []
        self.scan_count = 0
        self.start_time = datetime.now()
        self.last_scan_duration = 0.0
        self.symbols_scanned = 0
        self.signals_found = 0
        self.vetoed_count = 0
        self.last_error = ""
        self.trade_stats: Dict[str, Any] = {'total': 0, 'sl': 0, 'tp1': 0, 'tp2': 0, 'tp3': 0, 'wins': 0, 'win_rate': 0.0, 'expectancy': 0.0}
        self.active_trades: List[Dict[str, Any]] = []
    
    async def initialize(self):
        await self.ingestor.start()
        await self.database.initialize()
        # Load existing active trades to prevent duplicate logging on first scan
        self.active_trades = await self.database.get_active_trades()
        self.trade_stats = await self.database.get_trade_stats()
        log.info(f"🔱⚡ God of Gods Scanner v5.0 initialized | Active trades: {len(self.active_trades)}")
    
    async def close(self):
        await self.ingestor.stop()
        await self.database.close()
    
    def calculate_score(
        self,
        vpin: VPINResult,
        hurst: HurstResult,
        funding_rate: float,
        funding_flip: bool,
        ls_ratio: float,
        is_microcap: bool,
        bb_kc: BBKCSqueezeResult,
        spot_led: SpotLedResult,
        cvd: CVDResult,
        ofi: OFIResult,
        god_candle: GodCandleResult,
        rvol: RVOLResult,
        gates: WaterfallGates
    ) -> Tuple[float, List[str]]:
        """Calculate ignition score with all v5.0 factors"""
        
        score = 0.0
        reasons = []
        
        # === VPIN (25 pts max) ===
        if vpin.value > 0.50:
            score += 25
            reasons.append(f"VPIN {vpin.value:.2f} (toxic)")
        elif vpin.value > 0.40:
            score += 20
            reasons.append(f"VPIN {vpin.value:.2f}")
        elif vpin.value > 0.35:
            score += 15
        elif vpin.value > 0.25:
            score += 8
        
        # === Hurst (15 pts max) ===
        if hurst.value > 0.60:
            score += 15
            reasons.append(f"Hurst {hurst.value:.2f} (trending)")
        elif hurst.value > 0.55:
            score += 12
        elif hurst.value > 0.50:
            score += 8
        
        # === Funding (15 pts max) ===
        if funding_rate < -0.005:
            score += 15
            reasons.append(f"Funding {funding_rate*100:.3f}% (oversold)")
        elif funding_rate < -0.002:
            score += 12
        elif funding_rate < -0.001:
            score += 8
        elif funding_rate < 0:
            score += 4
        
        # === FUNDING FLIP BONUS (+7 pts - Research: 58% PPV) ===
        if funding_flip:
            score += CONFIG.FUNDING_FLIP_BONUS
            reasons.append("Funding FLIP +7")
        
        # === L/S Ratio (10 pts max) ===
        if ls_ratio < 0.7:
            score += 10
            reasons.append(f"L/S {ls_ratio:.2f} (short crowded)")
        elif ls_ratio < 0.85:
            score += 7
        elif ls_ratio < 1.0:
            score += 4
        
        # === Microcap bonus (20 pts) ===
        if is_microcap:
            score += CONFIG.MICROCAP_BONUS_SCORE
            reasons.append("Microcap +20")
        
        # === BB-KC Squeeze (15 pts max) ===
        if bb_kc.is_in_squeeze:
            squeeze_pts = min(15, bb_kc.squeeze_duration * 0.75)
            score += squeeze_pts
            reasons.append(f"Squeeze {bb_kc.squeeze_duration}bars +{squeeze_pts:.0f}")
        
        if bb_kc.squeeze_fired and bb_kc.squeeze_direction == "LONG":
            score += 10
            reasons.append("Squeeze FIRED +10")
        
        # === SPOT-LED BONUS (NEW - 15 pts max) ===
        if spot_led.is_spot_led:
            score += spot_led.bonus
            reasons.append(f"Spot-led +{spot_led.bonus:.0f}")
        
        # === CVD SCORE (NEW - 15 pts max) ===
        if cvd.cvd_slope > 0.002:
            score += 10
            reasons.append("Strong CVD slope")
        elif cvd.cvd_slope > 0.001:
            score += 6
        
        if cvd.cvd_acceleration > 0:
            score += 5
            reasons.append("CVD accelerating")
        
        if cvd.has_bullish_divergence:
            score += 8
            reasons.append("Bullish CVD divergence")
        
        # === OFI SCORE (NEW - 10 pts max) ===
        if ofi.is_buy_pressure:
            if ofi.strength == "STRONG":
                score += 10
                reasons.append(f"OFI z={ofi.ofi_zscore:.1f} STRONG")
            else:
                score += 6
                reasons.append(f"OFI buy pressure")
        
        # === GOD CANDLE BONUS (NEW - 25 pts) ===
        if god_candle.is_god_candle and god_candle.direction == "LONG":
            score += CONFIG.GOD_CANDLE_BONUS
            reasons.append(f"GOD CANDLE {god_candle.price_change_pct:.1f}% +25")
        
        # === RVOL BONUS (NEW - 10 pts max) ===
        if rvol.regime == "EXTREME":
            score += 10
            reasons.append(f"RVOL {rvol.rvol:.1f}x EXTREME")
        elif rvol.regime == "ALERT":
            score += 6
        
        # === GATES BONUS (NEW - 15 pts if all passed) ===
        if gates.all_passed:
            score += 15
            reasons.append("All gates PASSED +15")
        elif gates.gate_c_ignition == GateStatus.PASSED:
            score += 8
        elif gates.gate_b_compression == GateStatus.PASSED:
            score += 4
        
        return score, reasons
    
    def check_vetos(
        self,
        spread_pct: float,
        funding_rate: float,
        spot_led: SpotLedResult,
        bb_kc: BBKCSqueezeResult,
        btc_regime: BTCRegimeResult
    ) -> Tuple[bool, List[str]]:
        """Check hard veto conditions"""
        
        veto_reasons = []
        
        # === SPREAD VETO (Research: >1.5% = immediate loss) ===
        if CONFIG.SPREAD_VETO_ENABLED and spread_pct > CONFIG.SPREAD_MAX_PCT:
            veto_reasons.append(f"Spread {spread_pct:.2f}% > 1.5%")
        
        # === CROWDED FUNDING VETO (Research: >0.1% = late to party) ===
        if funding_rate > CONFIG.FUNDING_CROWDED_THRESHOLD:
            veto_reasons.append(f"Funding {funding_rate*100:.3f}% CROWDED")
        
        # === PERP-LED TRAP VETO (Research: ratio >4x = fake pump) ===
        # Only veto if we have actual spot data (Binance has it, Bybit doesn't)
        if spot_led.is_perp_led_trap and spot_led.spot_volume > 1:  # Has real spot data
            veto_reasons.append(f"PERP-LED TRAP ratio {spot_led.ratio:.1f}x")
        
        # === BB WIDTH SLEEP VETO (Research: <10th percentile = no trade) ===
        if CONFIG.BB_SLEEP_VETO_ENABLED and bb_kc.is_sleeping:
            veto_reasons.append(f"BB Width {bb_kc.bb_width_percentile:.0f}th% - SLEEPING")
        
        # === BTC REGIME VETO (Research: RISK_OFF = 27% vs 62% win rate) ===
        # Only hard veto if REGIME_HARD_VETO is True
        if CONFIG.REGIME_ENABLED and CONFIG.REGIME_HARD_VETO and not btc_regime.should_trade:
            veto_reasons.append(f"BTC REGIME: {btc_regime.reason}")
        
        is_veto = len(veto_reasons) > 0
        return is_veto, veto_reasons
    
    async def analyze_symbol(self, data: Dict) -> Optional[SqueezeSignalV5]:
        """Analyze a symbol with all v5.0 features"""
        try:
            if 'error' in data or not data.get('perp_klines'):
                return None
            
            symbol = data['symbol']
            exchange = data.get('exchange', 'binance')  # Get exchange
            ticker = data['ticker']
            perp_klines = data['perp_klines']
            spot_klines = data.get('spot_klines', {})
            funding_current = data.get('funding_current', 0.0)
            funding_previous = data.get('funding_previous', 0.0)
            oi = data.get('open_interest', 0.0)
            oi_change = data.get('oi_change_pct', 0.0)
            ls_ratio = data.get('ls_ratio', 1.0)
            
            # Extract arrays
            if len(perp_klines.get('closes', [])) < 50:
                return None
            
            closes = perp_klines['closes']
            highs = perp_klines['highs']
            lows = perp_klines['lows']
            opens = perp_klines['opens']
            volumes = perp_klines['volumes']
            taker_buy = perp_klines['taker_buy_vol']
            taker_sell = perp_klines['taker_sell_vol']
            
            current_price = float(closes[-1])
            volume_24h = float(ticker.get('quoteVolume', 0))
            highest_high = float(np.max(highs[-20:]))
            
            # Calculate spread
            bid = float(ticker.get('bidPrice', current_price))
            ask = float(ticker.get('askPrice', current_price))
            spread_pct = (ask - bid) / current_price * 100 if current_price > 0 else 0
            
            # Price change for gates
            price_change_5m = 0.0
            if len(closes) >= 5:
                price_change_5m = (closes[-1] - closes[-5]) / closes[-5] * 100
            
            # Taker buy ratio
            taker_ratio = 1.0
            if len(taker_buy) >= 5 and len(taker_sell) >= 5:
                recent_buy = np.sum(taker_buy[-5:])
                recent_sell = np.sum(taker_sell[-5:]) + 1e-10
                taker_ratio = recent_buy / recent_sell
            
            # === Calculate ATR ===
            atr = BBKCSqueezeEngineV5.calculate_atr(highs, lows, closes, CONFIG.ATR_PERIOD)
            if atr == 0:
                atr = current_price * 0.02
            
            # === Core Analysis ===
            vpin = VPINEngine.calculate(taker_buy, taker_sell)
            hurst = HurstEngine.calculate(closes)
            bb_kc = BBKCSqueezeEngineV5.calculate(highs, lows, closes)
            
            # === NEW v5.0 Analysis ===
            
            # CVD Engine
            cvd = CVDEngine.calculate(taker_buy, taker_sell, closes)
            
            # OFI Engine
            ofi = OFIEngine.calculate(taker_buy, taker_sell)
            
            # God Candle Detection
            god_candle = GodCandleEngine.calculate(opens, closes, volumes)
            
            # RVOL Engine
            rvol = RVOLEngine.calculate(volumes)
            
            # Spot-Led Detection
            perp_vol = float(np.sum(volumes[-20:]))
            spot_vol = 0.0
            has_real_spot_data = False
            if spot_klines and 'volumes' in spot_klines and len(spot_klines['volumes']) >= 20:
                spot_vol = float(np.sum(spot_klines['volumes'][-20:]))
                has_real_spot_data = True
            else:
                spot_vol = perp_vol * 0.5  # Estimate if no spot data (won't trigger trap)
            spot_led = SpotLedEngine.analyze(perp_vol, spot_vol)
            # Mark if we have real data
            if not has_real_spot_data:
                spot_led = SpotLedResult(
                    perp_volume=spot_led.perp_volume,
                    spot_volume=0,  # Mark as no real data
                    ratio=1.0,  # Neutral
                    is_spot_led=False,
                    is_perp_led_trap=False,  # Can't determine
                    bonus=0  # No bonus without data
                )
            
            # Funding Flip Detection
            funding_flip = (
                funding_previous < 0 and 
                funding_current >= 0 and 
                oi_change > 0
            )
            
            # OI Acceleration
            oi_accel = 0.0  # Would need historical OI data
            
            # Waterfall Gates
            gates = WaterfallGateEngine.evaluate(
                volume_24h=volume_24h,
                spread_pct=spread_pct,
                is_in_squeeze=bb_kc.is_in_squeeze,
                rvol=rvol.rvol,
                price_change_5m_pct=price_change_5m,
                taker_buy_ratio=taker_ratio,
                oi_delta=oi_change,
                spot_led_result=spot_led
            )
            
            # BTC Regime (from cached data)
            btc_regime = self.ingestor.btc_regime or BTCRegimeResult(
                regime=MarketRegime.UNKNOWN, btc_price=0, btc_sma50=0,
                btc_above_sma=True, btcdom_trend="UNKNOWN", btcdom_change_1h=0,
                should_trade=True, reason="No BTC data"
            )
            
            # Archetype Classification
            archetype, arch_reason = ArchetypeEngine.classify(
                rvol, cvd, bb_kc, ofi, price_change_5m, oi_change
            )
            
            # Microcap check
            is_microcap = (
                volume_24h >= CONFIG.MICROCAP_MIN_VOLUME_24H and
                volume_24h <= CONFIG.MICROCAP_MAX_VOLUME_24H
            )
            
            # === Check VETOS ===
            is_veto, veto_reasons = self.check_vetos(
                spread_pct, funding_current, spot_led, bb_kc, btc_regime
            )
            
            # === Calculate Score ===
            score, reasons = self.calculate_score(
                vpin, hurst, funding_current, funding_flip, ls_ratio,
                is_microcap, bb_kc, spot_led, cvd, ofi, god_candle,
                rvol, gates
            )
            
            # Conviction
            if score >= 90:
                conviction = "EXTREME"
            elif score >= 80:
                conviction = "HIGH"
            elif score >= 70:
                conviction = "MEDIUM"
            else:
                conviction = "LOW"
            
            # Should enter
            should_enter = (
                score >= CONFIG.MIN_IGNITION_SCORE and
                not is_veto and
                hurst.is_valid_for_squeeze and
                not vpin.is_exhausted
            )
            
            # Debug: Log why high-score signals don't enter
            if score >= 70 and not should_enter:
                fail_reasons = []
                if is_veto:
                    fail_reasons.append(f"VETOED:{','.join(veto_reasons[:2])}")
                if not hurst.is_valid_for_squeeze:
                    fail_reasons.append(f"Hurst={hurst.value:.2f}<0.53")
                if vpin.is_exhausted:
                    fail_reasons.append(f"VPIN_EXHAUSTED={vpin.value:.2f}")
                log.warning(f"⚠️ {symbol} Score:{score:.0f} NOT TRADEABLE: {' | '.join(fail_reasons)}")
            
            # === RSI Calculation (NEW - for 7-metric TP system) ===
            rsi_values = RSIEngine.calculate_multi_timeframe(closes)
            
            # === Volume Climax Detection (NEW - Reversal Guard) ===
            rsi_current = rsi_values.get("1m", 50)
            volume_climax = ShortSqueezeTPEngine.detect_volume_climax(
                opens, highs, lows, closes, volumes, rsi_current
            )
            
            # === TP Ladder with MAGNETIC TP SYSTEM + 7-Metric Short Squeeze System ===
            # Research: "Price is drawn to areas of high liquidity"
            # Uses Fibonacci, Swing Highs, VWAP Bands, HVNs, Value Area with Confluence Scoring
            tp_ladder = ShortSqueezeTPEngine.calculate_structural_tps(
                entry_price=current_price,
                atr=atr,
                highest_high=highest_high,
                volume_24h=volume_24h,
                closes=closes,
                volumes=volumes,
                initial_funding_rate=funding_current,
                vpvr_hvns=None,  # Will be estimated from price/volume data
                liquidation_clusters=None,  # Would need external API (CoinGlass/Hyblock)
                conviction=conviction,
                highs=highs,  # NEW: For magnetic TP calculation
                lows=lows     # NEW: For magnetic TP calculation
            )
            
            # === Evaluate Current TP Decision (for real-time signals) ===
            # This provides immediate TP guidance if entering now
            tp_decision = ShortSqueezeTPEngine.evaluate_tp_decision(
                tp_ladder=tp_ladder,
                current_price=current_price,
                current_funding_rate=funding_current,
                oi_current=oi,
                oi_change_from_peak_pct=oi_change,  # In real use, track peak OI
                rsi_values=rsi_values,
                cvd_result=cvd,
                volume_climax=volume_climax
            )
            
            # Check for panic exit signal - this is a VETO if climax detected
            if volume_climax.is_climax:
                is_veto = True
                veto_reasons.append("VOLUME CLIMAX: " + ", ".join(volume_climax.reasons))
            
            return SqueezeSignalV5(
                symbol=symbol,
                exchange=exchange,  # Include exchange
                timestamp=datetime.now(),
                price=current_price,
                ignition_score=score,
                vpin=vpin,
                hurst=hurst,
                bb_kc_squeeze=bb_kc,
                tp_ladder=tp_ladder,
                btc_regime=btc_regime,
                spot_led=spot_led,
                cvd=cvd,
                ofi=ofi,
                god_candle=god_candle,
                rvol=rvol,
                gates=gates,
                archetype=archetype,
                funding_rate=funding_current,
                funding_flip_detected=funding_flip,
                open_interest=oi,
                oi_change_pct=oi_change,
                oi_acceleration=oi_accel,
                volume_24h=volume_24h,
                long_short_ratio=ls_ratio,
                spread_pct=spread_pct,
                # NEW: 7-Metric TP System Data
                rsi_values=rsi_values,
                volume_climax_detected=volume_climax.is_climax,
                volume_climax_reasons=volume_climax.reasons,
                tp_source=tp_ladder.tp_source,
                # Flags
                is_microcap=is_microcap,
                is_veto=is_veto,
                veto_reasons=veto_reasons,
                conviction=conviction,
                should_enter=should_enter,
                reasons=reasons
            )
            
        except Exception as e:
            log.debug(f"Analysis error {data.get('symbol', '?')}: {e}")
            return None
    
    async def scan_market(self):
        """Main market scan - SCANS ALL COINS on Binance + Bybit"""
        scan_start = time.time()
        self.scan_count += 1
        
        try:
            # Update BTC regime
            await self.ingestor.update_btc_regime()
            
            # Fetch ALL tickers from ALL exchanges
            all_tickers = await self.ingestor.fetch_all_tickers()
            if not all_tickers:
                self.last_error = "No tickers from any exchange"
                return
            
            # Filter USDT perpetual pairs with minimum volume
            usdt_tickers = [
                t for t in all_tickers
                if t.get('symbol', '').endswith('USDT') and
                float(t.get('quoteVolume', 0)) >= CONFIG.MICROCAP_MIN_VOLUME_24H
            ]
            
            # DON'T limit - scan ALL of them!
            # Sort by a mix of volatility and low volume (squeeze candidates)
            # Research: squeezes happen in quiet coins that suddenly wake up
            usdt_tickers.sort(key=lambda x: (
                -abs(float(x.get('priceChangePercent', 0))),  # High volatility
            ))
            
            self.symbols_scanned = len(usdt_tickers)
            log.info(f"Scanning {self.symbols_scanned} perpetual contracts...")
            
            # Fetch and analyze in batches
            signals = []
            vetoed = 0
            batch_size = CONFIG.MAX_CONCURRENT_REQUESTS
            
            for i in range(0, len(usdt_tickers), batch_size):
                batch = usdt_tickers[i:i+batch_size]
                
                # Fetch data
                tasks = [self.ingestor.fetch_symbol_data(t['symbol'], t) for t in batch]
                batch_data = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Analyze
                for data in batch_data:
                    if isinstance(data, Exception) or not data or 'error' in data:
                        continue
                    
                    signal = await self.analyze_symbol(data)
                    if signal:
                        if signal.is_veto:
                            vetoed += 1
                        elif signal.ignition_score >= CONFIG.MIN_DISPLAY_SCORE:
                            signals.append(signal)
                
                # Small delay between batches to respect rate limits
                await asyncio.sleep(CONFIG.BATCH_DELAY_SECONDS)
            
            # Sort and store top signals
            signals.sort(key=lambda x: x.ignition_score, reverse=True)
            self.top_signals = signals[:CONFIG.TOP_CANDIDATES]
            self.signals_found = len(signals)
            self.vetoed_count = vetoed
            
            # Log top signals
            for sig in self.top_signals[:5]:
                await self.database.log_signal(sig)
            
            # Refresh active trades from DB BEFORE checking for duplicates
            self.active_trades = await self.database.get_active_trades()
            active_symbols = {t.get('symbol') for t in self.active_trades}
            
            # Auto-log trades when should_enter = True (avoid duplicates by checking symbol)
            for sig in self.top_signals:
                if sig.should_enter:
                    if sig.symbol not in active_symbols:
                        await self.database.log_trade(sig)
                        active_symbols.add(sig.symbol)  # Add to set to prevent same-scan duplicates
                        log.info(f"🚀 AUTO-TRADE LOGGED: {sig.symbol} @ ${sig.price:.4f} | Score: {sig.ignition_score:.0f}")
            
            # Fetch updated trade stats for ticker display
            self.trade_stats = await self.database.get_trade_stats()
            self.active_trades = await self.database.get_active_trades()
            
            self.last_scan_duration = time.time() - scan_start
            
        except Exception as e:
            self.last_error = str(e)
            log.error(f"Scan error: {e}")
    
    def generate_ui(self) -> Layout:
        """Generate Rich UI layout with color coding, emojis, ticker, and trades table"""
        layout = Layout()
        layout.split_column(
            Layout(name="ticker", size=3),      # Trade stats ticker
            Layout(name="header", size=7),
            Layout(name="signals", ratio=2),    # Main signals table
            Layout(name="flight_control", ratio=2),  # Active trades flight monitor
            Layout(name="archetypes", size=6),  # Archetype legend
            Layout(name="footer", size=3)
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # TICKER - Trade Statistics Bar
        # ═══════════════════════════════════════════════════════════════════
        stats = self.trade_stats
        ticker_text = Text()
        ticker_text.append("📊 TRADES: ", style="bold white")
        ticker_text.append(f"{stats.get('total', 0)}", style="bold cyan")
        ticker_text.append(" │ ", style="dim")
        
        # Win/Loss with colors
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        ticker_text.append("✅ Wins: ", style="green")
        ticker_text.append(f"{wins}", style="bold green")
        ticker_text.append(" │ ", style="dim")
        ticker_text.append("❌ SL: ", style="red")
        ticker_text.append(f"{losses}", style="bold red")
        ticker_text.append(" │ ", style="dim")
        
        # TP breakdown with emojis
        ticker_text.append("🎯 TP1: ", style="yellow")
        ticker_text.append(f"{stats.get('tp1', 0)}", style="bold yellow")
        ticker_text.append(" │ ", style="dim")
        ticker_text.append("🎯🎯 TP2: ", style="bright_yellow")
        ticker_text.append(f"{stats.get('tp2', 0)}", style="bold bright_yellow")
        ticker_text.append(" │ ", style="dim")
        ticker_text.append("🏆 TP3: ", style="bright_green")
        ticker_text.append(f"{stats.get('tp3', 0)}", style="bold bright_green")
        ticker_text.append(" │ ", style="dim")
        
        # Win rate with color coding
        win_rate = stats.get('win_rate', 0)
        wr_color = "bright_green" if win_rate >= 60 else "green" if win_rate >= 50 else "yellow" if win_rate >= 40 else "red"
        ticker_text.append("📈 WR: ", style=wr_color)
        ticker_text.append(f"{win_rate:.1f}%", style=f"bold {wr_color}")
        ticker_text.append(" │ ", style="dim")
        
        # Expectancy
        exp = stats.get('expectancy', 0)
        exp_color = "bright_green" if exp > 0 else "red"
        ticker_text.append("💰 Exp: ", style=exp_color)
        ticker_text.append(f"{exp:+.2f}%", style=f"bold {exp_color}")
        
        layout["ticker"].update(Panel(ticker_text, border_style="bright_blue", title="📈 Performance Ticker"))
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER - Scanner Status
        # ═══════════════════════════════════════════════════════════════════
        regime_emoji = "🟢" if self.ingestor.btc_regime and self.ingestor.btc_regime.should_trade else "🔴"
        regime_color = "green" if self.ingestor.btc_regime and self.ingestor.btc_regime.should_trade else "red"
        regime_text = self.ingestor.btc_regime.regime.value if self.ingestor.btc_regime else "UNKNOWN"
        
        binance_count = sum(1 for s in self.top_signals if s.exchange == 'binance')
        bybit_count = sum(1 for s in self.top_signals if s.exchange == 'bybit')
        
        header_text = Text()
        header_text.append("🔱⚡ GOD OF GODS v5.0 - RESEARCH EDITION\n", style="bold cyan")
        header_text.append(f"{regime_emoji} BTC Regime: ", style="white")
        header_text.append(f"{regime_text}", style=f"bold {regime_color}")
        if self.ingestor.btc_regime:
            header_text.append(f" │ 💰 BTC ${self.ingestor.btc_regime.btc_price:.0f}")
            dom_emoji = "📈" if self.ingestor.btc_regime.btcdom_trend == "UP" else "📉" if self.ingestor.btc_regime.btcdom_trend == "DOWN" else "➡️"
            header_text.append(f" │ {dom_emoji} BTC.D {self.ingestor.btc_regime.btcdom_trend}")
        header_text.append(f"\n🔄 Scans: {self.scan_count} │ ⏱️ {self.last_scan_duration:.1f}s")
        header_text.append(f" │ 🌐 Universe: {self.symbols_scanned} contracts")
        header_text.append(f"\n🎯 Signals: {self.signals_found} │ 🚫 Vetoed: {self.vetoed_count}")
        header_text.append(f" │ 🟠 Binance: {binance_count} │ 🟡 Bybit: {bybit_count}")
        
        layout["header"].update(Panel(header_text, title="⚙️ Scanner Status", border_style="cyan"))
        
        # ═══════════════════════════════════════════════════════════════════
        # SIGNALS TABLE - Color Coded with Emojis
        # ═══════════════════════════════════════════════════════════════════
        signals_table = Table(box=box.ROUNDED, expand=True, show_header=True, header_style="bold white")
        signals_table.add_column("Str", width=2)  # Strength: 🔥90+ 🚀80+ ⚡70+ 🟡60+ 🟠50+ ⚪<50
        signals_table.add_column("Symbol", width=10)
        signals_table.add_column("Score", justify="right", width=5)
        signals_table.add_column("Pattern", width=11)  # Archetype with full name
        signals_table.add_column("RVOL", justify="right", width=5)
        signals_table.add_column("CVD", justify="right", width=5)
        signals_table.add_column("VCIX", width=4)  # V=Viable C=Compress I=Ignite X=Confirm
        signals_table.add_column("FR", justify="right", width=5)
        signals_table.add_column("≥70", width=3)   # Score >= 70
        signals_table.add_column("NoV", width=3)   # Not Vetoed
        signals_table.add_column("Hst", width=3)   # Hurst Valid
        signals_table.add_column("VPN", width=3)   # VPIN not exhausted
        signals_table.add_column("GO", width=2)    # Should Enter
        
        for sig in self.top_signals[:10]:
            # Score-based row styling and emoji
            if sig.ignition_score >= 90:
                row_emoji = "🔥"
                row_style = "bold bright_green"
            elif sig.ignition_score >= 80:
                row_emoji = "🚀"
                row_style = "bold green"
            elif sig.ignition_score >= 70:
                row_emoji = "⚡"
                row_style = "green"
            elif sig.ignition_score >= 60:
                row_emoji = "🟡"
                row_style = "yellow"
            elif sig.ignition_score >= 50:
                row_emoji = "🟠"
                row_style = "bright_black"
            else:
                row_emoji = "⚪"
                row_style = "dim"
            
            # Archetype emoji
            arch_emojis = {
                "FRESH BREAK": "💎", "COIL SQUEEZE": "🌀", "HYPE PUMP": "🎭", "EXTREME VOL": "⚡",
                "DIP BOUNCE": "🔄", "MICRO MOVE": "🍪", "TREND CONT": "📈", "RANGE COIL": "🥦", "NONE": "❓"
            }
            arch_emoji = arch_emojis.get(sig.archetype.value, "❓")
            
            # CVD color
            cvd_val = sig.cvd.cvd_slope * 1000
            cvd_color = "bright_green" if cvd_val > 1 else "green" if cvd_val > 0 else "red" if cvd_val < -1 else "yellow"
            
            # OFI color  
            ofi_val = sig.ofi.ofi_zscore
            ofi_color = "bright_green" if ofi_val > 2 else "green" if ofi_val > 1 else "red" if ofi_val < -1 else "white"
            
            # RVOL color
            rvol_val = sig.rvol.rvol
            rvol_color = "bright_green" if rvol_val >= 5 else "green" if rvol_val >= 3 else "yellow" if rvol_val >= 1.5 else "dim"
            
            # Gates display - V=Viable C=Compress I=Ignite X=Confirm
            v_status = "[green]V[/]" if sig.gates.gate_a_viability == GateStatus.PASSED else "[red]v[/]"
            c_status = "[green]C[/]" if sig.gates.gate_b_compression == GateStatus.PASSED else "[red]c[/]"
            i_status = "[green]I[/]" if sig.gates.gate_c_ignition == GateStatus.PASSED else "[red]i[/]"
            x_status = "[green]X[/]" if sig.gates.gate_d_confirmation == GateStatus.PASSED else "[red]x[/]"
            gates_str = f"{v_status}{c_status}{i_status}{x_status}"
            
            # Funding rate color
            fr_pct = sig.funding_rate * 100
            if fr_pct < -0.3:
                fr_color = "bright_green"
                fr_emoji = "🔥"
            elif fr_pct < -0.1:
                fr_color = "green"
                fr_emoji = "✨"
            elif fr_pct < 0:
                fr_color = "yellow"
                fr_emoji = "⚡"
            elif fr_pct > 0.1:
                fr_color = "red"
                fr_emoji = "⚠️"
            else:
                fr_color = "white"
                fr_emoji = ""
            
            # Entry condition columns
            score_ok = sig.ignition_score >= CONFIG.MIN_IGNITION_SCORE
            no_veto = not sig.is_veto
            hurst_ok = sig.hurst.is_valid_for_squeeze
            vpin_ok = not sig.vpin.is_exhausted
            should_go = sig.should_enter
            
            score_col = "[green]✓[/]" if score_ok else "[red]✗[/]"
            veto_col = "[green]✓[/]" if no_veto else "[red]✗[/]"
            hurst_col = "[green]✓[/]" if hurst_ok else "[red]✗[/]"
            vpin_col = "[green]✓[/]" if vpin_ok else "[red]✗[/]"
            go_col = "[bold bright_green]🚀[/]" if should_go else "[dim]—[/]"
            
            signals_table.add_row(
                row_emoji,
                sig.symbol.replace("USDT", ""),
                f"[{row_style}]{sig.ignition_score:.0f}[/]",
                f"{arch_emoji}{sig.archetype.value[:9]}",
                f"[{rvol_color}]{rvol_val:.1f}x[/]",
                f"[{cvd_color}]{cvd_val:+.1f}[/]",
                gates_str,
                f"[{fr_color}]{fr_pct:.2f}[/]",
                score_col,
                veto_col,
                hurst_col,
                vpin_col,
                go_col
            )
        
        layout["signals"].update(Panel(signals_table, title="🎯 Top Signals", border_style="green"))
        
        # ═══════════════════════════════════════════════════════════════════
        # FLIGHT CONTROL - Live Trade Monitoring Dashboard
        # ═══════════════════════════════════════════════════════════════════
        import random  # For blinking animation effect
        blink_char = "●" if int(time.time()) % 2 == 0 else "○"
        
        flight_table = Table(box=box.DOUBLE_EDGE, expand=True, show_header=True, header_style="bold white on dark_blue", 
                            title_style="bold white", border_style="blue")
        flight_table.add_column("✈️", width=3, justify="center")  # Flight status
        flight_table.add_column("SYMBOL", width=10, style="bold")
        flight_table.add_column("ENTRY", justify="right", width=10)
        flight_table.add_column("CURRENT", justify="right", width=10)
        flight_table.add_column("P&L", justify="right", width=8)
        flight_table.add_column("━━ SL ━━", justify="center", width=10)
        flight_table.add_column("━ TP1 ━", justify="center", width=6)
        flight_table.add_column("━ TP2 ━", justify="center", width=6)
        flight_table.add_column("━ TP3 ━", justify="center", width=6)
        flight_table.add_column("STATUS", width=12, justify="center")
        
        if self.active_trades:
            for trade in self.active_trades[:10]:
                entry_price = trade.get('entry_price', 0)
                # Simulate current price movement for visual (in real use, fetch live price)
                current_price = entry_price  # Would be live price in production
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                # Flight status indicator with animation
                if trade.get('hit_tp3'):
                    flight_icon = "🏆"
                    status_text = "[bold bright_green]LANDED TP3[/]"
                    row_style = "on dark_green"
                elif trade.get('hit_tp2'):
                    flight_icon = "🎯"
                    status_text = "[bold green]LANDED TP2[/]"
                    row_style = "on green"
                elif trade.get('hit_tp1'):
                    flight_icon = "✅"
                    status_text = "[green]LANDED TP1[/]"
                    row_style = ""
                elif trade.get('status') == 'ACTIVE':
                    flight_icon = f"[bold cyan]{blink_char}[/]"
                    status_text = f"[bold cyan]✈️ IN FLIGHT[/]"
                    row_style = ""
                elif 'CLOSED_SL' in str(trade.get('status', '')):
                    flight_icon = "💥"
                    status_text = "[bold red]CRASHED SL[/]"
                    row_style = "on dark_red"
                else:
                    flight_icon = "⏳"
                    status_text = "[yellow]BOARDING[/]"
                    row_style = ""
                
                # Price levels with visual progress bars
                sl_price = trade.get('stop_loss', 0)
                tp1_price = trade.get('tp1_price', 0)
                tp2_price = trade.get('tp2_price', 0)
                tp3_price = trade.get('tp3_price', 0)
                
                # SL distance indicator
                if sl_price and entry_price:
                    sl_dist = ((entry_price - sl_price) / entry_price * 100)
                    sl_display = f"[red]${sl_price:.4f}[/]"
                else:
                    sl_display = "[dim]-[/]"
                
                # TP Progress indicators with checkmarks
                tp1_hit = trade.get('hit_tp1', False)
                tp2_hit = trade.get('hit_tp2', False)
                tp3_hit = trade.get('hit_tp3', False)
                
                tp1_display = "[bright_green]✓ HIT[/]" if tp1_hit else f"[dim]${tp1_price:.2f}[/]" if tp1_price else "[dim]-[/]"
                tp2_display = "[bright_green]✓ HIT[/]" if tp2_hit else f"[dim]${tp2_price:.2f}[/]" if tp2_price else "[dim]-[/]"
                tp3_display = "[bright_green]✓ HIT[/]" if tp3_hit else f"[dim]${tp3_price:.2f}[/]" if tp3_price else "[dim]-[/]"
                
                # P&L color
                pnl_color = "bright_green" if pnl_pct > 0 else "red" if pnl_pct < 0 else "white"
                pnl_display = f"[{pnl_color}]{pnl_pct:+.2f}%[/]"
                
                flight_table.add_row(
                    flight_icon,
                    trade.get('symbol', '').replace('USDT', ''),
                    f"[white]${entry_price:.4f}[/]",
                    f"[cyan]${current_price:.4f}[/]",
                    pnl_display,
                    sl_display,
                    tp1_display,
                    tp2_display,
                    tp3_display,
                    status_text,
                    style=row_style
                )
        else:
            # No active trades - show waiting message with animation
            flight_table.add_row(
                f"[dim]{blink_char}[/]",
                "[dim]Awaiting[/]",
                "[dim]signals[/]",
                "[dim]...[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                f"[dim yellow]SCANNING {blink_char}[/]"
            )
        
        # Flight control panel title with live count
        active_count = sum(1 for t in self.active_trades if t.get('status') == 'ACTIVE')
        landed_count = sum(1 for t in self.active_trades if t.get('hit_tp1') or t.get('hit_tp2') or t.get('hit_tp3'))
        crashed_count = sum(1 for t in self.active_trades if 'SL' in str(t.get('status', '')))
        
        flight_title = f"✈️ FLIGHT CONTROL │ In Flight: {active_count} │ Landed: {landed_count} │ Crashed: {crashed_count}"
        layout["flight_control"].update(Panel(flight_table, title=flight_title, border_style="blue"))
        
        # ═══════════════════════════════════════════════════════════════════
        # ARCHETYPE LEGEND - What patterns we're scanning for
        # ═══════════════════════════════════════════════════════════════════
        # Count archetypes in current signals
        arch_counts = {}
        for sig in self.top_signals:
            arch = sig.archetype.value
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
        
        arch_text = Text()
        arch_text.append("PATTERNS SCANNING:\n", style="bold white")
        
        # Archetype definitions with emojis and current counts
        archetypes_info = [
            ("💎", "FRESH BREAK", "FRESH BREAK", "Breakout + strong volume + CVD/OFI confirming", "bright_cyan"),
            ("🌀", "COIL SQUEEZE", "COIL SQUEEZE", "BB inside KC + quiet vol + OI building up", "magenta"),
            ("🎭", "HYPE PUMP", "HYPE PUMP", "Leverage pump + OI surge + CVD accelerating", "bright_magenta"),
            ("⚡", "EXTREME VOL", "EXTREME VOL", "RVOL 5x+ explosion + massive move (>5%)", "bright_yellow"),
            ("🔄", "DIP BOUNCE", "DIP BOUNCE", "Recovery after -3%+ dip + bid support", "cyan"),
            ("🥦", "RANGE COIL", "RANGE COIL", "Tight range + squeeze + low vol pre-break", "green"),
        ]
        
        for emoji, name, key, desc, color in archetypes_info:
            count = arch_counts.get(key, 0)
            count_style = "bold bright_green" if count > 0 else "dim"
            arch_text.append(f"{emoji} ", style=color)
            arch_text.append(f"{name}", style=f"bold {color}")
            arch_text.append(f" [{count}]", style=count_style)
            arch_text.append(f": {desc}\n", style="dim")
        
        layout["archetypes"].update(Panel(arch_text, title="🔬 Pattern Scanner Guide", border_style="magenta"))
        
        # ═══════════════════════════════════════════════════════════════════
        # FOOTER - Legend for columns
        # ═══════════════════════════════════════════════════════════════════
        footer_text = Text()
        # Strength legend
        footer_text.append("Str: ", style="bold white")
        footer_text.append("🔥90+ ", style="bright_green")
        footer_text.append("🚀80+ ", style="green")
        footer_text.append("⚡70+ ", style="yellow")
        footer_text.append("│ ", style="dim")
        # VCIX legend
        footer_text.append("VCIX: ", style="bold white")
        footer_text.append("V", style="green")
        footer_text.append("iable ", style="dim")
        footer_text.append("C", style="green")
        footer_text.append("omp ", style="dim")
        footer_text.append("I", style="green")
        footer_text.append("gnite ", style="dim")
        footer_text.append("X", style="green")
        footer_text.append("ec", style="dim")
        footer_text.append(" │ ", style="dim")
        # Entry conditions legend
        footer_text.append("ENTRY: ", style="bold white")
        footer_text.append("≥70", style="cyan")
        footer_text.append("=Score ", style="dim")
        footer_text.append("NoV", style="cyan")
        footer_text.append("=NoVeto ", style="dim")
        footer_text.append("Hst", style="cyan")
        footer_text.append("=Hurst ", style="dim")
        footer_text.append("VPN", style="cyan")
        footer_text.append("=VPIN ", style="dim")
        footer_text.append("🚀", style="bright_green")
        footer_text.append("=GO!", style="dim")
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return layout
    
    async def run(self):
        """Main run loop"""
        await self.initialize()
        
        console.print("[bold cyan]🔱⚡ GOD OF GODS v5.0 - RESEARCH EDITION[/bold cyan]")
        console.print("[dim]18 upgrades from merged_squeeze.md research[/dim]\n")
        
        with Live(self.generate_ui(), refresh_per_second=1, console=console) as live:
            try:
                while True:
                    await self.scan_market()
                    live.update(self.generate_ui())
                    await asyncio.sleep(CONFIG.SCAN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down...[/yellow]")
            finally:
                await self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    scanner = GodOfGodsV5()
    asyncio.run(scanner.run())


if __name__ == "__main__":
    main()
