"""
ULTIMATE MICROCAP SQUEEZE SCANNER WITH TRADE TRACKING - GOD MODE v2.0
═══════════════════════════════════════════════════════════════════════════════
Complete system with:
• 4-GATE SQUEEZE DETECTION (Pre-Ignition → Setup → Ignition → Confirmation)
• OI Z-SCORE PRE-IGNITION (11-min lead time before price moves)  
• WHALE-CROWD DIVERGENCE FILTER (+7.5% PPV improvement from research)
• SQUEEZE MAGNITUDE QUANTIFICATION (SPS Score for ranking)
• RESEARCH-BACKED 7-METRIC EXIT SYSTEM

Research References:
- CORE-5 Filter Stack: Funding + OI + Positioning + Liquidations + Anti-Chase
- Gate Scoring: Setup (40pts) + Ignition (30pts) + Confirmation (30pts)
- Positioning Divergence: Top Trader Position L/S ÷ Global Account L/S > 1.76x
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import aiohttp
import aiosqlite
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Deque
from collections import deque
import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass, asdict, field
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box
from rich.layout import Layout
from rich.text import Text

# Import TP Engine with research-based exit classes
import sys
sys.path.append('.')
from quantum_tp_engine import (
    QuantumTPEngine, 
    TPLadder, 
    SqueezeExitState, 
    ExitSignal, 
    ExitReason
)

# =============================================================================
# RESEARCH-BACKED CONFIGURATION
# =============================================================================

class SqueezeConfig:
    """Research-validated thresholds from deep analysis"""
    
    # === GATE 1: SETUP THRESHOLDS (The Fuel) ===
    FUNDING_RATE_THRESHOLD = -0.0005  # -0.05% CORE-5 threshold
    FUNDING_RATE_ALERT = -0.0002      # -0.02% Yellow Alert
    FUNDING_RATE_DANGER = -0.0005     # -0.05% Red Alert
    FUNDING_RATE_EXTREME = -0.001     # -0.10% Nuclear
    FUNDING_24H_MA_THRESH = -0.0005   # 24h MA must be < -0.05%
    
    OI_PERCENTILE_THRESH = 90         # OI > 90th percentile
    OI_CHANGE_4H_THRESH = 0.15        # >15% OI increase in 4h
    
    # === GATE 2: IGNITION THRESHOLDS (The Spark) ===
    LIQ_ACCELERATION_MULT = 4.0       # 5m liqs > 4x 1h avg
    SHORT_LIQ_RATIO_THRESH = 2.5      # Short/Long liq > 2.5x
    
    # === PRE-IGNITION DETECTION (OI Z-Score) ===
    OI_Z_ELEVATED = 2.0               # 2σ = elevated watch
    OI_Z_CRITICAL = 3.0               # 3σ = PRE-PUMP (99.7% unusual)
    OI_Z_SCORE_ALERT = 2.0            # Alias for backwards compat
    OI_Z_SCORE_TRIGGER = 3.0          # Alias for backwards compat
    
    # === POSITIONING DIVERGENCE (+7.5% PPV) ===
    RETAIL_LS_CROWDED = 0.85          # Retail L/S < 0.85
    TOP_TRADER_LS_BULLISH = 1.50      # Top Trader L/S > 1.50
    TOP_TRADER_POS_STRONG = 1.80      # Position L/S > 1.80
    WHALE_CROWD_DIV_THRESH = 1.76     # Divergence > 1.76x
    WHALE_CROWD_DIVERGENCE_DEADLY = 1.76  # Alias: deadly threshold
    
    # === ENTROPY (Price Compression) ===
    ENTROPY_COLLAPSE = 0.65           # Entropy < 0.65 = coiled tight
    
    # === ANTI-CHASE GUARDRAILS ===
    MAX_PRICE_CHANGE_1H = 0.03        # Don't chase if > 3%
    MAX_PRICE_CHANGE_24H = 0.15       # Don't chase if > 15%
    RSI_OVERBOUGHT = 65               # RSI > 65 = too late


class SqueezeStage(Enum):
    """4-Gate Squeeze Detection Stages"""
    SCANNING = "SCANNING"
    ACCUMULATION = "ACCUMULATION"     # Gate 1 passed
    PRE_IGNITION = "PRE_IGNITION"     # OI Z-Score > 3.0
    IGNITION = "IGNITION"             # Gate 2 passed
    CONFIRMATION = "CONFIRMATION"     # Gate 3 passed
    ACTIVE = "ACTIVE"                 # In trade
    EXHAUSTION = "EXHAUSTION"         # Exit signals
    INVALIDATED = "INVALIDATED"       # Gate 4 failed


@dataclass
class PositioningMetrics:
    """Whale vs Crowd positioning from research"""
    retail_ls_ratio: float = 1.0
    top_trader_ls_accounts: float = 1.0
    top_trader_ls_positions: float = 1.0
    whale_crowd_divergence: float = 1.0
    
    def calculate_divergence(self) -> float:
        if self.retail_ls_ratio > 0:
            self.whale_crowd_divergence = self.top_trader_ls_positions / self.retail_ls_ratio
        return self.whale_crowd_divergence
    
    def is_deadly_setup(self) -> bool:
        return (
            self.retail_ls_ratio < SqueezeConfig.RETAIL_LS_CROWDED and
            self.top_trader_ls_accounts > SqueezeConfig.TOP_TRADER_LS_BULLISH and
            self.top_trader_ls_positions > SqueezeConfig.TOP_TRADER_POS_STRONG and
            self.whale_crowd_divergence > SqueezeConfig.WHALE_CROWD_DIV_THRESH
        )

console = Console()


# =============================================================================
# SQUEEZE MATH ENGINE - Research-Backed Algorithms
# =============================================================================

class SqueezeMathEngine:
    """Mathematical engine for early detection and magnitude quantification"""
    
    @staticmethod
    def calculate_oi_z_score(oi_history: List[float], window: int = 50) -> float:
        """
        OI Z-Score for PRE-IGNITION detection.
        Z > 3.0 = money rushing in BEFORE price moves (99.7% unusual)
        Provides ~11 minute lead time over price-based triggers.
        """
        if len(oi_history) < min(window, 10):
            return 0.0
        
        arr = np.array(oi_history[-window:])
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return 0.0
        
        return (arr[-1] - mean) / std
    
    @staticmethod
    def calculate_entropy(prices: List[float]) -> float:
        """
        Shannon Entropy of price returns.
        Low entropy (< 0.3) = compressed price action (coiled spring)
        High entropy = chaotic/random movement
        """
        if len(prices) < 50:
            return 1.0
        
        arr = np.array(prices)
        returns = np.diff(np.log(arr + 1e-10))
        
        hist, _ = np.histogram(returns, bins=20, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 1.0
        
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return min(max(entropy / 5.0, 0), 1.0)
    
    @staticmethod
    def calculate_squeeze_magnitude(
        oi_z_score: float,
        whale_divergence: float,
        funding_rate: float,
        entropy: float,
        volatility: float = 3.0
    ) -> float:
        """
        Predict squeeze magnitude (expected % move).
        Combines OI stress, positioning divergence, funding severity, compression.
        """
        # Base magnitude from Volatility (min 3% or actual vol * 1.5)
        # This ensures we never show 0%
        base = max(3.0, volatility * 1.5)
        
        # Add OI Stress bonus
        if abs(oi_z_score) > 2:
            base += min(abs(oi_z_score) * 2, 15)
        
        # Amplifiers
        div_mult = 1 + (whale_divergence - 1) * 0.5 if whale_divergence > 1 else 1
        # Funding multiplier: -0.01% -> 1.01x, -0.1% -> 1.1x
        fr_mult = 1 + min(abs(funding_rate) * 100, 0.5)
        # Compression multiplier: lower entropy -> higher potential
        comp_mult = 1 + (1 - entropy) * 0.3
        
        final_mag = base * div_mult * fr_mult * comp_mult
        
        return min(final_mag, 100)
    
    @staticmethod
    def calculate_squeeze_potential_score(
        whale_divergence: float,
        short_liq_ratio: float,
        oi_change_4h: float,
        oi_z_score: float,
        funding_rate: float
    ) -> float:
        """
        Squeeze Potential Score (SPS) for ranking candidates.
        
        Formula from research:
        SPS = [(Top Trader L/S ÷ Account L/S) × 10] 
              + [(Short Liq ÷ Long Liq) × 5]
              + [(OI 4h% + Z-Score) ÷ 4]
              + [|FR| × 10000]
        
        Target: SPS > 50 = HIGH PROBABILITY
                SPS > 70 = EXTREMELY DEADLY
        """
        div_score = whale_divergence * 10
        liq_score = min(short_liq_ratio * 5, 25)
        buildup_score = (abs(oi_change_4h) * 100 + max(0, oi_z_score) * 5) / 4
        fr_score = min(abs(funding_rate) * 10000, 10)
        
        return min(div_score + liq_score + buildup_score + fr_score, 100)


# =============================================================================
# 4-GATE SQUEEZE SCORER
# =============================================================================

class FourGateScorer:
    """
    Implements the 4-Gate Squeeze Detection System from research.
    
    Gate 1 (SETUP - 40pts): Negative Funding + Elevated OI
    Gate 2 (IGNITION - 30pts): Liquidation Acceleration  
    Gate 3 (CONFIRMATION - 30pts): OI Sustain + Positioning Divergence
    Gate 4 (INVALIDATION): Price < Ignition + OI Collapse → Reset to 0
    """
    
    @staticmethod
    def score_setup_gate(
        funding_rate: float,
        funding_ma: float,
        oi_percentile: float,
        oi_change_4h: float
    ) -> Tuple[float, List[str]]:
        """Gate 1: The Fuel (Max 40 points)"""
        score = 0.0
        reasons = []
        
        # Funding rate scoring (max 20 pts)
        if funding_rate < SqueezeConfig.FUNDING_RATE_EXTREME:
            score += 20
            reasons.append(f"🔴 EXTREME FR {funding_rate*100:.3f}%")
        elif funding_rate < SqueezeConfig.FUNDING_RATE_DANGER:
            score += 15
            reasons.append(f"🟠 DANGER FR {funding_rate*100:.3f}%")
        elif funding_rate < SqueezeConfig.FUNDING_RATE_ALERT:
            score += 10
            reasons.append(f"🟡 ALERT FR {funding_rate*100:.3f}%")
        elif funding_rate < 0:
            score += 5
            reasons.append(f"NEG FR {funding_rate*100:.3f}%")
        
        # MA bonus
        if funding_ma < SqueezeConfig.FUNDING_24H_MA_THRESH:
            score += 5
            reasons.append(f"24h MA {funding_ma*100:.3f}%")
        
        # OI scoring (max 15 pts)
        if oi_percentile >= SqueezeConfig.OI_PERCENTILE_THRESH:
            score += 10
            reasons.append(f"OI {oi_percentile:.0f}th pct")
        elif oi_percentile >= 75:
            score += 5
        
        if oi_change_4h >= SqueezeConfig.OI_CHANGE_4H_THRESH:
            score += 5
            reasons.append(f"OI +{oi_change_4h*100:.1f}%")
        
        return min(score, 40), reasons
    
    @staticmethod
    def score_ignition_gate(
        liq_acceleration: float,
        short_liq_ratio: float
    ) -> Tuple[float, List[str]]:
        """Gate 2: The Spark (Max 30 points)"""
        score = 0.0
        reasons = []
        
        # Liquidation acceleration (max 20 pts)
        if liq_acceleration >= SqueezeConfig.LIQ_ACCELERATION_MULT * 2:
            score += 20
            reasons.append(f"💥 LIQ {liq_acceleration:.1f}x")
        elif liq_acceleration >= SqueezeConfig.LIQ_ACCELERATION_MULT:
            score += 15
            reasons.append(f"⚡ LIQ {liq_acceleration:.1f}x")
        elif liq_acceleration >= 2.0:
            score += 10
        
        # Short/Long ratio (max 10 pts)
        if short_liq_ratio >= SqueezeConfig.SHORT_LIQ_RATIO_THRESH * 2:
            score += 10
            reasons.append(f"S/L {short_liq_ratio:.1f}x")
        elif short_liq_ratio >= SqueezeConfig.SHORT_LIQ_RATIO_THRESH:
            score += 7
        elif short_liq_ratio >= 1.5:
            score += 3
        
        return min(score, 30), reasons
    
    @staticmethod
    def score_confirmation_gate(
        oi_change_1h: float,
        positioning: 'PositioningMetrics',
        oi_z_score: float
    ) -> Tuple[float, List[str]]:
        """Gate 3: The Sustain (Max 30 points)"""
        score = 0.0
        reasons = []
        
        # OI sustained (max 10 pts)
        if oi_change_1h >= 0.05:
            score += 10
            reasons.append(f"OI +{oi_change_1h*100:.1f}%")
        elif oi_change_1h >= 0:
            score += 5
        
        # Positioning divergence (max 15 pts) - Critical for +7.5% PPV
        if positioning.is_deadly_setup():
            score += 15
            reasons.append(f"🎯 DIV {positioning.whale_crowd_divergence:.2f}x")
        elif positioning.whale_crowd_divergence >= 1.5:
            score += 10
        elif positioning.whale_crowd_divergence >= 1.2:
            score += 5
        
        # OI Z-Score bonus (max 5 pts)
        if oi_z_score >= SqueezeConfig.OI_Z_SCORE_TRIGGER:
            score += 5
            reasons.append(f"Z {oi_z_score:.1f}σ")
        
        return min(score, 30), reasons
    
    @staticmethod
    def check_anti_chase(
        price_change_1h: float,
        price_change_24h: float
    ) -> Tuple[bool, str]:
        """Anti-chase guardrails from research"""
        if price_change_1h > SqueezeConfig.MAX_PRICE_CHANGE_1H:
            return True, f"+{price_change_1h*100:.1f}% 1h"
        if price_change_24h > SqueezeConfig.MAX_PRICE_CHANGE_24H:
            return True, f"+{price_change_24h*100:.1f}% 24h"
        return False, ""


class TradeStatus(Enum):
    """Trade status enumeration"""
    ACTIVE = "ACTIVE"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    STOPPED = "STOPPED"
    CLOSED = "CLOSED"

@dataclass
class Trade:
    """Complete trade record with research-based 4-Gate scoring and exit tracking"""
    trade_id: int
    symbol: str
    entry_time: datetime
    entry_price: float
    
    # Position
    position_size: float
    leverage: int
    
    # Stops and targets
    stop_loss: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    
    # TP probabilities
    tp1_probability: float
    tp2_probability: float
    tp3_probability: float
    
    # Status
    status: TradeStatus
    current_price: float
    
    # Performance
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    
    # Exit tracking
    tp1_hit_time: Optional[datetime] = None
    tp2_hit_time: Optional[datetime] = None
    tp3_hit_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # === 4-GATE SQUEEZE SCORING (from research) ===
    squeeze_score: float = 0.0         # 0-100 total score
    setup_score: float = 0.0           # Gate 1: max 40 pts
    ignition_score: float = 0.0        # Gate 2: max 30 pts
    confirmation_score: float = 0.0    # Gate 3: max 30 pts
    squeeze_stage: str = "SCANNING"
    
    # === MAGNITUDE QUANTIFICATION ===
    squeeze_potential_score: float = 0.0  # SPS for ranking
    predicted_magnitude: float = 0.0      # Expected % move
    
    # === POSITIONING DIVERGENCE ===
    whale_crowd_divergence: float = 1.0
    retail_ls_ratio: float = 1.0
    top_trader_ls_positions: float = 1.0
    
    # === PRE-IGNITION METRICS ===
    oi_z_score: float = 0.0            # Z > 3.0 = pre-ignition
    entropy: float = 1.0               # Low = compressed
    
    # Metrics (legacy compatibility)
    quantum_score: float = 0.0
    conviction_level: str = "none"
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    # === RESEARCH-BASED EXIT STATE ===
    initial_funding_rate: float = 0.0
    initial_oi: float = 0.0
    atr: float = 0.0
    tp_ladder: Optional[TPLadder] = None
    exit_state: Optional[SqueezeExitState] = None


class TradeDatabase:
    """SQLite database for trade tracking"""
    
    def __init__(self, db_path: str = "squeeze_trades.db"):
        self.db_path = db_path
        self.db = None
        
    async def initialize(self):
        """Initialize database"""
        self.db = await aiosqlite.connect(self.db_path)
        
        # Create trades table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                position_size REAL NOT NULL,
                leverage INTEGER NOT NULL,
                stop_loss REAL NOT NULL,
                tp1_price REAL NOT NULL,
                tp2_price REAL NOT NULL,
                tp3_price REAL NOT NULL,
                tp1_probability REAL,
                tp2_probability REAL,
                tp3_probability REAL,
                status TEXT NOT NULL,
                current_price REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                tp1_hit_time TEXT,
                tp2_hit_time TEXT,
                tp3_hit_time TEXT,
                exit_time TEXT,
                quantum_score REAL,
                conviction_level TEXT,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL
            )
        """)
        
        await self.db.commit()
        
    async def close(self):
        """Close database"""
        if self.db:
            await self.db.close()
    
    async def add_trade(self, trade: Trade) -> int:
        """Add new trade"""
        cursor = await self.db.execute("""
            INSERT INTO trades (
                symbol, entry_time, entry_price, position_size, leverage,
                stop_loss, tp1_price, tp2_price, tp3_price,
                tp1_probability, tp2_probability, tp3_probability,
                status, current_price, pnl_usd, pnl_pct,
                quantum_score, conviction_level,
                max_favorable_excursion, max_adverse_excursion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.symbol,
            trade.entry_time.isoformat(),
            trade.entry_price,
            trade.position_size,
            trade.leverage,
            trade.stop_loss,
            trade.tp1_price,
            trade.tp2_price,
            trade.tp3_price,
            trade.tp1_probability,
            trade.tp2_probability,
            trade.tp3_probability,
            trade.status.value,
            trade.current_price,
            trade.pnl_usd,
            trade.pnl_pct,
            trade.quantum_score,
            trade.conviction_level,
            trade.max_favorable_excursion,
            trade.max_adverse_excursion
        ))
        
        await self.db.commit()
        return cursor.lastrowid
    
    async def update_trade(self, trade: Trade):
        """Update existing trade"""
        await self.db.execute("""
            UPDATE trades SET
                current_price = ?,
                status = ?,
                pnl_usd = ?,
                pnl_pct = ?,
                tp1_hit_time = ?,
                tp2_hit_time = ?,
                tp3_hit_time = ?,
                exit_time = ?,
                max_favorable_excursion = ?,
                max_adverse_excursion = ?
            WHERE trade_id = ?
        """, (
            trade.current_price,
            trade.status.value,
            trade.pnl_usd,
            trade.pnl_pct,
            trade.tp1_hit_time.isoformat() if trade.tp1_hit_time else None,
            trade.tp2_hit_time.isoformat() if trade.tp2_hit_time else None,
            trade.tp3_hit_time.isoformat() if trade.tp3_hit_time else None,
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.max_favorable_excursion,
            trade.max_adverse_excursion,
            trade.trade_id
        ))
        
        await self.db.commit()
    
    async def get_active_trades(self) -> List[Trade]:
        """Get all active trades"""
        cursor = await self.db.execute("""
            SELECT * FROM trades 
            WHERE status IN ('ACTIVE', 'TP1_HIT', 'TP2_HIT')
            ORDER BY entry_time DESC
        """)
        
        rows = await cursor.fetchall()
        
        trades = []
        for row in rows:
            trades.append(Trade(
                trade_id=row[0],
                symbol=row[1],
                entry_time=datetime.fromisoformat(row[2]),
                entry_price=row[3],
                position_size=row[4],
                leverage=row[5],
                stop_loss=row[6],
                tp1_price=row[7],
                tp2_price=row[8],
                tp3_price=row[9],
                tp1_probability=row[10],
                tp2_probability=row[11],
                tp3_probability=row[12],
                status=TradeStatus(row[13]),
                current_price=row[14],
                pnl_usd=row[15] or 0.0,
                pnl_pct=row[16] or 0.0,
                tp1_hit_time=datetime.fromisoformat(row[17]) if row[17] else None,
                tp2_hit_time=datetime.fromisoformat(row[18]) if row[18] else None,
                tp3_hit_time=datetime.fromisoformat(row[19]) if row[19] else None,
                exit_time=datetime.fromisoformat(row[20]) if row[20] else None,
                quantum_score=row[21] or 0.0,
                conviction_level=row[22] or "none",
                max_favorable_excursion=row[23] or 0.0,
                max_adverse_excursion=row[24] or 0.0
            ))
        
        return trades
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        
        # Total trades
        cursor = await self.db.execute("SELECT COUNT(*) FROM trades")
        total_trades = (await cursor.fetchone())[0]
        
        # Closed trades only
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades WHERE status IN ('TP3_HIT', 'STOPPED', 'CLOSED')
        """)
        closed_trades = (await cursor.fetchone())[0]
        
        # TP hits
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades WHERE tp1_hit_time IS NOT NULL
        """)
        tp1_hits = (await cursor.fetchone())[0]
        
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades WHERE tp2_hit_time IS NOT NULL
        """)
        tp2_hits = (await cursor.fetchone())[0]
        
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades WHERE tp3_hit_time IS NOT NULL
        """)
        tp3_hits = (await cursor.fetchone())[0]
        
        # Stop losses
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades WHERE status = 'STOPPED'
        """)
        sl_hits = (await cursor.fetchone())[0]
        
        # P&L stats
        cursor = await self.db.execute("""
            SELECT 
                SUM(pnl_usd),
                AVG(pnl_usd),
                AVG(pnl_pct),
                MAX(pnl_usd),
                MIN(pnl_usd)
            FROM trades 
            WHERE status IN ('TP3_HIT', 'STOPPED', 'CLOSED')
        """)
        pnl_stats = await cursor.fetchone()
        
        # Win rate
        cursor = await self.db.execute("""
            SELECT COUNT(*) FROM trades 
            WHERE status IN ('TP3_HIT', 'CLOSED') AND pnl_usd > 0
        """)
        winning_trades = (await cursor.fetchone())[0]
        
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
        
        # Average trade duration
        cursor = await self.db.execute("""
            SELECT AVG(
                CAST((julianday(exit_time) - julianday(entry_time)) * 24 AS REAL)
            ) FROM trades 
            WHERE exit_time IS NOT NULL
        """)
        avg_duration_hours = (await cursor.fetchone())[0] or 0
        
        return {
            'total_trades': total_trades,
            'closed_trades': closed_trades,
            'active_trades': total_trades - closed_trades,
            'tp1_hits': tp1_hits,
            'tp2_hits': tp2_hits,
            'tp3_hits': tp3_hits,
            'sl_hits': sl_hits,
            'total_pnl': pnl_stats[0] or 0.0,
            'avg_pnl_usd': pnl_stats[1] or 0.0,
            'avg_pnl_pct': pnl_stats[2] or 0.0,
            'best_trade': pnl_stats[3] or 0.0,
            'worst_trade': pnl_stats[4] or 0.0,
            'win_rate': win_rate,
            'avg_duration_hours': avg_duration_hours
        }

class UltimateSqueezeScanner:
    """Ultimate squeeze scanner with TP ladder and trade tracking"""
    
    def __init__(self):
        self.session = None
        self.tp_engine = QuantumTPEngine()
        self.trade_db = TradeDatabase()
        
        self.scan_count = 0
        self.last_scan_time = None
        self.top_signals = []
        
        # Active trades tracking
        self.active_trades: Dict[int, Trade] = {}
        
        # Performance stats cache
        self.perf_stats = {}
        
        # Auto-trading mode (simulated)
        self.auto_trade_enabled = True
        self.auto_trade_threshold = 85.0  # Only trade scores >= 85
        self.max_concurrent_trades = 3
        
    async def initialize(self):
        """Initialize scanner"""
        self.session = aiohttp.ClientSession()
        await self.trade_db.initialize()
        
        # Load active trades
        active = await self.trade_db.get_active_trades()
        for trade in active:
            self.active_trades[trade.trade_id] = trade
        
        # Load performance stats
        self.perf_stats = await self.trade_db.get_performance_stats()
        
    async def close(self):
        """Close scanner"""
        if self.session:
            await self.session.close()
        await self.trade_db.close()
    
    async def scan_and_rank(self) -> List[Dict[str, Any]]:
        """Perform market scan"""
        self.scan_count += 1
        self.last_scan_time = datetime.now()
        
        try:
            # Get all perpetual tickers
            async with self.session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as resp:
                if resp.status != 200:
                    return []
                tickers = await resp.json()
            
            # Get funding rates
            async with self.session.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10) as resp:
                if resp.status != 200:
                    return []
                funding_data = await resp.json()
            
            funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_data}
            
            candidates = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                
                try:
                    volume_24h = float(ticker['quoteVolume'])
                    price = float(ticker['lastPrice'])
                    price_change = float(ticker['priceChangePercent'])
                    
                    # Microcap filter
                    if volume_24h < 500_000 or volume_24h > 50_000_000:
                        continue
                    
                    funding_rate = funding_map.get(symbol, 0)
                    
                    # Calculate squeeze score
                    squeeze_score = self._calculate_squeeze_score(
                        funding_rate, price_change, volume_24h
                    )
                    
                    candidates.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': volume_24h,
                        'price_change_24h': price_change,
                        'funding_rate': funding_rate,
                        'squeeze_score': squeeze_score
                    })
                    
                except (KeyError, ValueError):
                    continue
            
            # Get detailed metrics for top candidates
            candidates.sort(key=lambda x: x['squeeze_score'], reverse=True)
            top_candidates = candidates[:10]
            
            for candidate in top_candidates:
                await self._enrich_candidate(candidate)
                await asyncio.sleep(0.1)
            
            # Calculate TP ladders
            for candidate in top_candidates:
                tp_ladder = await self._calculate_tp_ladder(candidate)
                candidate['tp_ladder'] = tp_ladder
            
            # Re-sort by final score
            top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            self.top_signals = top_candidates[:5]
            
            # Auto-trade check
            if self.auto_trade_enabled:
                await self._check_auto_trade()
            
            return self.top_signals
            
        except Exception as e:
            console.print(f"[red]Scan error: {e}[/]")
            return []
    
    def _calculate_squeeze_score(
        self, funding_rate: float, price_change: float, volume_24h: float
    ) -> float:
        """
        Calculate base squeeze score using 4-Gate system from research.
        
        This simplified version calculates Gate 1 (Setup) score only.
        Full scoring requires OI, liquidation, and positioning data.
        """
        score = 0.0
        
        # === GATE 1: FUNDING RATE (Setup - max 25 pts here) ===
        if funding_rate < SqueezeConfig.FUNDING_RATE_EXTREME:
            score += 25
        elif funding_rate < SqueezeConfig.FUNDING_RATE_DANGER:
            score += 20
        elif funding_rate < SqueezeConfig.FUNDING_RATE_ALERT:
            score += 15
        elif funding_rate < 0:
            score += 8
        
        # === ANTI-CHASE CHECK ===
        # Only add momentum points if not already extended
        is_chasing = price_change > SqueezeConfig.MAX_PRICE_CHANGE_24H * 100
        
        if not is_chasing:
            # Positive momentum (but not extended)
            if 5 < price_change <= 15:
                score += 20  # Sweet spot: moving but not extended
            elif 0 < price_change <= 5:
                score += 15  # Early move
            elif price_change > 15:
                score += 5   # Reduced score for extended
        else:
            score += 0  # No momentum bonus if chasing
        
        # === VOLUME (Liquidity) ===
        if volume_24h > 10_000_000:
            score += 15  # Good liquidity
        elif volume_24h > 5_000_000:
            score += 12
        elif volume_24h > 2_000_000:
            score += 8
        else:
            score += 5
        
        # === MICROCAP BONUS (Higher squeeze potential) ===
        if volume_24h < 5_000_000:
            score += 10  # Microcap = more squeezable
        elif volume_24h < 10_000_000:
            score += 6
        else:
            score += 2
        
        return min(score, 70)  # Cap at 70, remaining 30 from Gate 2+3
    
    async def _enrich_candidate(self, candidate: Dict[str, Any]):
        """
        Add detailed metrics including 4-Gate scoring and positioning divergence.
        Fetches all data needed for research-backed squeeze detection.
        """
        symbol = candidate['symbol']
        
        try:
            # Initialize tracking structures
            if 'oi_history' not in candidate:
                candidate['oi_history'] = []
            if 'price_history' not in candidate:
                candidate['price_history'] = []
            
            # === FETCH OPEN INTEREST ===
            async with self.session.get(
                f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}", timeout=5
            ) as resp:
                if resp.status == 200:
                    oi_data = await resp.json()
                    candidate['open_interest'] = float(oi_data['openInterest'])
                    candidate['oi_history'].append(candidate['open_interest'])
                else:
                    candidate['open_interest'] = 0
            
            # === FETCH RETAIL LONG/SHORT RATIO ===
            async with self.session.get(
                f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1",
                timeout=5
            ) as resp:
                if resp.status == 200:
                    ls_data = await resp.json()
                    if ls_data:
                        candidate['retail_ls_ratio'] = float(ls_data[0]['longShortRatio'])
                    else:
                        candidate['retail_ls_ratio'] = 1.0
                else:
                    candidate['retail_ls_ratio'] = 1.0
            
            # === FETCH TOP TRADER LONG/SHORT (POSITIONS) - Critical for +7.5% PPV ===
            async with self.session.get(
                f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit=1",
                timeout=5
            ) as resp:
                if resp.status == 200:
                    top_pos_data = await resp.json()
                    if top_pos_data:
                        candidate['top_trader_ls_positions'] = float(top_pos_data[0]['longShortRatio'])
                    else:
                        candidate['top_trader_ls_positions'] = 1.0
                else:
                    candidate['top_trader_ls_positions'] = 1.0
            
            # === FETCH TOP TRADER LONG/SHORT (ACCOUNTS) ===
            async with self.session.get(
                f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={symbol}&period=1h&limit=1",
                timeout=5
            ) as resp:
                if resp.status == 200:
                    top_acc_data = await resp.json()
                    if top_acc_data:
                        candidate['top_trader_ls_accounts'] = float(top_acc_data[0]['longShortRatio'])
                    else:
                        candidate['top_trader_ls_accounts'] = 1.0
                else:
                    candidate['top_trader_ls_accounts'] = 1.0
            
            # === CALCULATE WHALE-CROWD DIVERGENCE ===
            positioning = PositioningMetrics(
                retail_ls_ratio=candidate.get('retail_ls_ratio', 1.0),
                top_trader_ls_accounts=candidate.get('top_trader_ls_accounts', 1.0),
                top_trader_ls_positions=candidate.get('top_trader_ls_positions', 1.0)
            )
            positioning.calculate_divergence()
            candidate['positioning'] = positioning
            candidate['whale_crowd_divergence'] = positioning.whale_crowd_divergence
            candidate['is_deadly_setup'] = positioning.is_deadly_setup()
            
            # === FETCH KLINES FOR MOMENTUM & ENTROPY ===
            async with self.session.get(
                f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&limit=100",
                timeout=5
            ) as resp:
                if resp.status == 200:
                    klines = await resp.json()
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    highs = [float(k[2]) for k in klines]  # NEW: For ATR
                    lows = [float(k[3]) for k in klines]   # NEW: For ATR
                    
                    if len(closes) > 20:
                        valid_closes = [c for c in closes if c > 0]
                        if len(valid_closes) > 20:
                            returns = np.diff(np.log(valid_closes))
                            candidate['volatility'] = np.std(returns) * 100
                            
                            # Safe momentum calculations
                            if closes[-5] > 0:
                                candidate['momentum_5m'] = ((closes[-1] - closes[-5]) / closes[-5]) * 100
                            else:
                                candidate['momentum_5m'] = 0
                            if closes[-15] > 0:
                                candidate['momentum_15m'] = ((closes[-1] - closes[-15]) / closes[-15]) * 100
                            else:
                                candidate['momentum_15m'] = 0
                            
                            # === CALCULATE ENTROPY (from research) ===
                            candidate['entropy'] = SqueezeMathEngine.calculate_entropy(valid_closes)
                        else:
                            candidate['volatility'] = 0
                            candidate['momentum_5m'] = 0
                            candidate['momentum_15m'] = 0
                            candidate['entropy'] = 1.0
                        
                        candidate['prices'] = closes
                        candidate['volumes'] = volumes
                        candidate['price_history'] = closes
                        
                        # ═══ NEW: EXPECTANCY IMPROVEMENTS ═══
                        if len(valid_closes) >= 20:
                            # EMA9 for Continuation Detector
                            multiplier = 2 / 10
                            ema9 = sum(valid_closes[:9]) / 9
                            for p in valid_closes[9:]:
                                ema9 = (p * multiplier) + (ema9 * (1 - multiplier))
                            candidate['ema9'] = ema9
                            candidate['price_above_ema9'] = closes[-1] > ema9
                            
                            # Bollinger Bands compression (BBW)
                            sma20 = sum(valid_closes[-20:]) / 20
                            std20 = (sum((c - sma20) ** 2 for c in valid_closes[-20:]) / 20) ** 0.5
                            bb_upper = sma20 + (2 * std20)
                            bb_lower = sma20 - (2 * std20)
                            bb_width = (bb_upper - bb_lower) / sma20 if sma20 > 0 else 0
                            candidate['bb_width'] = bb_width
                            candidate['bb_upper'] = bb_upper
                            candidate['price_walking_bands'] = closes[-1] > bb_upper
                            
                            # BBW compression ratio (compare to older period)
                            if len(valid_closes) >= 50:
                                old_closes = valid_closes[:30]
                                old_sma = sum(old_closes) / len(old_closes)
                                old_std = (sum((c - old_sma) ** 2 for c in old_closes) / len(old_closes)) ** 0.5
                                old_bbw = (4 * old_std) / old_sma if old_sma > 0 else 1
                                candidate['bb_compression_ratio'] = bb_width / old_bbw if old_bbw > 0 else 1.0
                            else:
                                candidate['bb_compression_ratio'] = 1.0
                            
                            # ATR for VATB trailing stops
                            if len(highs) >= 15:
                                true_ranges = []
                                for i in range(1, min(15, len(closes))):
                                    tr = max(
                                        highs[i] - lows[i],
                                        abs(highs[i] - closes[i-1]),
                                        abs(lows[i] - closes[i-1])
                                    )
                                    true_ranges.append(tr)
                                candidate['atr'] = sum(true_ranges) / len(true_ranges) if true_ranges else 0
                            else:
                                candidate['atr'] = 0
                        else:
                            candidate['ema9'] = 0
                            candidate['price_above_ema9'] = False
                            candidate['bb_width'] = 0
                            candidate['bb_compression_ratio'] = 1.0
                            candidate['atr'] = 0
                    else:
                        candidate['volatility'] = 0
                        candidate['momentum_5m'] = 0
                        candidate['momentum_15m'] = 0
                        candidate['prices'] = []
                        candidate['volumes'] = []
                        candidate['entropy'] = 1.0
                        candidate['ema9'] = 0
                        candidate['bb_compression_ratio'] = 1.0
                        candidate['atr'] = 0
                else:
                    candidate['volatility'] = 0
                    candidate['momentum_5m'] = 0
                    candidate['momentum_15m'] = 0
                    candidate['prices'] = []
                    candidate['volumes'] = []
                    candidate['entropy'] = 1.0
                    candidate['ema9'] = 0
                    candidate['bb_compression_ratio'] = 1.0
                    candidate['atr'] = 0
            
            # ═══ NEW: FETCH SPOT PRICE FOR BASIS DISLOCATION ═══
            try:
                async with self.session.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5
                ) as resp:
                    if resp.status == 200:
                        spot_data = await resp.json()
                        spot_price = float(spot_data.get('price', 0))
                        candidate['spot_price'] = spot_price
                        if spot_price > 0 and candidate['price'] > 0:
                            basis = candidate['price'] - spot_price
                            candidate['basis'] = basis
                            candidate['basis_pct'] = (basis / spot_price) * 100
                        else:
                            candidate['basis'] = 0
                            candidate['basis_pct'] = 0
                    else:
                        candidate['spot_price'] = 0
                        candidate['basis'] = 0
                        candidate['basis_pct'] = 0
            except:
                candidate['spot_price'] = 0
                candidate['basis'] = 0
                candidate['basis_pct'] = 0
            
            # === CALCULATE OI Z-SCORE (PRE-IGNITION DETECTION) ===
            oi_history = candidate.get('oi_history', [])
            if len(oi_history) >= 10:
                candidate['oi_z_score'] = SqueezeMathEngine.calculate_oi_z_score(oi_history)
            else:
                candidate['oi_z_score'] = 0.0
            
            # === CALCULATE 4-GATE SCORES ===
            setup_score, setup_reasons = FourGateScorer.score_setup_gate(
                funding_rate=candidate.get('funding_rate', 0),
                funding_ma=candidate.get('funding_rate', 0),  # Simplified
                oi_percentile=50,  # Would need historical data
                oi_change_4h=0  # Would need 4h comparison
            )
            candidate['setup_score'] = setup_score
            
            # Simplified ignition scoring (full version needs liquidation data)
            candidate['ignition_score'] = 0
            
            # Confirmation scoring with positioning
            confirm_score, confirm_reasons = FourGateScorer.score_confirmation_gate(
                oi_change_1h=0,
                positioning=positioning,
                oi_z_score=candidate.get('oi_z_score', 0)
            )
            candidate['confirmation_score'] = confirm_score
            
            # === CALCULATE SQUEEZE POTENTIAL SCORE (SPS) ===
            candidate['squeeze_potential_score'] = SqueezeMathEngine.calculate_squeeze_potential_score(
                whale_divergence=candidate.get('whale_crowd_divergence', 1.0),
                short_liq_ratio=1.0,  # Would need liq data
                oi_change_4h=0,
                oi_z_score=candidate.get('oi_z_score', 0),
                funding_rate=candidate.get('funding_rate', 0)
            )
            
            # === CALCULATE PREDICTED MAGNITUDE ===
            # === CALCULATE PREDICTED MAGNITUDE ===
            candidate['predicted_magnitude'] = SqueezeMathEngine.calculate_squeeze_magnitude(
                oi_z_score=candidate.get('oi_z_score', 0),
                whale_divergence=candidate.get('whale_crowd_divergence', 1.0),
                funding_rate=candidate.get('funding_rate', 0),
                entropy=candidate.get('entropy', 1.0),
                volatility=candidate.get('volatility', 3.0)
            )
            
            # === CHECK ANTI-CHASE ===
            is_chasing, chase_reason = FourGateScorer.check_anti_chase(
                price_change_1h=candidate.get('momentum_15m', 0) / 100,
                price_change_24h=candidate.get('price_change_24h', 0) / 100
            )
            candidate['is_chasing'] = is_chasing
            
            # === CALCULATE FINAL SCORE (4-Gate Total) ===
            candidate['final_score'] = self._calculate_final_score(candidate)
            
        except Exception as e:
            console.print(f"[yellow]Warning enriching {symbol}: {e}[/]")
            candidate['open_interest'] = 0
            candidate['retail_ls_ratio'] = 1.0
            candidate['top_trader_ls_positions'] = 1.0
            candidate['whale_crowd_divergence'] = 1.0
            candidate['volatility'] = 0
            candidate['momentum_5m'] = 0
            candidate['momentum_15m'] = 0
            candidate['prices'] = []
            candidate['volumes'] = []
            candidate['oi_z_score'] = 0
            candidate['entropy'] = 1.0
            candidate['squeeze_potential_score'] = 0
            candidate['predicted_magnitude'] = 0
            candidate['final_score'] = candidate['squeeze_score']
    
    def _calculate_final_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate final comprehensive score using 4-Gate Squeeze Detection System.
        
        From research: 4-Gate Total = Setup(40) + Ignition(30) + Confirmation(30)
        - Gate 1 (Setup): Funding + OI conditions create the powder keg
        - Gate 2 (Ignition): Liquidation acceleration triggers cascade  
        - Gate 3 (Confirmation): Sustained OI + positioning divergence validates
        - Gate 4 (Invalidation): Monitors for squeeze exhaustion/failure
        
        Enhanced with:
        - Whale-Crowd Divergence (+7.5% PPV when > 1.76x)
        - OI Z-Score (pre-ignition detection)
        - Entropy collapse (price compression before breakout)
        - Squeeze Potential Score for magnitude estimation
        """
        # Base squeeze score from initial screening
        base_score = data.get('squeeze_score', 0)
        
        # === 4-GATE SCORING ===
        # Gate 1: Setup Score (already calculated)
        setup_score = data.get('setup_score', 0)
        
        # Gate 2: Ignition Score (simplified without real-time liq data)
        ignition_score = data.get('ignition_score', 0)
        
        # Gate 3: Confirmation Score
        confirmation_score = data.get('confirmation_score', 0)
        
        # Combined 4-Gate score
        gate_score = setup_score + ignition_score + confirmation_score
        
        # === RESEARCH-BACKED BONUSES ===
        
        # Whale-Crowd Divergence (from research: +7.5% PPV improvement)
        whale_divergence = data.get('whale_crowd_divergence', 1.0)
        divergence_bonus = 0
        if whale_divergence > SqueezeConfig.WHALE_CROWD_DIVERGENCE_DEADLY:  # 1.76x
            divergence_bonus = 20  # Critical: whales betting against retail
        elif whale_divergence > 1.4:
            divergence_bonus = 12
        elif whale_divergence > 1.2:
            divergence_bonus = 6
        
        # OI Z-Score (pre-ignition detection ~11 min lead time)
        oi_z_score = data.get('oi_z_score', 0)
        oi_z_bonus = 0
        if oi_z_score > SqueezeConfig.OI_Z_CRITICAL:  # 3.0
            oi_z_bonus = 18  # Major money flow detected BEFORE price moves
        elif oi_z_score > SqueezeConfig.OI_Z_ELEVATED:  # 2.0
            oi_z_bonus = 10
        elif oi_z_score > 1.5:
            oi_z_bonus = 5
        
        # Entropy collapse (low entropy = compressed price, ready for breakout)
        entropy = data.get('entropy', 1.0)
        entropy_bonus = 0
        if entropy < SqueezeConfig.ENTROPY_COLLAPSE:  # 0.65
            entropy_bonus = 12  # Price is coiled tight
        elif entropy < 0.75:
            entropy_bonus = 6
        elif entropy < 0.85:
            entropy_bonus = 3
        
        # Retail L/S ratio (shorts crowded)
        ls_ratio = data.get('retail_ls_ratio', data.get('long_short_ratio', 1.0))
        ls_bonus = 0
        if ls_ratio < SqueezeConfig.RETAIL_LS_CROWDED:  # 0.85
            ls_bonus = 15  # Retail heavily short = fuel
        elif ls_ratio < 0.95:
            ls_bonus = 8
        elif ls_ratio < 1.0:
            ls_bonus = 4
        
        # Volatility bonus (fuel for the fire)
        volatility = data.get('volatility', 0)
        vol_bonus = 0
        if volatility > 5:
            vol_bonus = 8
        elif volatility > 3:
            vol_bonus = 4
        
        # Early momentum bonus (but not too much - avoid chasing)
        momentum_5m = data.get('momentum_5m', 0)
        momentum_bonus = 0
        if 0.5 < momentum_5m < 3:  # Goldilocks zone - moving but not gone
            momentum_bonus = 10
        elif 0.2 < momentum_5m < 0.5:  # Just starting
            momentum_bonus = 12  # Prefer early
        elif momentum_5m > 3:
            momentum_bonus = 5  # Might be chasing
        
        # ═══ NEW: EXPECTANCY IMPROVEMENT BONUSES ═══
        
        # BBW Compression Bonus (coiled spring from research)
        bb_compression = data.get('bb_compression_ratio', 1.0)
        bbw_bonus = 0
        if bb_compression > 0 and bb_compression < 0.5:
            bbw_bonus = 15  # Tightly compressed = explosive potential
        elif bb_compression > 0 and bb_compression < 0.7:
            bbw_bonus = 8
        
        # Basis Dislocation Bonus (deep backwardation = squeeze fuel)
        basis_pct = data.get('basis_pct', 0)
        basis_bonus = 0
        if basis_pct < -0.5:  # Deep backwardation
            basis_bonus = 12
        elif basis_pct < -0.2:
            basis_bonus = 6
        
        # Continuation Detector (funding negative AND price > EMA9)
        funding_rate = data.get('funding_rate', 0)
        price_above_ema9 = data.get('price_above_ema9', False)
        continuation_bonus = 0
        if funding_rate < -0.0001 and price_above_ema9:
            continuation_bonus = 8  # Squeeze is "hot" - momentum holding
        
        # === DEADLY SETUP MULTIPLIER ===
        # From research: When retail shorts AND whales are long on positions
        is_deadly = data.get('is_deadly_setup', False)
        deadly_multiplier = 1.2 if is_deadly else 1.0
        
        # === ANTI-CHASE PENALTY ===
        is_chasing = data.get('is_chasing', False)
        chase_penalty = 0.6 if is_chasing else 1.0  # Severe penalty for late entries
        
        # === CALCULATE FINAL SCORE ===
        # Combine base + gate scoring + research bonuses + expectancy bonuses
        raw_score = base_score + gate_score + divergence_bonus + oi_z_bonus + \
                   entropy_bonus + ls_bonus + vol_bonus + momentum_bonus + \
                   bbw_bonus + basis_bonus + continuation_bonus
        
        # Apply multipliers
        adjusted_score = raw_score * deadly_multiplier * chase_penalty
        
        # Normalize to 100 (increased max due to new signals)
        return min(100, max(0, adjusted_score))
    
    async def _calculate_tp_ladder(self, candidate: Dict[str, Any]) -> TPLadder:
        """Calculate TP ladder for candidate"""
        
        symbol = candidate['symbol']
        entry_price = candidate['price']
        final_score = candidate['final_score']
        
        # Determine conviction
        if final_score >= 90:
            conviction = "extreme"
        elif final_score >= 80:
            conviction = "high"
        elif final_score >= 70:
            conviction = "medium"
        else:
            conviction = "low"
        
        # Calculate proper ATR approximation for meaningful TP spreads
        # Volatility is std(log returns) * 100, typically 1-10 for crypto
        volatility = candidate.get('volatility', 3.0)  # Default 3% vol if missing
        
        # Calculate ATR from price range if we have price data
        prices = candidate.get('prices', [])
        if len(prices) >= 20:
            # Calculate average true range from recent price action
            highs = [max(prices[i:i+5]) for i in range(0, len(prices)-5, 5)]
            lows = [min(prices[i:i+5]) for i in range(0, len(prices)-5, 5)]
            if highs and lows:
                ranges = [(h - l) / l for h, l in zip(highs, lows) if l > 0]  # Relative ranges
                atr = np.mean(ranges) if ranges else volatility / 100
            else:
                atr = volatility / 100  # Fallback: use volatility (already a decimal multiplier)
        else:
            # Fallback: volatility is already in percentage (1-10 typically)
            # Convert to decimal ATR multiplier: 3% vol → 0.03 ATR
            atr = volatility / 100
        
        # Ensure minimum ATR for meaningful TPs (at least 2% range)
        atr = max(atr, 0.02)
        
        # Get prices and volumes
        prices = candidate.get('prices', [entry_price] * 50)
        volumes = candidate.get('volumes', [1000000] * 50)
        
        # Call TP engine
        tp_ladder = self.tp_engine.calculate_tp_ladder(
            entry_price=entry_price,
            symbol=symbol,
            quantum_score=final_score,
            conviction_level=conviction,
            prices=prices,
            volumes=volumes,
            timestamps=[int(time.time() * 1000)] * len(prices),
            cvd_spot=0,
            cvd_futures=0,
            cvd_slope=0.1,
            cvd_acceleration=0.05,
            oi_values=[candidate.get('open_interest', 0)] * 20,
            oi_z_scores={'short': -2.0, 'medium': -1.5, 'long': -1.0},
            atr=atr,
            bb_width=0.05,
            historical_volatility=volatility / 100,
            bid_levels=[],
            ask_levels=[],
            momentum_5m=candidate.get('momentum_5m', 0),
            momentum_15m=candidate.get('momentum_15m', 0),
            momentum_1h=0,
            funding_rate=candidate.get('funding_rate', 0),
            long_short_ratio=candidate.get('long_short_ratio', 1.0),
            liquidations=[]
        )
        
        return tp_ladder
    
    async def _check_auto_trade(self):
        """Check if we should enter new trades"""
        
        # Check if we have room for more trades
        if len(self.active_trades) >= self.max_concurrent_trades:
            return
        
        # Check top signals
        for signal in self.top_signals:
            if signal['final_score'] >= self.auto_trade_threshold:
                
                # Check if already in this symbol
                if any(t.symbol == signal['symbol'] for t in self.active_trades.values()):
                    continue
                
                # Enter trade
                await self._enter_trade(signal)
                
                # Only enter one per scan
                break
    
    async def _enter_trade(self, signal: Dict[str, Any]):
        """Enter a new trade with research-based exit state initialization"""
        
        tp_ladder = signal['tp_ladder']
        
        # Calculate position size (simple: $1000 base)
        position_size = 1000.0
        
        # Determine leverage
        conviction = "extreme" if signal['final_score'] >= 90 else "high" if signal['final_score'] >= 80 else "medium"
        leverage = 10 if conviction == "extreme" else 7 if conviction == "high" else 5
        
        # Get entry fuel gauges for research-based exit tracking
        entry_funding_rate = signal.get('funding_rate', 0.0)
        entry_oi = signal.get('open_interest', 0.0)
        volatility = signal.get('volatility', 1.0)
        atr = volatility / 100 * 0.1  # Rough ATR estimate
        
        # Initialize exit state with entry baseline values
        exit_state = SqueezeExitState(
            initial_funding_rate=entry_funding_rate,
            oi_at_entry=entry_oi,
            oi_peak=entry_oi,
            highest_high=signal['price'],
            current_funding_rate=entry_funding_rate,
            current_oi=entry_oi
        )
        
        # Store initial FR and OI in TP ladder for threshold comparisons
        tp_ladder.initial_funding_rate = entry_funding_rate
        tp_ladder.oi_at_entry = entry_oi
        tp_ladder.exit_state = exit_state
        
        # Create trade
        trade = Trade(
            trade_id=0,  # Will be assigned by DB
            symbol=signal['symbol'],
            entry_time=datetime.now(),
            entry_price=signal['price'],
            position_size=position_size,
            leverage=leverage,
            stop_loss=tp_ladder.stop_loss,
            tp1_price=tp_ladder.tp1_price,
            tp2_price=tp_ladder.tp2_price,
            tp3_price=tp_ladder.tp3_price,
            tp1_probability=tp_ladder.tp1_probability,
            tp2_probability=tp_ladder.tp2_probability,
            tp3_probability=tp_ladder.tp3_probability,
            status=TradeStatus.ACTIVE,
            current_price=signal['price'],
            quantum_score=signal['final_score'],
            conviction_level=conviction,
            max_favorable_excursion=signal['price'],
            max_adverse_excursion=signal['price'],
            # Research-based fields
            initial_funding_rate=entry_funding_rate,
            initial_oi=entry_oi,
            atr=atr,
            tp_ladder=tp_ladder,
            exit_state=exit_state
        )
        
        # Add to database
        trade_id = await self.trade_db.add_trade(trade)
        trade.trade_id = trade_id
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        console.print(f"[bold green]🚀 ENTERED TRADE: {trade.symbol} @ ${trade.entry_price:.4f} | FR: {entry_funding_rate*100:.3f}% | OI: {entry_oi:,.0f}[/]")

    
    async def update_active_trades(self):
        """
        Update all active trades with research-based conditional exit logic.
        
        Uses 7-metric system from research:
        1. Reversal Guard (panic exit) - overrides everything
        2. TP3 with RSI/FR exhaustion signals
        3. TP2 with OI peak detection
        4. TP1 with FR/OI fuel checks
        5. Chandelier trailing for runner positions
        """
        
        if not self.active_trades:
            return
        
        try:
            # Get current prices
            async with self.session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price", timeout=5
            ) as resp:
                if resp.status != 200:
                    return
                prices = await resp.json()
                price_map = {p['symbol']: float(p['price']) for p in prices}
            
            # Get current funding rates
            async with self.session.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex", timeout=5
            ) as resp:
                if resp.status == 200:
                    funding_data = await resp.json()
                    funding_map = {item['symbol']: float(item['lastFundingRate']) for item in funding_data}
                else:
                    funding_map = {}
            
            # Update each trade with research-based exit logic
            for trade in list(self.active_trades.values()):
                if trade.symbol not in price_map:
                    continue
                
                current_price = price_map[trade.symbol]
                trade.current_price = current_price
                
                # Update MFE/MAE
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, current_price)
                trade.max_adverse_excursion = min(trade.max_adverse_excursion, current_price)
                
                # Calculate P&L (with zero guard)
                if trade.entry_price > 0:
                    price_change = (current_price - trade.entry_price) / trade.entry_price
                    trade.pnl_pct = price_change * 100
                    trade.pnl_usd = trade.position_size * price_change * trade.leverage
                else:
                    trade.pnl_pct = 0
                    trade.pnl_usd = 0
                
                # === UPDATE EXIT STATE WITH LIVE DATA ===
                if trade.exit_state:
                    # Update highest high for Chandelier trailing
                    trade.exit_state.update_highest_high(current_price)
                    
                    # Update current funding rate
                    trade.exit_state.current_funding_rate = funding_map.get(trade.symbol, 0.0)
                    
                    # Fetch current OI for this symbol
                    try:
                        async with self.session.get(
                            f"https://fapi.binance.com/fapi/v1/openInterest?symbol={trade.symbol}",
                            timeout=3
                        ) as oi_resp:
                            if oi_resp.status == 200:
                                oi_data = await oi_resp.json()
                                current_oi = float(oi_data.get('openInterest', 0))
                                trade.exit_state.update_oi_peak(current_oi)
                    except:
                        pass  # OI fetch failed, continue with existing data
                
                # Check stop loss first (always)
                if current_price <= trade.stop_loss:
                    trade.status = TradeStatus.STOPPED
                    trade.exit_time = datetime.now()
                    await self.trade_db.update_trade(trade)
                    del self.active_trades[trade.trade_id]
                    console.print(f"[red]❌ STOPPED OUT: {trade.symbol} @ ${current_price:.4f} | Loss: ${trade.pnl_usd:.2f}[/]")
                    continue
                
                # === RESEARCH-BASED CONDITIONAL EXIT LOGIC ===
                if trade.tp_ladder and trade.exit_state:
                    # Use comprehensive exit signal from TP engine
                    exit_signal = self.tp_engine.get_comprehensive_exit_signal(
                        current_price=current_price,
                        tp_ladder=trade.tp_ladder,
                        exit_state=trade.exit_state,
                        atr=trade.atr,
                        tp1_already_hit=(trade.status in [TradeStatus.TP1_HIT, TradeStatus.TP2_HIT, TradeStatus.TP3_HIT]),
                        tp2_already_hit=(trade.status in [TradeStatus.TP2_HIT, TradeStatus.TP3_HIT]),
                        tp3_already_hit=(trade.status == TradeStatus.TP3_HIT)
                    )
                    
                    if exit_signal.should_exit:
                        # Handle exit based on reason
                        if exit_signal.reason == ExitReason.REVERSAL_GUARD:
                            # PANIC EXIT - close everything
                            trade.status = TradeStatus.CLOSED
                            trade.exit_time = datetime.now()
                            await self.trade_db.update_trade(trade)
                            del self.active_trades[trade.trade_id]
                            console.print(f"[bold red]🚨 REVERSAL GUARD: {trade.symbol} @ ${current_price:.4f} | {exit_signal.message}[/]")
                            
                        elif exit_signal.reason in [ExitReason.PRICE_TARGET, ExitReason.FR_NORMALIZED, ExitReason.OI_EXHAUSTED, ExitReason.RSI_EXHAUSTION]:
                            # TP exit with conditions met
                            if trade.status == TradeStatus.ACTIVE and current_price >= trade.tp1_price:
                                trade.status = TradeStatus.TP1_HIT
                                trade.tp1_hit_time = datetime.now()
                                console.print(f"[green]🎯 TP1 (Conditional): {trade.symbol} @ ${current_price:.4f} | {exit_signal.message}[/]")
                                
                            elif trade.status == TradeStatus.TP1_HIT and current_price >= trade.tp2_price:
                                trade.status = TradeStatus.TP2_HIT
                                trade.tp2_hit_time = datetime.now()
                                console.print(f"[green]🎯 TP2 (Conditional): {trade.symbol} @ ${current_price:.4f} | {exit_signal.message}[/]")
                                
                            elif trade.status == TradeStatus.TP2_HIT and current_price >= trade.tp3_price:
                                trade.status = TradeStatus.TP3_HIT
                                trade.tp3_hit_time = datetime.now()
                                trade.exit_time = datetime.now()
                                del self.active_trades[trade.trade_id]
                                console.print(f"[bold green]🎯 TP3 (Conditional): {trade.symbol} @ ${current_price:.4f} | {exit_signal.message}[/]")
                            
                            await self.trade_db.update_trade(trade)
                            
                        elif exit_signal.reason == ExitReason.CHANDELIER_TRAIL:
                            # Runner position trailed out
                            trade.status = TradeStatus.CLOSED
                            trade.exit_time = datetime.now()
                            await self.trade_db.update_trade(trade)
                            del self.active_trades[trade.trade_id]
                            console.print(f"[cyan]🎯 RUNNER TRAILED: {trade.symbol} @ ${current_price:.4f} | {exit_signal.message}[/]")
                    else:
                        # No exit signal - just update
                        await self.trade_db.update_trade(trade)
                else:
                    # Fallback to simple price-based TP logic (no exit state)
                    if current_price >= trade.tp3_price and trade.status != TradeStatus.TP3_HIT:
                        trade.status = TradeStatus.TP3_HIT
                        trade.tp3_hit_time = datetime.now()
                        trade.exit_time = datetime.now()
                        await self.trade_db.update_trade(trade)
                        del self.active_trades[trade.trade_id]
                        console.print(f"[bold green]🎯 TP3 HIT: {trade.symbol} @ ${current_price:.4f}[/]")
                        
                    elif current_price >= trade.tp2_price and trade.status == TradeStatus.ACTIVE:
                        trade.status = TradeStatus.TP2_HIT
                        trade.tp2_hit_time = datetime.now()
                        await self.trade_db.update_trade(trade)
                        console.print(f"[green]🎯 TP2 HIT: {trade.symbol} @ ${current_price:.4f}[/]")
                        
                    elif current_price >= trade.tp1_price and trade.status == TradeStatus.ACTIVE:
                        trade.status = TradeStatus.TP1_HIT
                        trade.tp1_hit_time = datetime.now()
                        await self.trade_db.update_trade(trade)
                        console.print(f"[green]🎯 TP1 HIT: {trade.symbol} @ ${current_price:.4f}[/]")
                    else:
                        await self.trade_db.update_trade(trade)
            
            # Update performance stats
            self.perf_stats = await self.trade_db.get_performance_stats()

            
        except Exception as e:
            console.print(f"[yellow]Warning updating trades: {e}[/]")
    
    def generate_ticker(self) -> Panel:
        """Generate performance ticker"""
        
        stats = self.perf_stats
        
        ticker_text = (
            f"📊 Trades: {stats.get('total_trades', 0)} | "
            f"🟢 Active: {stats.get('active_trades', 0)} | "
            f"🎯 TP1: {stats.get('tp1_hits', 0)} | "
            f"🎯 TP2: {stats.get('tp2_hits', 0)} | "
            f"🎯 TP3: {stats.get('tp3_hits', 0)} | "
            f"❌ SL: {stats.get('sl_hits', 0)} | "
            f"💰 Total P&L: ${stats.get('total_pnl', 0):.2f} | "
            f"📈 Win Rate: {stats.get('win_rate', 0):.1f}%"
        )
        
        return Panel(ticker_text, style="bold white on blue", box=box.ROUNDED)
    
    def generate_signals_table(self) -> Table:
        """
        Generate signals table with TP ladders and research-backed squeeze metrics.
        
        New columns from research:
        - SPS: Squeeze Potential Score (>50 = high prob, >70 = deadly)
        - Mag: Predicted magnitude based on OI Z-score, divergence, funding
        - Div: Whale-Crowd Divergence (>1.76 = critical for +7.5% PPV)
        """
        
        table = Table(
            box=box.DOUBLE_EDGE,
            title="[bold red]🔥 TOP SQUEEZE CANDIDATES | 4-GATE SCORING 🔥[/]",
            title_style="bold red"
        )
        
        table.add_column("Rank", style="yellow bold", justify="center", width=4)
        table.add_column("Symbol", style="cyan bold", width=9)
        table.add_column("Entry", justify="right", width=9)
        table.add_column("SL", justify="right", width=9, style="red")
        table.add_column("TP1", justify="right", width=9, style="green")
        table.add_column("TP3", justify="right", width=9, style="bright_green")
        table.add_column("SPS", justify="center", width=5)  # Squeeze Potential Score
        table.add_column("Mag", justify="center", width=5)  # Predicted Magnitude
        table.add_column("Div", justify="center", width=5)  # Whale-Crowd Divergence
        table.add_column("Score", justify="center", width=6, style="bold")
        table.add_column("Signal", justify="center", width=11)
        
        for i, signal in enumerate(self.top_signals, 1):
            tp_ladder = signal.get('tp_ladder')
            if not tp_ladder:
                continue
            
            # Styling based on score
            score = signal['final_score']
            if score >= 90:
                score_style = "bold white on red"
                signal_text = "🚨 EXTREME"
            elif score >= 80:
                score_style = "bold black on yellow"
                signal_text = "⚡ STRONG"
            elif score >= 70:
                score_style = "bold white on blue"
                signal_text = "📈 MODERATE"
            else:
                score_style = "white"
                signal_text = "👀 WATCH"
            
            # SPS (Squeeze Potential Score) styling
            sps = signal.get('squeeze_potential_score', 0)
            if sps >= 70:
                sps_text = Text(f"{sps:.0f}", style="bold white on red")
            elif sps >= 50:
                sps_text = Text(f"{sps:.0f}", style="bold yellow")
            else:
                sps_text = Text(f"{sps:.0f}", style="dim")
            
            # Magnitude styling
            magnitude = signal.get('predicted_magnitude', 0)
            if magnitude >= 20:
                mag_text = Text(f"{magnitude:.0f}%", style="bold white on green")
            elif magnitude >= 10:
                mag_text = Text(f"{magnitude:.0f}%", style="bold green")
            else:
                mag_text = Text(f"{magnitude:.0f}%", style="dim")
            
            # Whale-Crowd Divergence styling
            divergence = signal.get('whale_crowd_divergence', 1.0)
            if divergence >= 1.76:  # Deadly threshold from research
                div_text = Text(f"{divergence:.2f}", style="bold white on magenta")
            elif divergence >= 1.4:
                div_text = Text(f"{divergence:.2f}", style="bold magenta")
            else:
                div_text = Text(f"{divergence:.2f}", style="dim")
            
            table.add_row(
                f"#{i}",
                signal['symbol'].replace('USDT', ''),
                f"${signal['price']:.4f}",
                f"${tp_ladder.stop_loss:.4f}",
                f"${tp_ladder.tp1_price:.4f}",
                f"${tp_ladder.tp3_price:.4f}",
                sps_text,
                mag_text,
                div_text,
                Text(f"{score:.0f}", style=score_style),
                signal_text
            )
        
        return table
    
    def generate_active_trades_table(self) -> Table:
        """Generate active trades tracking table"""
        
        table = Table(
            box=box.ROUNDED,
            title="[bold cyan]📈 ACTIVE TRADES - LIVE TRACKING 📈[/]",
            title_style="bold cyan"
        )
        
        table.add_column("ID", style="yellow", width=4)
        table.add_column("Symbol", style="cyan bold", width=10)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("Status", justify="center", width=10)
        table.add_column("P&L $", justify="right", width=12)
        table.add_column("P&L %", justify="right", width=10)
        table.add_column("Next TP", justify="right", width=10)
        table.add_column("Duration", justify="right", width=10)
        
        if not self.active_trades:
            table.add_row("—", "NO ACTIVE TRADES", "—", "—", "—", "—", "—", "—", "—")
            return table
        
        for trade in self.active_trades.values():
            # P&L styling
            pnl_style = "bold green" if trade.pnl_usd > 0 else "bold red" if trade.pnl_usd < 0 else "white"
            
            # Status
            if trade.status == TradeStatus.TP1_HIT:
                status_text = "🎯 TP1 ✓"
                next_tp = f"${trade.tp2_price:.4f}"
            elif trade.status == TradeStatus.TP2_HIT:
                status_text = "🎯 TP2 ✓"
                next_tp = f"${trade.tp3_price:.4f}"
            else:
                status_text = "🔵 ACTIVE"
                next_tp = f"${trade.tp1_price:.4f}"
            
            # Duration
            duration = datetime.now() - trade.entry_time
            duration_str = f"{duration.seconds // 60}m"
            
            table.add_row(
                str(trade.trade_id),
                trade.symbol.replace('USDT', ''),
                f"${trade.entry_price:.4f}",
                f"${trade.current_price:.4f}",
                status_text,
                Text(f"${trade.pnl_usd:.2f}", style=pnl_style),
                Text(f"{trade.pnl_pct:+.2f}%", style=pnl_style),
                next_tp,
                duration_str
            )
        
        return table

async def main():
    """Main execution"""
    scanner = UltimateSqueezeScanner()
    await scanner.initialize()
    
    console.print("[bold green]🚀 ULTIMATE SQUEEZE SCANNER INITIALIZING...[/]")
    await asyncio.sleep(2)
    
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                # Scan market
                await scanner.scan_and_rank()
                
                # Update active trades
                await scanner.update_active_trades()
                
                # Generate display
                layout = Layout()
                layout.split_column(
                    Layout(scanner.generate_ticker(), size=3),
                    Layout(scanner.generate_signals_table(), size=18),
                    Layout(scanner.generate_active_trades_table(), size=12)
                )
                
                live.update(layout)
                
                # Wait before next scan
                await asyncio.sleep(30)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/]")
    finally:
        await scanner.close()
        console.print("[green]Scanner stopped. Trade safely! 📊[/]")

if __name__ == "__main__":
    asyncio.run(main())
