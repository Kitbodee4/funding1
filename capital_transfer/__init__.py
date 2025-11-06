"""
Capital Transfer System (Part 2)

Automatic capital transfer system with 7-layer safety mechanism.

Components:
- transfer_engine: Core transfer engine with safety checks
- chain_selector: 3-tier chain selection strategy
- chain_health_checker: Blockchain network health monitoring
- capital_config_parser: Excel to YAML configuration parser
- transaction_history: SQLite-based transaction logging
- transfer_integration: Integration with funding scanner

Safety Layers:
1. Prerequisites Check - API credentials, balance, network status
2. Test Transfer - Small amount test (10.1 USDT)
3. Test Confirmation - Verify test transfer arrival
4. Actual Transfer - Transfer remaining amount
5. Actual Confirmation - Verify actual transfer
6. History Recording - Log all transactions
7. Automatic Fallback - Switch to backup chains on failure

3-Tier Chain Strategy:
- Tier 1 (Primary): Cheapest chain (70% fee weight)
- Tier 2 (Secondary): Balanced speed/cost (40% each)
- Tier 3 (Tertiary): Most reliable (60% reliability weight)
"""

from .transfer_engine import (
    CapitalTransferEngine,
    TransferRequest,
    TransferResult,
    graceful_shutdown,
)
from .chain_selector import (
    ChainSelector,
    ChainOption,
)
from .chain_health_checker import (
    ChainHealthChecker,
    ChainHealth,
)
from .capital_config_parser import (
    CapitalConfigParser,
)
from .transaction_history import (
    TransactionHistoryDB,
    TransferRecord,
)
from .transfer_integration import (
    CapitalTransferIntegration,
)

__all__ = [
    # Main engine
    'CapitalTransferEngine',
    'TransferRequest',
    'TransferResult',
    'graceful_shutdown',

    # Chain selection
    'ChainSelector',
    'ChainOption',

    # Health checking
    'ChainHealthChecker',
    'ChainHealth',

    # Configuration
    'CapitalConfigParser',

    # History tracking
    'TransactionHistoryDB',
    'TransferRecord',

    # Integration
    'CapitalTransferIntegration',
]

__version__ = '2.0.0'
__author__ = 'Capital Transfer System'
__description__ = 'Automated capital transfer with 7-layer safety'
