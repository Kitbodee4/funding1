# Capital Transfer System (Part 2)

**Automated capital transfer system with 7-layer safety mechanism**

Based on: หมวดที่ 2: ระบบโอนย้ายเงินทุนอัตโนมัติ

## 📋 Overview

The Capital Transfer System provides secure, automated transfer of funds between cryptocurrency exchanges with comprehensive safety checks and fallback mechanisms. It's designed to support funding rate arbitrage strategies by safely moving capital to optimal locations.

## ✨ Key Features

### 7-Layer Safety System

1. **Prerequisites Check** - Verify API credentials, balance, network status
2. **Test Transfer** - Execute small test transfer (10.1 USDT)
3. **Test Confirmation** - Verify test transfer arrival
4. **Actual Transfer** - Transfer remaining amount
5. **Actual Confirmation** - Verify actual transfer arrival
6. **History Recording** - Log all transactions to SQLite
7. **Automatic Fallback** - Switch to backup chains on failure

### 3-Tier Chain Selection Strategy

- **Tier 1 (Primary)**: Cheapest chain (70% fee weight, 20% speed, 10% reliability)
- **Tier 2 (Secondary)**: Balanced chain (40% fee, 40% speed, 20% reliability)
- **Tier 3 (Tertiary)**: Most reliable chain (20% fee, 20% speed, 60% reliability)

### Additional Features

- ✅ Multi-exchange support (Binance, Bybit, OKX, Bitget, KuCoin, Gate.io)
- ✅ Multi-chain support (Arbitrum, Optimism, Polygon, BNB Chain, Tron, etc.)
- ✅ Address validation (checksums, format verification)
- ✅ Chain health monitoring
- ✅ Transaction history with SQLite
- ✅ Excel configuration support
- ✅ Integration with funding scanner
- ✅ Reconciliation and reporting

## 📦 Module Structure

```
capital_transfer/
├── __init__.py                    # Module exports
├── README.md                      # This file
├── transfer_engine.py             # Core transfer engine (648 lines)
├── chain_selector.py              # 3-tier chain selection (368 lines)
├── chain_health_checker.py        # Network health monitoring (348 lines)
├── capital_config_parser.py       # Excel to YAML parser (316 lines)
├── transaction_history.py         # SQLite transaction logging (650 lines)
├── transfer_integration.py        # Scanner integration (408 lines)
└── complete_examples.py           # 7 complete examples (500+ lines)
```

## 🚀 Quick Start

### Basic Transfer

```python
import asyncio
from cross_ex_fund.capital_transfer import (
    CapitalTransferEngine,
    TransferRequest
)

async def main():
    # Initialize engine
    engine = CapitalTransferEngine(
        config_path="capital_config.yaml",
        enable_test_transfer=True,
        enable_health_check=True
    )

    # Initialize exchanges
    await engine.initialize_exchange("binance", {
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret'
    })

    await engine.initialize_exchange("gateio", {
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret'
    })

    # Create transfer request
    request = TransferRequest(
        request_id="TRANSFER001",
        source_exchange="binance",
        target_exchange="gateio",
        amount_usdt=100.0,
        purpose="funding_arbitrage"
    )

    # Execute transfer with automatic fallback
    result = await engine.transfer_with_fallback(request)

    if result.success:
        print(f"✅ Transfer completed!")
        print(f"  Chain: {result.chain_used}")
        print(f"  Fee: ${result.fee_paid}")
        print(f"  Time: {result.total_time_seconds}s")
    else:
        print(f"❌ Transfer failed: {result.error_message}")

    await engine.close()

asyncio.run(main())
```

### Chain Selection

```python
from cross_ex_fund.capital_transfer import ChainSelector

# Initialize selector
selector = ChainSelector("capital_config.yaml")

# Get 3-tier chain selection
chains = selector.select_chains_3tier("binance", "gateio")

# Access chains
primary = chains['primary']      # Cheapest
secondary = chains['secondary']  # Balanced
tertiary = chains['tertiary']    # Most reliable

# Print comparison
selector.print_chain_comparison("binance", "gateio")
```

### Transaction History

```python
from cross_ex_fund.capital_transfer import TransactionHistoryDB

# Initialize database
db = TransactionHistoryDB("transaction_history.db")

# Get statistics
stats = db.get_statistics()
print(f"Total Transfers: {stats['total_transfers']}")
print(f"Success Rate: {stats['success_rate']:.1f}%")
print(f"Total Volume: ${stats['total_volume_usdt']:,.2f}")

# Get transfers by exchange
transfers = db.get_transfers_by_exchange("binance", direction="source")

# Reconcile
reconcile = db.reconcile("binance")
print(f"Sent: ${reconcile['total_sent']}")
print(f"Received: ${reconcile['total_received']}")
print(f"Net Flow: ${reconcile['net_flow']}")

db.close()
```

## 📖 Complete Examples

The module includes 7 comprehensive examples demonstrating all features:

```bash
# Run all examples
python -m cross_ex_fund.capital_transfer.complete_examples

# Run specific example
python -m cross_ex_fund.capital_transfer.complete_examples 1
```

### Example 1: Basic Transfer
Simple transfer workflow demonstration

### Example 2: Multi-Chain Transfer
Compare costs across multiple blockchain networks

### Example 3: Automatic Fallback
Demonstrate 3-tier fallback on failure

### Example 4: Batch Transfers
Execute multiple concurrent transfers

### Example 5: Excel Config Integration
Load configuration from Excel spreadsheet

### Example 6: Transaction History Query
Query and analyze transaction history

### Example 7: Full Integration
Complete workflow: Scanner → Transfer → Accounting

## ⚙️ Configuration

### YAML Configuration Format

```yaml
exchanges:
  binance:
    chains:
      arbitrum:
        code: arb
        enabled: true
        whitelisted: true
        fee_usdt: 0.8
        withdraw_time_minutes: 1
        deposit_time_minutes: 1
      optimism:
        code: op
        enabled: true
        whitelisted: true
        fee_usdt: 0.1
        withdraw_time_minutes: 1
        deposit_time_minutes: 1

  gateio:
    chains:
      arbitrum:
        code: arb
        enabled: true
        whitelisted: true
        fee_usdt: 0.0
        deposit_time_minutes: 1
        deposit_address: "0x..."
      optimism:
        code: op
        enabled: true
        whitelisted: false
        fee_usdt: 0.0
        deposit_time_minutes: 1
        deposit_address: "0x..."
```

### Excel Configuration

Alternatively, use Excel format and convert to YAML:

```python
from cross_ex_fund.capital_transfer import CapitalConfigParser

parser = CapitalConfigParser("capital_config.xlsx")
config = parser.parse()
parser.save_yaml("capital_config.yaml")
```

## 🔗 Integration with Funding Scanner

```python
from cross_ex_fund.capital_transfer import (
    CapitalTransferEngine,
    CapitalTransferIntegration,
    TransactionHistoryDB
)
from cross_ex_fund.funding_scanner import EnhancedFundingScanner

# Initialize components
scanner = EnhancedFundingScanner(exchanges=['binance', 'gateio', 'bybit'])
engine = CapitalTransferEngine()
history = TransactionHistoryDB()

# Create integration
integration = CapitalTransferIntegration(
    transfer_engine=engine,
    history_db=history,
    main_hub_exchange="binance"
)

# Scan for opportunities
opportunities = await scanner.scan()

# Transfer capital for best opportunity
if opportunities:
    best = opportunities[0]
    results = await integration.transfer_capital_for_opportunity(
        opportunity=best,
        capital=10000.0
    )
```

## 📊 Transaction History Database Schema

### Tables

1. **transfers** - Main transfer records
   - request_id, source_exchange, target_exchange
   - amount_usdt, chain_code, success
   - test_txid, actual_txid
   - fee_paid, total_time_seconds

2. **transactions** - Individual transaction details
   - txid, transfer_request_id
   - transaction_type (test/actual)
   - amount, status

3. **fees** - Detailed fee tracking
   - transfer_request_id, fee_type
   - amount_usdt, chain_code

4. **errors** - Error logging
   - transfer_request_id, error_type
   - error_message, occurred_at

## 🛡️ Safety Features

### Address Validation

- EVM chains (Ethereum, BSC, Arbitrum, Optimism, Polygon, Avalanche)
  - Must start with '0x'
  - Must be 42 characters
  - Checksum validation (TODO)

- Tron (TRC20)
  - Must start with 'T'
  - Must be 34 characters

- Solana
  - Length: 32-44 characters

- Aptos
  - Must start with '0x'

### Health Checks

- Exchange status (maintenance mode)
- Wallet status (deposit/withdrawal enabled)
- Network congestion
- Gas fees monitoring
- Block height verification

### Retry Logic

- Maximum retry attempts: 3
- Exponential backoff: 5s base delay
- Automatic chain fallback
- Transaction status tracking

## 📈 Performance

- **Startup Time**: < 1 second
- **Transfer Time**: 2-10 minutes (depends on chain)
- **Test Transfer**: 10.1 USDT
- **Min Transfer**: 10.0 USDT
- **Max Transfer**: 100,000.0 USDT
- **Confirmation Wait**: Up to 30 minutes

## 🔧 API Reference

### CapitalTransferEngine

```python
engine = CapitalTransferEngine(
    config_path: str = "capital_config.yaml",
    enable_test_transfer: bool = True,
    enable_health_check: bool = True
)

# Initialize exchange
await engine.initialize_exchange(exchange_name, api_credentials)

# Execute transfer
result = await engine.transfer(request, chain=None)

# Transfer with fallback
result = await engine.transfer_with_fallback(request, max_fallback_attempts=2)

# Close
await engine.close()
```

### ChainSelector

```python
selector = ChainSelector(config_path="capital_config.yaml")

# Get all chains
chains = selector.get_all_chains(source_exchange, target_exchange)

# Get 3-tier selection
chains_dict = selector.select_chains_3tier(source_exchange, target_exchange)

# Get best chain
chain = selector.get_best_chain(source_exchange, target_exchange, tier='primary')

# Get fallback
chain = selector.get_fallback_chain(source_exchange, target_exchange, failed_chains=[])
```

### TransactionHistoryDB

```python
db = TransactionHistoryDB(db_path="transaction_history.db")

# Record transfer
success = db.record_transfer(record)

# Query
transfer = db.get_transfer(request_id)
transfers = db.get_successful_transfers()
transfers = db.get_failed_transfers()
transfers = db.get_transfers_by_exchange(exchange, direction='both')

# Statistics
stats = db.get_statistics()
reconcile = db.reconcile(exchange)

# Print
db.print_statistics()

# Close
db.close()
```

## 📝 Requirements

```
ccxt>=4.0.0
sqlalchemy>=2.0.0
loguru>=0.7.0
aiohttp>=3.8.0
pyyaml>=6.0
pandas>=2.0.0
python-dateutil
openpyxl>=3.0.0
```

## 🧪 Testing

```bash
# Test transfer engine
python -m cross_ex_fund.capital_transfer.transfer_engine

# Test chain selector
python -m cross_ex_fund.capital_transfer.chain_selector

# Test transaction history
python -m cross_ex_fund.capital_transfer.transaction_history

# Run all examples
python -m cross_ex_fund.capital_transfer.complete_examples
```

## 🔄 Integration with Strategy

```python
from cross_ex_fund.core.strategy import Strategy
from cross_ex_fund.capital_transfer import CapitalTransferEngine

class MyStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize transfer engine
        self.transfer_engine = CapitalTransferEngine()

        # Initialize exchanges
        for exchange_name in self.exchanges:
            await self.transfer_engine.initialize_exchange(
                exchange_name,
                self.api_credentials[exchange_name]
            )

    async def rebalance_capital(self):
        """Rebalance capital across exchanges"""
        # Create transfer requests based on positions
        # Execute transfers
        # Update accounting
        pass
```

## 📚 Documentation References

- **Main Documentation**: `/upgrade/DOCUMENTATION_ORGANIZED.md` (หมวดที่ 2)
- **System Overview**: Lines 329-722
- **7-Layer Safety**: Lines 354-418
- **3-Tier Strategy**: Lines 442-487
- **Complete Examples**: Lines 651-721

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| Total Files | 8 files |
| Total Lines | ~3,500 lines |
| Coverage | 7 components |
| Examples | 7 comprehensive |
| Safety Layers | 7 layers |
| Chain Tiers | 3 tiers |
| Supported Exchanges | 6+ exchanges |
| Supported Chains | 10+ chains |

## 🚨 Important Notes

1. **Test Mode**: Always test with `enable_test_transfer=False` in development
2. **API Keys**: Store securely, never commit to version control
3. **Whitelist**: Always whitelist withdrawal addresses
4. **Monitoring**: Monitor all transfers in real-time
5. **Reconciliation**: Daily reconciliation recommended
6. **Backup**: Always have fallback chains configured
7. **Limits**: Respect exchange withdrawal limits

## 🤝 Contributing

This module is part of the larger funding arbitrage system. See main project documentation for contribution guidelines.

## 📄 License

Part of the Cross-Exchange Funding Arbitrage System

## 🔮 Future Enhancements

- [ ] Checksum validation for EVM addresses
- [ ] Multi-asset support (beyond USDT)
- [ ] Gas price optimization
- [ ] Cross-chain bridges support
- [ ] Advanced reconciliation
- [ ] Performance analytics dashboard
- [ ] Alert system integration
- [ ] Multi-signature support

---

**Version**: 2.0.0
**Last Updated**: 2025-10-27
**Documentation Language**: English (Based on Thai documentation)
**Author**: Capital Transfer System
