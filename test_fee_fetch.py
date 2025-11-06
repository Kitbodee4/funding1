#!/usr/bin/env python3
"""Test if fee_fetcher is actually fetching from API"""

import sys
import logging
from pathlib import Path

# Setup logging to see DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s | %(name)s:%(funcName)s - %(message)s'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from capital_transfer.fee_fetcher import FeeFetcher
from capital_transfer.chain_selector import ChainSelector

print("="*80)
print("🧪 TESTING FEE FETCHER")
print("="*80)

# Test 1: Direct FeeFetcher test (without API keys)
print("\n1️⃣  Test FeeFetcher directly (no API keys):")
print("-"*60)
fetcher = FeeFetcher(cache_ttl_hours=0)  # Disable cache to force fresh fetch
fee, from_cache = fetcher.fetch_fee_from_exchange('binance', 'bnb', 'USDT')
print(f"\n✅ Result: Fee = ${fee}, From Cache = {from_cache}")
print(f"   Expected: Should see DEBUG logs above showing 'API', 'cache', or 'fallback'")

# Test 2: Direct FeeFetcher test for APTOS
print("\n2️⃣  Test FeeFetcher for APTOS:")
print("-"*60)
fee2, from_cache2 = fetcher.fetch_fee_from_exchange('binance', 'aptos', 'USDT')
print(f"\n✅ Result: Fee = ${fee2}, From Cache = {from_cache2}")

# Test 3: ChainSelector test
print("\n3️⃣  Test ChainSelector (uses FeeFetcher internally):")
print("-"*60)
selector = ChainSelector('capital_config.yaml')
chains = selector.get_all_chains('binance', 'gateio')
print(f"\nFound {len(chains)} chains:")
for chain in chains[:5]:
    print(f"   • {chain.chain_code:10s}: ${chain.fee_usdt:.4f}")

print("\n" + "="*80)
print("📋 ANALYSIS:")
print("="*80)
print("If you see logs with:")
print("  - '💰 ... (API)' → Successfully fetched from exchange API")
print("  - '💰 ... (cache)' → Using cached value from previous API call")
print("  - '💾 ... (historical fallback)' → Using hardcoded fallback (API failed)")
print("  - '⚠️  ... (config fallback)' → Using capital_config.yaml (last resort)")
print("\nIf NO logs appear, the fee_fetcher may not be called at all!")
print("="*80)
