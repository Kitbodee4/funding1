#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Fee Fetcher with Network Verification

Combines the existing FeeFetcher with NetworkVerifier to provide:
- Real-time withdrawal fee fetching with authentication
- Network availability verification
- Cross-exchange compatibility checking

This module enhances the existing fee_fetcher.py with network verification capabilities.

Author: Capital Transfer System
Version: 2.0.0
"""

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, Optional, Tuple
from loguru import logger
from pathlib import Path
import json
from datetime import datetime, timedelta

from .network_verifier import NetworkVerifier, NetworkInfo


class EnhancedFeeFetcher:
    """
    Enhanced fee fetcher with network verification

    Combines fee fetching with network availability checking to ensure
    accurate and real-time transfer cost calculation.
    """

    def __init__(
        self,
        api_credentials: Dict[str, Dict[str, str]],
        cache_file: str = "enhanced_fee_cache.json",
        cache_ttl_hours: int = 24
    ):
        """
        Initialize enhanced fee fetcher

        Args:
            api_credentials: Dict of {exchange_id: {api_key, secret, password}}
            cache_file: Path to cache file
            cache_ttl_hours: Cache TTL in hours
        """
        self.api_credentials = api_credentials
        self.cache_file = Path(cache_file)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache = self._load_cache()

        # Initialize network verifier
        self.verifier = NetworkVerifier(api_credentials)

        logger.info(f"💰 Enhanced Fee Fetcher initialized (cache TTL: {cache_ttl_hours}h)")

    def _load_cache(self) -> Dict:
        """Load fee cache from file"""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)

            # Clean expired entries
            now = datetime.now()
            cleaned_cache = {}
            for key, entry in cache.items():
                expire_time = datetime.fromisoformat(entry['expires_at'])
                if expire_time > now:
                    cleaned_cache[key] = entry

            if len(cleaned_cache) < len(cache):
                logger.info(f"🧹 Cleaned {len(cache) - len(cleaned_cache)} expired cache entries")
                self._save_cache(cleaned_cache)

            return cleaned_cache

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}

    def _save_cache(self, cache: Optional[Dict] = None):
        """Save fee cache to file"""
        try:
            cache_to_save = cache if cache is not None else self.cache
            with open(self.cache_file, 'w') as f:
                json.dump(cache_to_save, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_cache_key(
        self,
        source_exchange: str,
        target_exchange: str,
        chain_code: str,
        currency: str = 'USDT'
    ) -> str:
        """Generate cache key for transfer route"""
        return f"{source_exchange}:{target_exchange}:{currency}:{chain_code}"

    def _get_cached_fee(
        self,
        source_exchange: str,
        target_exchange: str,
        chain_code: str,
        currency: str = 'USDT'
    ) -> Optional[Dict]:
        """Get fee from cache if valid"""
        key = self._get_cache_key(source_exchange, target_exchange, chain_code, currency)

        if key in self.cache:
            entry = self.cache[key]
            expire_time = datetime.fromisoformat(entry['expires_at'])

            if datetime.now() < expire_time:
                logger.debug(f"💾 Cache HIT: {key}")
                return entry

        return None

    def _set_cached_fee(
        self,
        source_exchange: str,
        target_exchange: str,
        chain_code: str,
        fee_data: Dict,
        currency: str = 'USDT'
    ):
        """Store fee in cache"""
        key = self._get_cache_key(source_exchange, target_exchange, chain_code, currency)

        self.cache[key] = {
            **fee_data,
            'cached_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + self.cache_ttl).isoformat(),
        }

        self._save_cache()
        logger.debug(f"💾 Cache SET: {key}")

    async def fetch_transfer_fee_with_verification(
        self,
        source_exchange: str,
        target_exchange: str,
        chain_code: str,
        currency: str = 'USDT'
    ) -> Dict:
        """
        Fetch transfer fee and verify network availability

        Args:
            source_exchange: Source exchange ID
            target_exchange: Target exchange ID
            chain_code: Standard chain code (arb, trc, erc, etc.)
            currency: Currency code (default: USDT)

        Returns:
            Dict with:
            - available: bool
            - fee: float or None
            - source_network_id: str
            - target_network_id: str
            - verified: bool (True if checked with live API)
            - from_cache: bool
        """
        # Check cache first
        cached = self._get_cached_fee(source_exchange, target_exchange, chain_code, currency)
        if cached:
            return {
                **cached,
                'from_cache': True
            }

        # Verify chain availability and get fees
        try:
            available, src_net, tgt_net, src_fee, tgt_fee = await self.verifier.verify_chain_availability(
                source_exchange, target_exchange, chain_code, currency
            )

            result = {
                'available': available,
                'fee': src_fee,
                'source_network_id': src_net,
                'target_network_id': tgt_net,
                'source_exchange': source_exchange,
                'target_exchange': target_exchange,
                'chain_code': chain_code,
                'currency': currency,
                'verified': True,
                'from_cache': False,
            }

            # Cache the result
            if available and src_fee is not None:
                self._set_cached_fee(source_exchange, target_exchange, chain_code, result, currency)

            return result

        except Exception as e:
            logger.error(f"Error fetching transfer fee: {e}")
            return {
                'available': False,
                'fee': None,
                'source_network_id': None,
                'target_network_id': None,
                'source_exchange': source_exchange,
                'target_exchange': target_exchange,
                'chain_code': chain_code,
                'currency': currency,
                'verified': False,
                'from_cache': False,
                'error': str(e)
            }

    async def get_all_transfer_options(
        self,
        source_exchange: str,
        target_exchange: str,
        currency: str = 'USDT'
    ) -> list[Dict]:
        """
        Get all available transfer options between two exchanges

        Args:
            source_exchange: Source exchange ID
            target_exchange: Target exchange ID
            currency: Currency code

        Returns:
            List of transfer option dicts, sorted by fee (lowest first)
        """
        logger.info(f"🔍 Finding transfer options: {source_exchange} -> {target_exchange}")

        # Get common chains
        common_chains = await self.verifier.get_common_chains(
            source_exchange, target_exchange, currency
        )

        # Fetch detailed info for each chain
        options = []
        for chain_info in common_chains:
            chain_code = chain_info['chain_code']

            option = {
                'chain_code': chain_code,
                'source_network_id': chain_info['source_network_id'],
                'target_network_id': chain_info['target_network_id'],
                'fee': chain_info['fee'],
                'available': True,
            }

            options.append(option)

        # Sort by fee (lowest first)
        options.sort(key=lambda x: x['fee'] if x['fee'] is not None else float('inf'))

        logger.info(f"✓ Found {len(options)} transfer options")

        return options

    async def get_best_transfer_route(
        self,
        source_exchange: str,
        target_exchange: str,
        currency: str = 'USDT',
        preferred_chains: Optional[list[str]] = None
    ) -> Optional[Dict]:
        """
        Get the best transfer route based on fees and preferences

        Args:
            source_exchange: Source exchange ID
            target_exchange: Target exchange ID
            currency: Currency code
            preferred_chains: Optional list of preferred chain codes (in priority order)

        Returns:
            Best transfer option dict or None
        """
        options = await self.get_all_transfer_options(source_exchange, target_exchange, currency)

        if not options:
            logger.warning(f"No transfer options available between {source_exchange} and {target_exchange}")
            return None

        # If preferred chains specified, try them first
        if preferred_chains:
            for chain_code in preferred_chains:
                for option in options:
                    if option['chain_code'] == chain_code and option['available']:
                        logger.info(f"✓ Selected preferred chain: {chain_code} (fee: ${option['fee']})")
                        return option

        # Otherwise, return the cheapest option
        best = options[0]
        logger.info(f"✓ Selected cheapest chain: {best['chain_code']} (fee: ${best['fee']})")
        return best

    def clear_cache(self):
        """Clear all cached fees"""
        self.cache = {}
        self._save_cache()
        logger.info("🧹 Cache cleared")


# Standalone test
async def main():
    """Test enhanced fee fetcher"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("="*100)
    print("ENHANCED FEE FETCHER TEST")
    print("="*100)
    print()

    # Setup credentials
    credentials = {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
        },
        'gateio': {
            'api_key': os.getenv('GATEIO_API_KEY'),
            'secret': os.getenv('GATEIO_API_SECRET'),
        },
        'kucoin': {
            'api_key': os.getenv('KUCOINFUTURES_API_KEY'),
            'secret': os.getenv('KUCOINFUTURES_API_SECRET'),
            'password': os.getenv('KUCOINFUTURES_PASSWORD'),
        },
        'bitget': {
            'api_key': os.getenv('BITGET_API_KEY'),
            'secret': os.getenv('BITGET_API_SECRET'),
            'password': os.getenv('BITGET_PASSWORD'),
        },
    }

    fetcher = EnhancedFeeFetcher(credentials, cache_ttl_hours=1)

    # Test 1: Fetch single transfer fee
    print("TEST 1: Fetch transfer fee with verification")
    print("-"*100)

    result = await fetcher.fetch_transfer_fee_with_verification(
        'bitget', 'kucoin', 'arb', 'USDT'
    )

    print(f"Available: {result['available']}")
    print(f"Fee: ${result['fee']}")
    print(f"Source network: {result['source_network_id']}")
    print(f"Target network: {result['target_network_id']}")
    print(f"From cache: {result['from_cache']}")
    print()

    # Test 2: Get all transfer options
    print("TEST 2: Get all transfer options")
    print("-"*100)

    options = await fetcher.get_all_transfer_options('bitget', 'kucoin', 'USDT')

    print(f"{'Chain':<15} {'Source Network':<20} {'Target Network':<20} {'Fee':<10}")
    print("-"*100)
    for opt in options:
        fee_str = f"${opt['fee']:.4f}" if opt['fee'] is not None else "N/A"
        print(f"{opt['chain_code']:<15} {opt['source_network_id']:<20} {opt['target_network_id']:<20} {fee_str:<10}")
    print()

    # Test 3: Get best transfer route
    print("TEST 3: Get best transfer route")
    print("-"*100)

    best = await fetcher.get_best_transfer_route(
        'bitget', 'kucoin', 'USDT',
        preferred_chains=['trc', 'arb']  # Prefer TRC20, then Arbitrum
    )

    if best:
        print(f"Best route: {best['chain_code']}")
        print(f"Fee: ${best['fee']}")
        print(f"Networks: {best['source_network_id']} -> {best['target_network_id']}")

    print()
    print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
