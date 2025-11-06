#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Network Verifier for Capital Transfer System

Verifies that networks are available and active on both source and target exchanges.
Fetches real-time withdrawal fees using authenticated CCXT API calls.

Features:
- Network availability checking across exchanges
- Real-time withdrawal fee fetching with authentication
- Network name normalization and mapping
- Integration with existing FeeFetcher for caching

Author: Capital Transfer System
Version: 1.0.0
"""

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
from datetime import datetime


@dataclass
class NetworkInfo:
    """Information about a network on an exchange"""
    network_id: str
    network_name: str
    deposit_enabled: bool
    withdraw_enabled: bool
    fee: Optional[float]
    active: bool

    def __repr__(self):
        status = "OK" if self.active else "NO"
        fee_str = f"${self.fee:.4f}" if self.fee is not None else "N/A"
        return f"[{status}] {self.network_id}: {fee_str}"


class NetworkVerifier:
    """
    Verify network availability and fetch fees across exchanges

    Integrates with capital transfer system to ensure selected networks
    are actually available on both source and target exchanges.
    """

    # Standard chain code to network name mapping
    # This maps the internal chain codes to possible network names across exchanges
    CHAIN_TO_NETWORK_MAPPING = {
        'arb': ['ARBITRUM', 'ARB', 'ARBONE', 'Arbitrum One'],
        'op': ['OPTIMISM', 'OP', 'OPETH', 'Optimism'],
        'polygon': ['POLYGON', 'MATIC', 'PLASMA', 'plasma'],
        'trc': ['TRX', 'TRC20', 'TRON'],
        'bnb': ['BSC', 'BNB', 'BEP20'],
        'erc': ['ETH', 'ERC20', 'ETHEREUM', 'Ethereum'],
        'sol': ['SOL', 'SOLANA'],
        'avax': ['AVAX', 'AVAXC', 'AVAXC-CHAIN', 'Avalanche C'],
        'apt': ['APT', 'APTOS'],
        'near': ['NEAR'],
        'ton': ['TON'],
    }

    # Reverse mapping: network name -> standard chain code
    NETWORK_TO_CHAIN_MAPPING = {}

    def __init__(self, api_credentials: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize network verifier

        Args:
            api_credentials: Dict of {exchange_id: {api_key, secret, password}}
                           Required for authenticated fee fetching
        """
        self.api_credentials = api_credentials or {}

        # Build reverse mapping
        if not self.NETWORK_TO_CHAIN_MAPPING:
            for chain_code, network_names in self.CHAIN_TO_NETWORK_MAPPING.items():
                for network_name in network_names:
                    self.NETWORK_TO_CHAIN_MAPPING[network_name.upper()] = chain_code

        logger.info(f"🔍 Network Verifier initialized with credentials for {len(self.api_credentials)} exchanges")

    async def fetch_networks_for_currency(
        self,
        exchange_id: str,
        currency: str = 'USDT'
    ) -> Dict[str, NetworkInfo]:
        """
        Fetch available networks for a currency on an exchange

        Args:
            exchange_id: Exchange identifier (binance, gateio, etc.)
            currency: Currency code (default: USDT)

        Returns:
            Dict mapping network_id -> NetworkInfo
        """
        networks = {}

        try:
            # Create exchange instance
            exchange = await self._create_exchange(exchange_id)

            try:
                # Load markets
                await exchange.load_markets()

                # Try to fetch withdrawal fees via authenticated endpoint
                fee_data = {}
                creds = self.api_credentials.get(exchange_id, {})

                if creds.get('api_key') and hasattr(exchange, 'fetch_deposit_withdraw_fee'):
                    try:
                        fee_info = await exchange.fetch_deposit_withdraw_fee(currency)
                        if 'networks' in fee_info:
                            fee_data = fee_info['networks']
                            logger.debug(f"✓ Fetched fee data for {exchange_id} via authenticated API")
                    except Exception as e:
                        logger.debug(f"Could not fetch fee data for {exchange_id}: {e}")

                # Get currency information
                if currency not in exchange.currencies:
                    logger.warning(f"{currency} not available on {exchange_id}")
                    return networks

                currency_info = exchange.currencies[currency]

                if 'networks' not in currency_info or not currency_info['networks']:
                    logger.warning(f"No network information for {currency} on {exchange_id}")
                    return networks

                # Parse network information
                for network_id, network_info in currency_info['networks'].items():
                    if not isinstance(network_info, dict):
                        continue

                    # Try to get fee from multiple sources
                    fee = network_info.get('fee')

                    # If fee not in network_info, check fee_data from authenticated API
                    if (fee is None or fee == 'N/A') and network_id in fee_data:
                        fee_network = fee_data[network_id]
                        if isinstance(fee_network, dict) and 'withdraw' in fee_network:
                            withdraw_info = fee_network['withdraw']
                            if isinstance(withdraw_info, dict):
                                fee = withdraw_info.get('fee')

                    # Create NetworkInfo
                    deposit_enabled = network_info.get('deposit', False)
                    withdraw_enabled = network_info.get('withdraw', False)
                    active = deposit_enabled and withdraw_enabled

                    networks[network_id] = NetworkInfo(
                        network_id=network_id,
                        network_name=network_info.get('name', network_id),
                        deposit_enabled=deposit_enabled,
                        withdraw_enabled=withdraw_enabled,
                        fee=fee,
                        active=active
                    )

                logger.info(f"✓ {exchange_id}: Found {len(networks)} networks for {currency}")

            finally:
                await exchange.close()

        except Exception as e:
            logger.error(f"Error fetching networks for {exchange_id}: {e}")

        return networks

    async def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Create CCXT exchange instance with credentials"""
        exchange_class = getattr(ccxt, exchange_id)

        config = {
            'enableRateLimit': True,
            'timeout': 30000,
        }

        # Add credentials if available
        creds = self.api_credentials.get(exchange_id, {})
        if creds.get('api_key'):
            config['apiKey'] = creds['api_key']
            config['secret'] = creds['secret']
            if creds.get('password'):
                config['password'] = creds['password']

        return exchange_class(config)

    def normalize_network_to_chain_code(self, network_id: str) -> Optional[str]:
        """
        Normalize exchange-specific network ID to standard chain code

        Args:
            network_id: Exchange-specific network ID (e.g., 'ERC20', 'TRC20', 'ARBONE')

        Returns:
            Standard chain code (e.g., 'erc', 'trc', 'arb') or None
        """
        network_upper = network_id.upper()

        # Direct lookup
        if network_upper in self.NETWORK_TO_CHAIN_MAPPING:
            return self.NETWORK_TO_CHAIN_MAPPING[network_upper]

        # Partial match
        for network_name, chain_code in self.NETWORK_TO_CHAIN_MAPPING.items():
            if network_name in network_upper or network_upper in network_name:
                return chain_code

        return None

    async def verify_chain_availability(
        self,
        source_exchange: str,
        target_exchange: str,
        chain_code: str,
        currency: str = 'USDT'
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[float], Optional[float]]:
        """
        Verify that a chain is available on both source and target exchanges

        Args:
            source_exchange: Source exchange ID
            target_exchange: Target exchange ID
            chain_code: Standard chain code (e.g., 'arb', 'trc', 'erc')
            currency: Currency code (default: USDT)

        Returns:
            Tuple of:
            - available: bool (True if chain is available on both exchanges)
            - source_network_id: Network ID on source exchange
            - target_network_id: Network ID on target exchange
            - source_fee: Withdrawal fee on source exchange
            - target_fee: Deposit fee on target exchange (usually 0)
        """
        # Fetch networks from both exchanges concurrently
        source_task = self.fetch_networks_for_currency(source_exchange, currency)
        target_task = self.fetch_networks_for_currency(target_exchange, currency)

        source_networks, target_networks = await asyncio.gather(source_task, target_task)

        # Find matching networks
        source_network_id = None
        target_network_id = None
        source_fee = None
        target_fee = None

        # Get possible network names for this chain code
        possible_names = self.CHAIN_TO_NETWORK_MAPPING.get(chain_code, [])

        # Find on source exchange
        for network_id, network_info in source_networks.items():
            if self.normalize_network_to_chain_code(network_id) == chain_code:
                if network_info.withdraw_enabled:
                    source_network_id = network_id
                    source_fee = network_info.fee
                    break

        # Find on target exchange
        for network_id, network_info in target_networks.items():
            if self.normalize_network_to_chain_code(network_id) == chain_code:
                if network_info.deposit_enabled:
                    target_network_id = network_id
                    target_fee = 0.0  # Deposits usually free
                    break

        available = (source_network_id is not None and target_network_id is not None)

        if available:
            logger.info(
                f"[OK] Chain {chain_code} available: "
                f"{source_exchange}({source_network_id}, ${source_fee}) -> "
                f"{target_exchange}({target_network_id})"
            )
        else:
            missing = []
            if source_network_id is None:
                missing.append(f"{source_exchange} (withdraw)")
            if target_network_id is None:
                missing.append(f"{target_exchange} (deposit)")
            logger.warning(f"[NO] Chain {chain_code} NOT available on: {', '.join(missing)}")

        return available, source_network_id, target_network_id, source_fee, target_fee

    async def get_common_chains(
        self,
        source_exchange: str,
        target_exchange: str,
        currency: str = 'USDT'
    ) -> List[Dict]:
        """
        Get all common chains available between two exchanges

        Args:
            source_exchange: Source exchange ID
            target_exchange: Target exchange ID
            currency: Currency code

        Returns:
            List of dicts with chain information
        """
        # Fetch networks
        source_networks, target_networks = await asyncio.gather(
            self.fetch_networks_for_currency(source_exchange, currency),
            self.fetch_networks_for_currency(target_exchange, currency)
        )

        # Find common chains
        common_chains = []

        for source_net_id, source_info in source_networks.items():
            if not source_info.withdraw_enabled:
                continue

            source_chain = self.normalize_network_to_chain_code(source_net_id)
            if not source_chain:
                continue

            # Look for matching chain on target
            for target_net_id, target_info in target_networks.items():
                if not target_info.deposit_enabled:
                    continue

                target_chain = self.normalize_network_to_chain_code(target_net_id)

                if source_chain == target_chain:
                    common_chains.append({
                        'chain_code': source_chain,
                        'source_network_id': source_net_id,
                        'target_network_id': target_net_id,
                        'fee': source_info.fee,
                        'source_active': source_info.active,
                        'target_active': target_info.active,
                    })
                    break

        logger.info(f"Found {len(common_chains)} common chains between {source_exchange} and {target_exchange}")

        return common_chains


# Standalone test
async def main():
    """Test network verifier"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("="*100)
    print("NETWORK VERIFIER TEST")
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

    verifier = NetworkVerifier(credentials)

    # Test 1: Fetch networks for an exchange
    print("TEST 1: Fetch USDT networks for Bitget")
    print("-"*100)
    networks = await verifier.fetch_networks_for_currency('bitget', 'USDT')
    for net_id, net_info in sorted(networks.items()):
        print(f"  {net_info}")
    print()

    # Test 2: Verify chain availability
    print("TEST 2: Verify chain availability between exchanges")
    print("-"*100)

    test_cases = [
        ('binance', 'gateio', 'arb'),
        ('binance', 'gateio', 'trc'),
        ('kucoin', 'bitget', 'erc'),
    ]

    for source, target, chain in test_cases:
        available, src_net, tgt_net, src_fee, tgt_fee = await verifier.verify_chain_availability(
            source, target, chain
        )
        status = "[OK] AVAILABLE" if available else "[NO] NOT AVAILABLE"
        print(f"  {source} -> {target} via {chain}: {status}")
        if available:
            print(f"    Source: {src_net} (fee: ${src_fee})")
            print(f"    Target: {tgt_net}")
    print()

    # Test 3: Get common chains
    print("TEST 3: Get all common chains")
    print("-"*100)

    common = await verifier.get_common_chains('bitget', 'kucoin', 'USDT')
    for chain_info in common:
        print(f"  {chain_info['chain_code']:10} "
              f"{chain_info['source_network_id']:15} -> {chain_info['target_network_id']:15} "
              f"Fee: ${chain_info['fee']}")

    print()
    print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
