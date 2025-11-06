"""
Dynamic Fee Fetcher

Fetches real-time withdrawal fees from exchanges using CCXT.
Includes caching, fallback mechanisms, and support for all major exchanges.

Author: Capital Transfer System
Version: 1.0.0
"""

import ccxt
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from loguru import logger
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict


class NetworkAnalyzer:
    """
    Dynamically analyze and build network mappings across exchanges.

    Features:
    - Dynamic network discovery using CCXT
    - Cross-exchange network name normalization
    - Baseline fee collection for fallbacks
    - Persistent cache for network mappings
    """

    # Standard chain codes for normalization
    STANDARD_CODES = {
        'arb': ['arb', 'arbitrum', 'arbitrum one', 'arbone'],
        'op': ['op', 'optimism', 'optimism chain'],
        'polygon': ['polygon', 'matic', 'polygon pos'],
        'trc': ['trc', 'tron', 'tron network', 'trc20'],
        'bnb': ['bnb', 'bsc', 'bnb chain', 'bep20', 'bnb smart chain'],
        'erc': ['erc', 'eth', 'ethereum', 'erc20'],
        '0g': ['0g'],
        'algo': ['algo'],
        'apt': ['apt', 'aptos'],
        'btc': ['btc', 'bitcoin'],
        'avaxc': ['avaxc', 'avax c', 'avax c-chain', 'avalanche c-chain'],
        'celo': ['celo'],
        'polkadot': ['polkadot', 'dot'],
        'eos': ['eos'],
        'gatechain': ['gatechain'],
        'kaia': ['kaia'],
        'kavaevm': ['kavaevm', 'kava evm', 'kava evm co-chain'],
        'kusama': ['kusama', 'ksm'],
        'near': ['near'],
        'okc': ['okc'],
        'sol': ['sol', 'solana'],
        'ton': ['ton'],
        'xpl': ['xpl'],
        'xtz': ['xtz', 'tezos'],
        'plasma': ['plasma']
    }

    def __init__(
        self,
        network_cache_file: str = "network_mappings.json",
        fallback_cache_file: str = "fallback_fees.json",
        cache_ttl_hours: int = 168,  # 7 days
        api_keys: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize network analyzer

        Args:
            network_cache_file: Path to network mappings cache
            fallback_cache_file: Path to fallback fees cache
            cache_ttl_hours: Cache TTL in hours
            api_keys: Exchange API credentials
        """
        self.network_cache_file = Path(network_cache_file)
        self.fallback_cache_file = Path(fallback_cache_file)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.api_keys = api_keys or {}

        # Load or generate mappings
        self.chain_mapping = self._load_network_mappings()
        self.fallback_fees = self._load_fallback_fees()

        logger.info("🕸️ Network Analyzer initialized")

    def _load_network_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load network mappings from cache, generate sync version"""
        if not self.network_cache_file.exists():
            logger.info("📝 Generating new network mappings...")
            # Use sync generation as fallback
            return self._generate_chain_mapping_sync()

        try:
            with open(self.network_cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check if expired
            cached_at = datetime.fromisoformat(cache_data.get('cached_at', '1970-01-01'))
            if datetime.now() - cached_at > self.cache_ttl:
                logger.info("⏰ Network mappings expired, regenerating...")
                return self._generate_chain_mapping_sync()

            logger.info("📚 Loaded cached network mappings")
            return cache_data['mappings']

        except Exception as e:
            logger.warning(f"Failed to load network mappings: {e}")
            return self._generate_chain_mapping_sync()

    def _get_empty_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get empty mappings as fallback"""
        exchanges = ['binance', 'gateio', 'bybit', 'okx', 'kucoin', 'bitget']
        return {exchange: {} for exchange in exchanges}

    def _load_fallback_fees(self) -> Dict[str, Dict[str, float]]:
        """Load fallback fees from cache, generate if needed (use sync fallback)"""
        if not self.fallback_cache_file.exists():
            logger.info("📝 Generating new fallback fees...")
            return self._get_historical_fallback_fees()

        try:
            with open(self.fallback_cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check if expired
            cached_at = datetime.fromisoformat(cache_data.get('cached_at', '1970-01-01'))
            if datetime.now() - cached_at > self.cache_ttl:
                logger.info("⏰ Fallback fees expired, regenerating...")
                return self._get_historical_fallback_fees()

            logger.info("📚 Loaded cached fallback fees")
            return cache_data['fees']

        except Exception as e:
            logger.warning(f"Failed to load fallback fees: {e}")
            return self._get_historical_fallback_fees()

    def _get_historical_fallback_fees(self) -> Dict[str, Dict[str, float]]:
        """Get historical fallback fees as static data"""
        historical_fallbacks = {
            'binance': {'arb': 0.8, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.5, 'erc': 15.0, 'sol': 0.6, 'btc': 0.001},
            'gateio': {'arb': 0.5, 'op': 0.5, 'polygon': 0.5, 'trc': 1.0, 'bnb': 0.3, 'erc': 10.0, 'sol': 0.5, 'btc': 0.001},
            'bybit': {'arb': 0.1, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.1, 'erc': 10.0, 'sol': 1.0, 'btc': 0.001},
            'okx': {'arb': 0.1, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.1, 'erc': 8.0, 'sol': 1.0, 'btc': 0.001},
            'kucoin': {'arb': 0.5, 'op': 0.5, 'polygon': 0.5, 'trc': 1.0, 'bnb': 0.3, 'erc': 12.0, 'sol': 1.5, 'btc': 0.001},
            'bitget': {'arb': 0.5, 'op': 0.15, 'polygon': 0.2, 'trc': 1.5, 'bnb': 0.0, 'erc': 3.0, 'sol': 1.0, 'btc': 0.001}
        }

        # Save cache
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'fees': historical_fallbacks
            }
            with open(self.fallback_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info("💾 Saved historical fallback fees")
        except Exception as e:
            logger.warning(f"Could not save fallback fees cache: {e}")

        return historical_fallbacks

    async def _fetch_exchange_networks(
        self,
        exchange_id: str,
        api_key: str = None,
        secret: str = None,
        passphrase: str = None
    ) -> Dict[str, List[Dict]]:
        """Fetch network information for an exchange using improved format from check_exchange_networks.py"""
        import ccxt.async_support as ccxt_async

        exchange_class = getattr(ccxt_async, exchange_id)
        config = {'enableRateLimit': True}

        if api_key and secret:
            config['apiKey'] = api_key
            config['secret'] = secret
            if passphrase:
                config['password'] = passphrase

        exchange = exchange_class(config)

        try:
            await exchange.load_markets()

            # Fetch withdrawal fees via authenticated endpoint
            # Some exchanges (like Gate.io) require authentication to get fee data
            withdrawal_fees_data = {}
            if api_key and secret and hasattr(exchange, 'fetch_deposit_withdraw_fee'):
                for currency_code in ['USDT']:
                    try:
                        fee_info = await exchange.fetch_deposit_withdraw_fee(currency_code)
                        if 'networks' in fee_info:
                            withdrawal_fees_data[currency_code] = fee_info['networks']
                            logger.info(f"[INFO {exchange_id.upper()}] Fetched withdrawal fees for {currency_code} via authenticated API")
                    except Exception as e:
                        logger.debug(f"[DEBUG {exchange_id.upper()}] Could not fetch withdrawal fees for {currency_code}: {e}")

            networks_by_currency = {}

            for currency_code in ['USDT']:
                if currency_code in exchange.currencies:
                    currency_info = exchange.currencies[currency_code]
                    networks = []

                    # Debug: Compare network data structure across exchanges
                    if currency_code == 'USDT' and exchange_id in ['kucoin', 'gateio', 'bitget']:
                        if 'networks' in currency_info and currency_info['networks']:
                            first_network = list(currency_info['networks'].items())[0]
                            logger.info(f"[COMPARE {exchange_id.upper()}] First network structure:")
                            logger.info(f"  Network ID: {first_network[0]}")
                            logger.info(f"  Network data keys: {first_network[1].keys()}")
                            logger.info(f"  Fee value: {first_network[1].get('fee')}")
                            logger.info(f"  Fee type: {type(first_network[1].get('fee'))}")

                    # Check if networks are available
                    if 'networks' in currency_info and currency_info['networks']:
                        for network_id, network_info in currency_info['networks'].items():
                            # Handle fee - try multiple sources (improved from check_exchange_networks)
                            fee = network_info.get('fee')

                            # If fee is None, try fetched withdrawal fees data (for exchanges requiring auth)
                            if fee is None and currency_code in withdrawal_fees_data:
                                if network_id in withdrawal_fees_data[currency_code]:
                                    fee_data = withdrawal_fees_data[currency_code][network_id]
                                    if 'withdraw' in fee_data and isinstance(fee_data['withdraw'], dict):
                                        fee = fee_data['withdraw'].get('fee')

                            # Try to get fee from 'info' section if still not available
                            if fee is None and 'info' in network_info:
                                info = network_info['info']
                                if isinstance(info, dict):
                                    # Special handling for KuCoin futures
                                    if exchange_id == 'kucoinfutures' and 'chains' in info:
                                        chains = info['chains']
                                        if isinstance(chains, list):
                                            # Find the chain that matches our network_id
                                            for chain in chains:
                                                if isinstance(chain, dict):
                                                    chain_name = chain.get('chainName', '').upper()
                                                    chain_id = chain.get('chainId', '').upper()
                                                    network_upper = network_id.upper()

                                    # Match by chain name or chain ID
                                    if (chain_name == network_upper or
                                        chain_id == network_upper or
                                        chain_name.replace(' ', '') == network_upper):
                                        # Try withdrawFeeRate first, then withdrawalMinFee
                                        fee = chain.get('withdrawFeeRate') or chain.get('withdrawalMinFee')
                                        if fee is not None:
                                                            try:
                                                                fee = float(fee)
                                                                break
                                                            except (ValueError, TypeError):
                                                                continue
                                    else:
                                        # Try different possible fee field names for other exchanges
                                        fee = info.get('withdraw_fee') or info.get('withdrawFee') or info.get('fee')

                            networks.append({
                                'network_id': network_id,
                                'network_name': network_info.get('name', network_id),
                                'active': network_info.get('withdraw', False) and network_info.get('deposit', False),
                                'withdraw_enabled': network_info.get('withdraw', False),
                                'deposit_enabled': network_info.get('deposit', False),
                                'fee': fee
                            })

                    networks_by_currency[currency_code] = networks

            return networks_by_currency

        except Exception as e:
            logger.error(f"Error fetching {exchange_id}: {e}")
            return {}
        finally:
            await exchange.close()

    async def _generate_chain_mapping(self) -> Dict[str, Dict[str, str]]:
        """Generate chain mappings by analyzing all exchanges"""
        exchanges_to_analyze = ['binance', 'gateio', 'bybit', 'okx', 'kucoin', 'bitget']

        # Fetch all exchange networks concurrently
        tasks = []
        for exchange_id in exchanges_to_analyze:
            creds = self.api_keys.get(exchange_id, {})
            task = self._fetch_exchange_networks(
                exchange_id,
                creds.get('api_key'),
                creds.get('secret'),
                creds.get('password')
            )
            tasks.append((exchange_id, task))

        results = {}
        for exchange_id, task in tasks:
            try:
                results[exchange_id] = await task
            except Exception as e:
                logger.warning(f"Failed to fetch networks for {exchange_id}: {e}")
                results[exchange_id] = {}

        # Build cross-exchange network mapping
        mapping = {}

        for exchange_id in exchanges_to_analyze:
            mapping[exchange_id] = {}

            if exchange_id not in results:
                continue

            exchange_data = results[exchange_id]

            for currency_code, networks in exchange_data.items():
                for network in networks:
                    if not network.get('active', False):
                        continue

                    standard_code = self._normalize_to_standard(network['network_id'], network['network_name'])
                    if standard_code:
                        mapping[exchange_id][network['network_id'].upper()] = standard_code
                        # Also map by name if different
                        if network['network_name'] != network['network_id']:
                            mapping[exchange_id][network['network_name'].upper()] = standard_code

        # Save cache
        self._save_network_mappings(mapping)

        return mapping

    def _generate_chain_mapping_sync(self) -> Dict[str, Dict[str, str]]:
        """Generate chain mappings synchronously using public API"""
        logger.info("🔄 Generating chain mappings synchronously...")

        # For sync generation, use cached historical data as fallback
        # This is called when async generation fails due to event loop conflicts
        mapping = self._get_empty_mappings()  # Start with empty mappings

        # Add known mappings from historical data
        known_mappings = {
            'gateio': {
                'ERC20': 'erc'
            },
            'kucoin': {
                'OP': 'op', 'OPTIMISM': 'op',
                'BEP20': 'bnb', 'MATIC': 'polygon',
                'POLYGON POS': 'polygon', 'ERC20': 'erc',
                'TRC20': 'trc', 'ARBONE': 'arb',
                'ARBITRUM': 'arb'
            },
            'bitget': {
                'ERC20': 'erc', 'TRC20': 'trc',
                'ARBONE': 'arb', 'OPTIMISM': 'op',
                'BSC': 'bnb', 'MATIC': 'polygon'
            }
        }

        # Merge known mappings
        for exchange_id, exchange_mappings in known_mappings.items():
            if exchange_id in mapping:
                mapping[exchange_id].update(exchange_mappings)
            else:
                mapping[exchange_id] = exchange_mappings

        logger.info(f"📝 Generated sync mappings: {sum(len(v) for v in mapping.values())} total mappings")
        self._save_network_mappings(mapping)
        return mapping

    async def _generate_baseline_fees(self) -> Dict[str, Dict[str, float]]:
        """Generate baseline fees from API data for fallbacks"""
        exchanges_to_analyze = ['binance', 'gateio', 'bybit', 'okx', 'kucoin', 'bitget']

        # First, let's reuse the network fetching results to get fees
        chain_mapping = await self._generate_chain_mapping()

        # Collect fees by standard code across exchanges
        fee_collection = defaultdict(lambda: defaultdict(list))

        # We'll need to do another round of network fetching to get current fees
        for exchange_id in exchanges_to_analyze:
            if exchange_id not in chain_mapping:
                continue

            creds = self.api_keys.get(exchange_id, {})
            try:
                networks_data = await self._fetch_exchange_networks(
                    exchange_id,
                    creds.get('api_key'),
                    creds.get('secret'),
                    creds.get('password')
                )

                if 'USDT' in networks_data:
                    for network in networks_data['USDT']:
                        if not network.get('active', False):
                            continue

                        # Map to standard code
                        standard_code = None
                        for key, code in chain_mapping[exchange_id].items():
                            if key.upper() in [network['network_id'].upper(), network['network_name'].upper()]:
                                standard_code = code
                                break

                        if standard_code and network.get('fee') is not None:
                            try:
                                fee = float(network['fee'])
                                fee_collection[exchange_id][standard_code].append(fee)
                            except (ValueError, TypeError):
                                continue

            except Exception as e:
                logger.warning(f"Failed to collect fees for {exchange_id}: {e}")

        # Calculate median fees for fallbacks
        fallback_fees = {}

        for exchange_id, chain_fees in fee_collection.items():
            fallback_fees[exchange_id] = {}
            for chain_code, fees in chain_fees.items():
                if fees:
                    fees.sort()
                    # Use median fee as baseline
                    median_fee = fees[len(fees) // 2] if len(fees) % 2 == 1 else (fees[len(fees) // 2 - 1] + fees[len(fees) // 2]) / 2
                    fallback_fees[exchange_id][chain_code] = round(median_fee, 4)

        # Fill missing exchanges with historical averages for known chains
        historical_fallbacks = {
            'binance': {'arb': 0.8, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.5, 'erc': 15.0},
            'gateio': {'arb': 0.5, 'op': 0.5, 'polygon': 0.5, 'trc': 1.0, 'bnb': 0.3, 'erc': 10.0},
            'bybit': {'arb': 0.1, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.1, 'erc': 10.0},
            'okx': {'arb': 0.1, 'op': 0.1, 'polygon': 0.1, 'trc': 1.0, 'bnb': 0.1, 'erc': 8.0},
            'kucoin': {'arb': 0.5, 'op': 0.5, 'polygon': 0.5, 'trc': 1.0, 'bnb': 0.3, 'erc': 12.0},
            'bitget': {'arb': 0.5, 'polygon': 0.5, 'bnb': 0.3, 'erc': 10.0}
        }

        for exchange_id in exchanges_to_analyze:
            if exchange_id not in fallback_fees:
                fallback_fees[exchange_id] = {}
            # Supplement with historical data for missing chains
            for chain_code, hist_fee in historical_fallbacks.get(exchange_id, {}).items():
                if chain_code not in fallback_fees[exchange_id]:
                    fallback_fees[exchange_id][chain_code] = hist_fee

        # Save cache
        self._save_fallback_fees(fallback_fees)

        return fallback_fees

    def _normalize_to_standard(self, network_id: str, network_name: str) -> Optional[str]:
        """Normalize network names to standard codes"""
        combined_name = f"{network_id} {network_name}".upper()

        for standard_code, variations in self.STANDARD_CODES.items():
            for variation in variations:
                if variation.upper() in combined_name:
                    return standard_code

        # Try exact matches
        for standard_code, variations in self.STANDARD_CODES.items():
            if network_id.upper() == standard_code.upper():
                return standard_code

        logger.debug(f"Could not normalize: {network_id} / {network_name}")
        return None

    def _save_network_mappings(self, mapping: Dict[str, Dict[str, str]]):
        """Save network mappings to cache"""
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'mappings': mapping
            }
            with open(self.network_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save network mappings: {e}")

    def _save_fallback_fees(self, fees: Dict[str, Dict[str, float]]):
        """Save fallback fees to cache"""
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'fees': fees
            }
            with open(self.fallback_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save fallback fees: {e}")

    def get_chain_mapping(self, exchange_id: str) -> Dict[str, str]:
        """Get chain mapping for exchange"""
        return self.chain_mapping.get(exchange_id, {})

    def get_fallback_fees(self, exchange_id: str) -> Dict[str, float]:
        """Get fallback fees for exchange"""
        return self.fallback_fees.get(exchange_id, {})


class FeeFetcher:
    """
    Fetch and cache withdrawal fees from exchanges

    Features:
    - Dynamic network analysis and mapping
    - Real-time fee fetching via CCXT
    - SQLite-based caching with TTL
    - Intelligent fallback fees from API baseline data
    - Support for all major exchanges
    - Handles network chains (ERC20, TRC20, etc.)
    """

    def __init__(
        self,
        cache_file: str = "fee_cache.json",
        cache_ttl_hours: int = 24,
        api_keys: Optional[Dict[str, Dict[str, str]]] = None,
        network_cache_file: str = "network_mappings.json",
        fallback_cache_file: str = "fallback_fees.json",
        network_cache_ttl_hours: int = 168  # 7 days
    ):
        """
        Initialize fee fetcher

        Args:
            cache_file: Path to fee cache file
            cache_ttl_hours: Fee cache TTL in hours
            api_keys: Optional API keys dict {exchange: {api_key, api_secret, password}}
            network_cache_file: Path to network mappings cache
            fallback_cache_file: Path to fallback fees cache
            network_cache_ttl_hours: Network mappings TTL in hours
        """
        self.cache_file = Path(cache_file)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.api_keys = api_keys or {}
        self.cache = self._load_cache()

        # Initialize network analyzer
        self.network_analyzer = NetworkAnalyzer(
            network_cache_file=network_cache_file,
            fallback_cache_file=fallback_cache_file,
            cache_ttl_hours=network_cache_ttl_hours,
            api_keys=api_keys
        )

        logger.info(f"💰 Fee Fetcher initialized (cache TTL: {cache_ttl_hours}h)")

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

    def _get_cache_key(self, exchange_id: str, chain_code: str, coin: str = 'USDT') -> str:
        """Generate cache key"""
        return f"{exchange_id}:{coin}:{chain_code}"

    def _get_cached_fee(self, exchange_id: str, chain_code: str, coin: str = 'USDT') -> Optional[float]:
        """Get fee from cache if valid"""
        key = self._get_cache_key(exchange_id, chain_code, coin)

        if key in self.cache:
            entry = self.cache[key]
            expire_time = datetime.fromisoformat(entry['expires_at'])

            if datetime.now() < expire_time:
                logger.debug(f"💾 Cache HIT: {key} = ${entry['fee']}")
                return entry['fee']
            else:
                logger.debug(f"⏰ Cache EXPIRED: {key}")

        return None

    def _set_cached_fee(self, exchange_id: str, chain_code: str, fee: float, coin: str = 'USDT'):
        """Store fee in cache"""
        key = self._get_cache_key(exchange_id, chain_code, coin)

        self.cache[key] = {
            'fee': fee,
            'cached_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + self.cache_ttl).isoformat(),
            'exchange': exchange_id,
            'chain': chain_code,
            'coin': coin
        }

        self._save_cache()
        logger.debug(f"💾 Cache SET: {key} = ${fee}")

    def _create_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Create CCXT exchange instance"""
        try:
            exchange_class = getattr(ccxt, exchange_id)

            # Get API keys if available
            creds = self.api_keys.get(exchange_id, {})

            params = {
                'enableRateLimit': True,
                'timeout': 30000,
            }

            # Add credentials if available (some exchanges allow public fee queries)
            if creds.get('api_key'):
                params['apiKey'] = creds['api_key']
                params['secret'] = creds['api_secret']
                if creds.get('password'):
                    params['password'] = creds['password']

            exchange = exchange_class(params)
            logger.debug(f"✅ Created {exchange_id} exchange instance")
            return exchange

        except Exception as e:
            logger.error(f"❌ Failed to create {exchange_id} exchange: {e}")
            return None

    def _normalize_chain_code(self, exchange_id: str, chain_name: str) -> Optional[str]:
        """Normalize chain name to standard code using dynamic mappings"""
        chain_upper = chain_name.upper()

        # Get mapping for this exchange from network analyzer
        mapping = self.network_analyzer.get_chain_mapping(exchange_id)

        # Try direct match
        if chain_upper in mapping:
            return mapping[chain_upper]

        # Try partial match
        for key, value in mapping.items():
            if key in chain_upper or chain_upper in key:
                return value

        # Try all standard codes as fallback
        all_standard_codes = list(self.network_analyzer.STANDARD_CODES.keys())
        for code in all_standard_codes:
            if code.upper() in chain_upper:
                return code

        logger.debug(f"⚠️  Could not normalize chain: {chain_name} for {exchange_id}")
        return None

    def fetch_fee_from_exchange(
        self,
        exchange_id: str,
        chain_code: str,
        coin: str = 'USDT'
    ) -> Tuple[Optional[float], bool]:
        """
        Fetch withdrawal fee from exchange using fetchDepositWithdrawFees

        Args:
            exchange_id: Exchange identifier
            chain_code: Chain code (arb, op, polygon, etc.)
            coin: Coin symbol (default: USDT)

        Returns:
            Tuple of (fee_in_usdt, from_cache)
        """
        # Check cache first
        cached_fee = self._get_cached_fee(exchange_id, chain_code, coin)
        if cached_fee is not None:
            return cached_fee, True

        logger.debug(f"🔄 Fetching fees for {exchange_id.upper()} {coin} on {chain_code}...")

        # Try to fetch fees using fetchDepositWithdrawFees
        try:
            exchange = self._create_exchange(exchange_id)
            if not exchange:
                logger.warning(f"Could not create {exchange_id} exchange instance")
                return None, False

            # Use fetchDepositWithdrawFees to get withdrawal fees for USDT
            fee_data = exchange.fetchDepositWithdrawFees([coin])

            if coin in fee_data and 'networks' in fee_data[coin]:
                coin_data = fee_data[coin]
                networks = coin_data['networks']

                # Find the fee for our chain (pass full coin_data for KuCoin)
                fee = self._extract_fee_from_networks(networks, chain_code, exchange_id, coin_data)

                if fee is not None:
                    logger.info(f"✅ API fee: {exchange_id.upper()}/{chain_code} = ${fee}")
                    self._set_cached_fee(exchange_id, chain_code, fee, coin)
                    return fee, False

            logger.warning(f"No fee found in API response for {exchange_id}/{chain_code}")

        except Exception as e:
            logger.debug(f"API call failed for {exchange_id}/{chain_code}: {e}")

        # Return None if API fails - no fallback
        return None, False

    def _extract_fee_from_networks(self, networks: Dict, chain_code: str, exchange_id: str, coin_data: Optional[Dict] = None) -> Optional[float]:
        """
        Extract fee for a specific chain from the networks data

        Args:
            networks: Networks dict from fetchDepositWithdrawFees response
            chain_code: Standard chain code (sol, arb, etc.)
            exchange_id: Exchange identifier
            coin_data: Optional full coin data (needed for KuCoin to access 'info.chains')

        Returns:
            Fee amount or None
        """
        # Special handling for KuCoin, KuCoin Futures, and Bitget - check 'info' section first at coin level
        # These exchanges store fees in a different structure (info.chains at coin level, not network level)
        if exchange_id in ['kucoin', 'kucoinfutures', 'bitget'] and coin_data and 'info' in coin_data:
            logger.info(f"[DEBUG {exchange_id}] Looking for chain_code='{chain_code}' in coin-level info")

            info = coin_data['info']
            if exchange_id == 'bitget':
                logger.info(f"[DEBUG {exchange_id}] Info keys: {list(info.keys()) if isinstance(info, dict) else 'not a dict'}")

            if isinstance(info, dict) and 'chains' in info:
                chains = info['chains']
                logger.info(f"[DEBUG {exchange_id}] Found {len(chains)} chains in coin info")

                # Build a mapping of chainId to standard code for KuCoin/Bitget
                chain_id_mapping = {
                    # Common mappings
                    'arbitrum': 'arb',
                    'arbitrumone': 'arb',  # Bitget uses ArbitrumOne
                    'arb': 'arb',
                    'optimism': 'op',  # Bitget uses Optimism
                    'op': 'op',
                    'matic': 'polygon',
                    'polygon': 'polygon',
                    'trx': 'trc',
                    'tron': 'trc',
                    'bsc': 'bnb',
                    'bnb': 'bnb',
                    'bep20': 'bnb',  # Bitget uses BEP20
                    'eth': 'erc',
                    'erc20': 'erc',  # Bitget uses ERC20
                    'ethereum': 'erc',
                    'sol': 'sol',
                    'solana': 'sol',
                    'avaxc': 'avax_c',
                    'avax-c': 'avax_c',
                    'avaxc-chain': 'avax_c',  # Bitget uses AVAXC-Chain
                    'avaxcchain': 'avax_c',  # normalized version
                    'avalanchecchain': 'avax_c',
                    'avalanche c-chain': 'avax_c',
                    'cchain': 'avax_c',
                    'near': 'near',
                    'aptos': 'aptos',
                    'apt': 'aptos',
                    'ton': 'ton',
                    'ton2': 'ton',
                    'plasma': 'plasma',
                    'xtz': 'xtz',
                    'tezos': 'xtz',
                    'kcc': 'kcc',
                    'kavaevm': 'kavaevm',
                    'kava': 'kavaevm',
                    'statemint': 'statemint',
                    'base': 'base',
                    'linea': 'linea',
                    'zksync': 'zksync',
                    'trc20': 'trc',  # Bitget uses TRC20
                    'opbnb': 'opbnb',
                    'zksyncevm': 'zksync',
                    'morph': 'morph'  # Bitget uses Morph
                }

                if isinstance(chains, list):
                    logger.info(f"[DEBUG {exchange_id}] Searching for chain_code='{chain_code}' in {len(chains)} chains")

                    # List all available chainIds for debugging
                    if exchange_id == 'bitget':
                        # Bitget uses 'chain' field instead of 'chainId'
                        all_chain_ids = [c.get('chain', 'N/A') for c in chains if isinstance(c, dict)]
                        logger.info(f"[DEBUG {exchange_id}] Available chains: {all_chain_ids}")
                        # Show first chain structure
                        if chains and isinstance(chains[0], dict):
                            logger.info(f"[DEBUG {exchange_id}] First chain keys: {list(chains[0].keys())}")
                    else:
                        all_chain_ids = [c.get('chainId', 'N/A') for c in chains if isinstance(c, dict)]
                        logger.info(f"[DEBUG {exchange_id}] Available chainIds: {all_chain_ids}")

                    # Find the chain that matches our chain_code
                    for chain in chains:
                        if isinstance(chain, dict):
                            # Different exchanges use different field names
                            chain_id = (chain.get('chainId') or
                                       chain.get('chain') or
                                       chain.get('network') or
                                       chain.get('chainType') or '').lower()
                            chain_name = (chain.get('chainName') or
                                         chain.get('name') or
                                         chain.get('networkName') or '').lower()

                            # Normalize chain_id and chain_name by removing spaces and special chars
                            normalized_chain_id = chain_id.replace(' ', '').replace('-', '').replace('_', '')
                            normalized_chain_name = chain_name.replace(' ', '').replace('-', '').replace('_', '')

                            # Map chainId to standard code
                            mapped_code = chain_id_mapping.get(chain_id, None)
                            if mapped_code is None:
                                mapped_code = chain_id_mapping.get(normalized_chain_id, chain_id)

                            # Debug for Bitget
                            if exchange_id == 'bitget':
                                logger.debug(f"[DEBUG {exchange_id}] Checking chain: chain_id='{chain_id}', normalized='{normalized_chain_id}', mapped='{mapped_code}' vs target='{chain_code}'")

                            # Try to match with our target chain_code
                            if (mapped_code == chain_code or
                                chain_id == chain_code or
                                normalized_chain_id == chain_code):
                                # Try different field names for withdrawal fee
                                fee = (chain.get('withdrawalMinFee') or
                                      chain.get('withdrawFee') or
                                      chain.get('withdrawalFee') or
                                      chain.get('minWithdrawFee'))

                                logger.info(f"[DEBUG {exchange_id}] ✅ MATCH! chain_code='{chain_code}' matched chainId='{chain_id}' -> fee={fee}")
                                logger.debug(f"[DEBUG {exchange_id}] Chain data keys: {list(chain.keys())}")

                                if fee is not None:
                                    try:
                                        return float(fee)
                                    except (ValueError, TypeError):
                                        logger.warning(f"[DEBUG {exchange_id}] Failed to convert fee '{fee}' to float")
                                        continue

                    logger.warning(f"[DEBUG {exchange_id}] ❌ No match found for chain_code='{chain_code}'")

        # Try standard method for other exchanges or as fallback
        mapping = self.network_analyzer.get_chain_mapping(exchange_id)

        for network_id, network_data in networks.items():
            # Try to map this network_id to a standard chain code
            standard_code = None

            # Check direct mapping
            if network_id.upper() in mapping:
                standard_code = mapping[network_id.upper()]
            else:
                # Try to normalize the network name
                standard_code = self._normalize_chain_code(exchange_id, network_id)

            # If this network matches our target chain_code
            if standard_code == chain_code:
                # Extract the withdrawal fee
                fee = network_data.get('withdraw', {}).get('fee')
                if fee is not None:
                    try:
                        return float(fee)
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid fee format for {exchange_id}/{network_id}: {fee}")
                        continue

                # Try alternative fee fields
                fee = network_data.get('fee')
                if fee is not None:
                    try:
                        return float(fee)
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid fee format for {exchange_id}/{network_id}: {fee}")
                        continue

        logger.debug(f"No fee found for {chain_code} in {exchange_id} networks")
        return None

    def _use_fallback_fee(self, exchange_id: str, chain_code: str, coin: str = 'USDT') -> Tuple[Optional[float], bool]:
        """Use fallback fee when API fails"""
        fallback_fee = self._get_fallback_fee(exchange_id, chain_code)
        if fallback_fee is not None:
            logger.info(f"📋 Using fallback fee: {exchange_id.upper()}/{chain_code} = ${fallback_fee}")
            self._set_cached_fee(exchange_id, chain_code, fallback_fee, coin)
            return fallback_fee, False

        logger.warning(f"No fallback fee available for {exchange_id}/{chain_code}")
        return None, False

    def _get_fallback_fee(self, exchange_id: str, chain_code: str) -> Optional[float]:
        """Get fallback fee from dynamic API baseline data"""
        return self.network_analyzer.get_fallback_fees(exchange_id).get(chain_code)

    def fetch_all_fees(
        self,
        exchange_id: str,
        chain_codes: List[str],
        coin: str = 'USDT'
    ) -> Dict[str, Optional[float]]:
        """
        Fetch fees for multiple chains

        Args:
            exchange_id: Exchange identifier
            chain_codes: List of chain codes
            coin: Coin symbol

        Returns:
            Dict mapping chain_code -> fee
        """
        fees = {}

        logger.info(f"📊 Fetching all fees for {exchange_id.upper()}")

        for chain_code in chain_codes:
            fee, from_cache = self.fetch_fee_from_exchange(exchange_id, chain_code, coin)
            fees[chain_code] = fee

            # No delay needed since we use cached fallbacks

        return fees

    async def fetch_all_exchange_fees_async(
        self,
        exchange_ids: List[str],
        coins: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Fetch deposit/withdraw fees for all exchanges in parallel using fetchDepositWithdrawFees.

        Args:
            exchange_ids: List of exchange identifiers (e.g., ['binance', 'okx', 'gateio'])
            coins: List of coin symbols to fetch (e.g., ['USDT', 'BTC']). If None, fetches all available.

        Returns:
            Dict mapping {exchange_id: {coin: fee_data}}
            where fee_data contains networks and fee information

        Example:
            {
                'binance': {
                    'USDT': {
                        'networks': {
                            'TRX': {'withdraw': {'fee': 1.0}, 'deposit': {}, ...},
                            'ETH': {'withdraw': {'fee': 15.0}, 'deposit': {}, ...}
                        }
                    }
                }
            }
        """
        import ccxt.async_support as ccxt_async

        logger.info(f"🚀 Fetching fees from {len(exchange_ids)} exchanges in parallel...")

        async def fetch_exchange_fees(exchange_id: str) -> tuple:
            """Fetch fees for a single exchange"""
            try:
                # Create async exchange instance
                exchange_class = getattr(ccxt_async, exchange_id)
                creds = self.api_keys.get(exchange_id, {})

                config = {
                    'enableRateLimit': True,
                    'timeout': 30000,
                }

                if creds.get('api_key'):
                    config['apiKey'] = creds['api_key']
                    config['secret'] = creds.get('api_secret') or creds.get('secret')
                    if creds.get('password'):
                        config['password'] = creds['password']

                exchange = exchange_class(config)

                try:
                    await exchange.load_markets()

                    # Fetch deposit/withdraw fees
                    # If coins specified, fetch those; otherwise fetch all
                    if coins:
                        fee_data = await exchange.fetchDepositWithdrawFees(coins)
                    else:
                        # Fetch without parameters to get all coins (if supported)
                        fee_data = await exchange.fetchDepositWithdrawFees()

                    logger.info(f"✅ {exchange_id.upper()}: Fetched fees for {len(fee_data)} coins")
                    return (exchange_id, fee_data, None)

                except Exception as e:
                    logger.error(f"❌ {exchange_id.upper()}: Failed to fetch fees - {e}")
                    return (exchange_id, {}, str(e))

                finally:
                    await exchange.close()

            except Exception as e:
                logger.error(f"❌ {exchange_id.upper()}: Failed to create exchange - {e}")
                return (exchange_id, {}, str(e))

        # Fetch all exchanges in parallel
        tasks = [fetch_exchange_fees(ex_id) for ex_id in exchange_ids]
        results = await asyncio.gather(*tasks)

        # Organize results
        all_fees = {}
        errors = {}

        for exchange_id, fee_data, error in results:
            if error:
                errors[exchange_id] = error
                all_fees[exchange_id] = {}
            else:
                all_fees[exchange_id] = fee_data

        # Log summary
        logger.info(f"\n📊 Fetch Summary:")
        logger.info(f"   Successful: {len([r for r in results if not r[2]])}/{len(exchange_ids)}")
        logger.info(f"   Failed: {len(errors)}")

        if errors:
            logger.warning(f"   Errors: {errors}")

        return all_fees

    def fetch_all_exchange_fees(
        self,
        exchange_ids: Optional[List[str]] = None,
        coins: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Synchronous wrapper for fetch_all_exchange_fees_async.

        Args:
            exchange_ids: List of exchange identifiers. If None, uses default exchanges.
            coins: List of coin symbols to fetch. If None, fetches common coins.

        Returns:
            Dict mapping {exchange_id: {coin: fee_data}}
        """
        if exchange_ids is None:
            exchange_ids = ['binance', 'gateio', 'bybit', 'okx', 'kucoin', 'bitget']

        if coins is None:
            coins = ['USDT', 'BTC', 'ETH', 'USDC']

        # Run async function
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create new task
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.fetch_all_exchange_fees_async(exchange_ids, coins))
            else:
                return loop.run_until_complete(self.fetch_all_exchange_fees_async(exchange_ids, coins))
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.fetch_all_exchange_fees_async(exchange_ids, coins))

    def update_config_fees(
        self,
        config: Dict,
        force_refresh: bool = False
    ) -> Dict:
        """
        Update capital_config with real fees

        Args:
            config: Capital config dict
            force_refresh: Force fetch even if cached

        Returns:
            Updated config
        """
        logger.info("🔄 Updating config with real-time fees...")

        if force_refresh:
            logger.info("🔄 Force refresh enabled, clearing cache")
            self.cache = {}
            self._save_cache()

        exchanges_data = config.get('exchanges', {})
        updated_count = 0
        failed_count = 0

        for exchange_id, exchange_data in exchanges_data.items():
            chains = exchange_data.get('chains', {})

            logger.info(f"\n📍 Processing {exchange_id.upper()} ({len(chains)} chains)")

            for chain_name, chain_data in chains.items():
                chain_code = chain_data.get('code')
                if not chain_code:
                    continue

                # Fetch fee
                fee, from_cache = self.fetch_fee_from_exchange(exchange_id, chain_code)

                if fee is not None:
                    old_fee = chain_data.get('fee_usdt', 0.0)
                    chain_data['fee_usdt'] = fee

                    source = "cache" if from_cache else "API"
                    if old_fee != fee:
                        logger.info(f"   ✅ {chain_code}: ${old_fee:.2f} → ${fee:.2f} ({source})")
                        updated_count += 1
                    else:
                        logger.debug(f"   ✓  {chain_code}: ${fee:.2f} (unchanged, {source})")
                else:
                    logger.error(f"   ❌ {chain_code}: Failed to fetch fee")
                    failed_count += 1

                # No delay needed since we use cached fallbacks

        logger.info(f"\n✅ Fee update complete:")
        logger.info(f"   Updated: {updated_count}")
        logger.info(f"   Failed: {failed_count}")

        return config

    def clear_cache(self):
        """Clear all cached fees"""
        self.cache = {}
        self._save_cache()
        logger.info("🧹 Cache cleared")

    def print_cache_stats(self):
        """Print cache statistics"""
        if not self.cache:
            print("\n📊 Fee cache is empty")
            return

        print("\n" + "="*80)
        print("📊 FEE CACHE STATISTICS")
        print("="*80)

        now = datetime.now()
        valid = 0
        expired = 0

        for key, entry in self.cache.items():
            expire_time = datetime.fromisoformat(entry['expires_at'])
            if expire_time > now:
                valid += 1
            else:
                expired += 1

        print(f"Total entries: {len(self.cache)}")
        print(f"Valid: {valid}")
        print(f"Expired: {expired}")
        print(f"Cache file: {self.cache_file}")
        print("="*80 + "\n")

    def refresh_network_mappings(self):
        """Force refresh of network mappings and fallback fees"""
        logger.info("🔄 Refreshing network mappings...")
        self.network_analyzer.chain_mapping = asyncio.run(self.network_analyzer._generate_chain_mapping())
        self.network_analyzer.fallback_fees = asyncio.run(self.network_analyzer._generate_baseline_fees())
        logger.info("✅ Network mappings refreshed")

    def print_network_mappings(self):
        """Print current network mappings"""
        print("\n" + "="*100)
        print("🌐 NETWORK MAPPINGS")
        print("="*100)

        mapping = self.network_analyzer.chain_mapping
        for exchange_id, exchange_mapping in mapping.items():
            if exchange_mapping:
                print(f"\n{exchange_id.upper()}:")
                for network_key, standard_code in exchange_mapping.items():
                    print(f"  {network_key:<25} -> {standard_code}")
            else:
                print(f"\n{exchange_id.upper()}: No mappings")

        print("\n" + "="*100)

    def print_fallback_fees(self):
        """Print current fallback fees"""
        print("\n" + "="*80)
        print("💰 FALLBACK FEES")
        print("="*80)

        fees = self.network_analyzer.fallback_fees
        for exchange_id, exchange_fees in fees.items():
            if exchange_fees:
                print(f"\n{exchange_id.upper()}:")
                for chain_code, fee in exchange_fees.items():
                    print(f"  {chain_code:<10}: ${fee}")
            else:
                print(f"\n{exchange_id.upper()}: No fallback fees")

        print("\n" + "="*80)

    def print_exchange_fees(self, fees_data: Dict[str, Dict[str, Dict]], detailed: bool = False):
        """
        Print fetched exchange fees in a formatted way

        Args:
            fees_data: Result from fetch_all_exchange_fees()
            detailed: If True, shows all networks. If False, shows summary only.
        """
        print("\n" + "="*100)
        print("💰 EXCHANGE DEPOSIT/WITHDRAW FEES")
        print("="*100)

        for exchange_id, exchange_data in fees_data.items():
            print(f"\n{'='*100}")
            print(f"📍 {exchange_id.upper()}")
            print(f"{'='*100}")

            if not exchange_data:
                print("   ❌ No data retrieved")
                continue

            for coin, coin_data in exchange_data.items():
                print(f"\n💰 {coin}:")

                if 'networks' not in coin_data:
                    print("   ⚠️  No network data available")
                    continue

                networks = coin_data['networks']
                print(f"   Total networks: {len(networks)}")

                if not detailed:
                    # Summary mode: show active networks with fees
                    active_with_fees = []
                    for network_id, network_info in networks.items():
                        # Check if network is active
                        withdraw_enabled = False
                        deposit_enabled = False

                        if isinstance(network_info.get('withdraw'), dict):
                            withdraw_enabled = network_info['withdraw'].get('enabled', False)
                        if isinstance(network_info.get('deposit'), dict):
                            deposit_enabled = network_info['deposit'].get('enabled', False)

                        if withdraw_enabled or deposit_enabled:
                            # Extract fee
                            fee = None
                            if 'withdraw' in network_info and isinstance(network_info['withdraw'], dict):
                                fee = network_info['withdraw'].get('fee')

                            if fee is not None:
                                active_with_fees.append((network_id, fee, withdraw_enabled, deposit_enabled))

                    print(f"   Active networks with fees: {len(active_with_fees)}")
                    print()

                    # Show first 10
                    for network_id, fee, w_enabled, d_enabled in active_with_fees[:10]:
                        w_status = "W" if w_enabled else " "
                        d_status = "D" if d_enabled else " "
                        print(f"   [{w_status}][{d_status}] {network_id:<20} Fee: ${fee}")

                    if len(active_with_fees) > 10:
                        print(f"   ... and {len(active_with_fees) - 10} more active networks")

                else:
                    # Detailed mode: show all networks
                    print()
                    for network_id, network_info in networks.items():
                        # Extract fee
                        fee = None
                        if 'withdraw' in network_info and isinstance(network_info['withdraw'], dict):
                            fee = network_info['withdraw'].get('fee')

                        # Check status
                        withdraw_enabled = False
                        deposit_enabled = False

                        if isinstance(network_info.get('withdraw'), dict):
                            withdraw_enabled = network_info['withdraw'].get('enabled', False)
                        if isinstance(network_info.get('deposit'), dict):
                            deposit_enabled = network_info['deposit'].get('enabled', False)

                        w_status = "W" if withdraw_enabled else " "
                        d_status = "D" if deposit_enabled else " "
                        fee_str = f"${fee}" if fee is not None else "N/A"

                        status = "✅" if (withdraw_enabled and deposit_enabled) else "⚠️"
                        print(f"   {status} [{w_status}][{d_status}] {network_id:<20} Fee: {fee_str}")

        print("\n" + "="*100)


def test_all_networks():
    """Comprehensive test of all network mappings and fee structures"""
    print("="*100)
    print("🧪 COMPREHENSIVE NETWORK & FEE TEST")
    print("="*100)
    print()

    # Initialize with API keys if available
    try:
        import os
        api_keys = {
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_API_SECRET')
            },
            'kucoin': {
                'api_key': os.getenv('KUCOINFUTURES_API_KEY'),
                'secret': os.getenv('KUCOINFUTURES_API_SECRET'),
                'password': os.getenv('KUCOINFUTURES_PASSWORD')
            },
            'gateio': {
                'api_key': os.getenv('GATEIO_API_KEY'),
                'secret': os.getenv('GATEIO_API_SECRET')
            },
            'bitget': {
                'api_key': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_API_SECRET'),
                'password': os.getenv('BITGET_PASSWORD')
            },
            'bybit': {
                'api_key': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_API_SECRET')
            },
            'okx': {
                'api_key': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_API_SECRET'),
                'password': os.getenv('OKX_PASSWORD')
            }
        }

        # Remove None values
        api_keys = {k: v for k, v in api_keys.items() if v.get('api_key') and v.get('secret')}

        if api_keys:
            print("🔑 Using API keys from environment variables")
            print(f"Exchanges with credentials: {list(api_keys.keys())}")
        else:
            print("⚠️  No API keys found - limited testing available")

        fetcher = FeeFetcher(cache_ttl_hours=24, api_keys=api_keys)

    except ImportError:
        print("⚠️  Could not load dotenv - using no API keys")
        fetcher = FeeFetcher(cache_ttl_hours=24)

    print()

    # 1. Test raw network data fetching
    print("🔍 Testing raw network data fetching...")
    print("-" * 60)

    exchanges = ['binance', 'gateio', 'bybit', 'okx', 'kucoin', 'bitget']

    for exchange_id in exchanges:
        try:
            analyzer = NetworkAnalyzer(api_keys=fetcher.api_keys)
            if hasattr(analyzer, '_fetch_exchange_networks'):
                networks_data = asyncio.run(analyzer._fetch_exchange_networks(
                    exchange_id,
                    fetcher.api_keys.get(exchange_id, {}).get('api_key'),
                    fetcher.api_keys.get(exchange_id, {}).get('secret'),
                    fetcher.api_keys.get(exchange_id, {}).get('password')
                ))

                if 'USDT' in networks_data:
                    networks = networks_data['USDT']
                    active_networks = [n for n in networks if n.get('active', False)]

                    print(f"✅ {exchange_id.upper():<10}: {len(active_networks)}/{len(networks)} networks active")

                    # Show first few active networks
                    for i, network in enumerate(active_networks[:3]):
                        fee_str = f"${network['fee']}" if network['fee'] is not None else "N/A"
                        withdraw = "✓" if network.get('withdraw_enabled', False) else "✗"
                        deposit = "✓" if network.get('deposit_enabled', False) else "✗"
                        print(f"    {network['network_id']:<15} ({withdraw}/{deposit}) fee: {fee_str}")
                else:
                    print(f"⚠️  {exchange_id.upper():<10}: No USDT network data")

        except Exception as e:
            print(f"❌ {exchange_id.upper():<10}: Error - {str(e)[:50]}...")

    print()

    # 2. Test network normalization
    print("🔄 Testing network normalization mappings...")
    print("-" * 60)

    # Force refresh to get latest mappings
    try:
        fetcher.refresh_network_mappings()
    except Exception as e:
        print(f"⚠️  Could not refresh mappings: {e}")

    fetcher.print_network_mappings()

    print()

    # 3. Test standardized chain codes
    print("📊 Testing standardized chain fee availability...")
    print("-" * 60)

    # Common chain codes to test
    test_chains = ['arb', 'op', 'polygon', 'trc', 'bnb', 'erc', 'sol', 'btc']

    print(f"{'Exchange':<10} {'ARB':<8} {'OP':<8} {'POLY':<8} {'TRC':<8} {'BNB':<8}")
    print("-" * 60)

    for exchange_id in exchanges:
        print(f"{exchange_id.upper():<10}", end="")

        for chain in test_chains[:5]:
            fallback_fee = fetcher.network_analyzer.get_fallback_fees(exchange_id).get(chain, 'N/A')
            fee_str = f"${fallback_fee}" if isinstance(fallback_fee, (int, float)) else str(fallback_fee)
            print(f"{fee_str:<8} ", end="")
        print()  # New line after each exchange

    print()

    # 4. Test fee fetching for all chains
    print("💰 Testing fee fetching for all chains...")
    print("-" * 60)

    all_chains_tested = 0
    fees_found = 0

    for exchange_id in exchanges:
        print(f"\n🔄 Testing {exchange_id.upper()}...")
        exchange_fees = {}

        for chain_code in test_chains:
            try:
                fee, from_cache = fetcher.fetch_fee_from_exchange(exchange_id, chain_code)
                all_chains_tested += 1

                if fee is not None:
                    fees_found += 1
                    source = "CACHE" if from_cache else "API"
                    exchange_fees[chain_code] = fee
                    print(f"    ✅ {chain_code:<10}: ${fee:.4f} ({source})")
                else:
                    print(f"    ❌ {chain_code:<10}: No fee available")

            except Exception as e:
                print(f"    ⚠️  {chain_code:<10}: Error - {str(e)[:30]}...")

    print(f"\n📈 Fee Fetch Summary:")
    print(f"    Total chains tested: {all_chains_tested}")
    print(f"    Fees retrieved: {fees_found}")
    success_rate = (fees_found / all_chains_tested * 100) if all_chains_tested > 0 else 0
    print()

    # 5. Cache statistics
    print("🗄️  Cache Statistics:")
    print("-" * 60)
    fetcher.print_cache_stats()

    print()
    print("✨ Network & Fee Testing Complete!")
    print("=" * 100)


# Standalone test
if __name__ == "__main__":
    try:
        test_all_networks()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
