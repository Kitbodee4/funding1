"""
Chain Health Checker

Checks if blockchain networks are operational before transfers
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import ccxt.async_support as ccxt
from loguru import logger


@dataclass
class ChainHealth:
    """Chain health status"""
    chain_code: str
    exchange: str
    is_healthy: bool
    is_deposit_enabled: bool
    is_withdraw_enabled: bool
    last_checked: datetime
    error_message: Optional[str] = None
    estimated_arrival_time: Optional[int] = None  # minutes
    
    def __repr__(self):
        status = "✓ HEALTHY" if self.is_healthy else "✗ UNHEALTHY"
        return f"ChainHealth({self.exchange}/{self.chain_code}: {status})"


class ChainHealthChecker:
    """
    Check blockchain network health before transfers
    
    Features:
    - Check if deposits/withdrawals enabled
    - Verify network status
    - Cache health checks
    - Auto-fallback on unhealthy chains
    """
    
    # Chain name mapping - list of possible network names (each exchange uses different conventions)
    # Based on actual API responses from Binance, Gate.io, OKX, Bybit, Bitget, and KuCoin
    CHAIN_MAPPING = {
        'bnb': ['BSC', 'BEP20', 'BNB', 'BEP20(BSC)'],             # BNB Smart Chain - BSC is most common
        'trc': ['TRC20', 'TRON'],                                  # Tron
        'sol': ['SOL', 'SOLANA'],                                  # Solana
        'op': ['OP', 'OPTIMISM', 'OPTIMISTIC'],                   # Optimism
        'polygon': ['MATIC', 'POLYGON', 'PLASMA', 'plasma'],      # Polygon (note: some exchanges use lowercase)
        'avax_c': ['AVAXC', 'AVAXC-CHAIN', 'AVALANCHE'],          # Avalanche C-Chain
        'aptos': ['APT', 'APTOS'],                                 # Aptos - Fixed typo
        'arb': ['ARBONE', 'ARBITRUM', 'ARB'],                     # Arbitrum One
        'erc': ['ERC20', 'ETH', 'ETHEREUM']                        # Ethereum
    }

    # Dynamic chain mappings built from CCXT API responses
    # Maps exchange_name -> {chain_code: [network_names]}
    dynamic_chain_mappings: Dict[str, Dict[str, List[str]]] = {}

    def get_network_mapping_for_exchange(self, exchange_name: str, chain_code: str) -> List[str]:
        """
        Get network names for a specific exchange and chain code

        This method prioritizes dynamic mappings built from CCXT API,
        falling back to legacy mappings if dynamic ones are not available.

        Args:
            exchange_name: Name of the exchange
            chain_code: Chain code (e.g., 'sol', 'bnb')

        Returns:
            List of possible network names for this exchange/chain combination
        """
        # PRIORITY 1: Dynamic mappings built from exchange API (highest priority - real-time data)
        # Try to fetch from API if not already cached
        if exchange_name not in self.dynamic_chain_mappings:
            try:
                # Fetch network mappings from API asynchronously (this will be fast if already cached)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # If event loop is already running, skip dynamic fetch for now
                    logger.debug(f"Event loop already running, using fallback mappings for {exchange_name}")
                else:
                    # Fetch dynamic mappings from API
                    mappings = loop.run_until_complete(self.fetch_network_mappings_from_api(exchange_name))
                    if mappings:
                        self.dynamic_chain_mappings[exchange_name] = mappings
                        logger.debug(f"Fetched dynamic mappings for {exchange_name}: {len(mappings)} chains")
            except Exception as e:
                logger.debug(f"Could not fetch dynamic mappings for {exchange_name}: {e}")

        # Use dynamic mappings if available
        if exchange_name in self.dynamic_chain_mappings:
            exchange_mapping = self.dynamic_chain_mappings[exchange_name]

            # Check if we have a direct mapping for this chain_code
            if chain_code in exchange_mapping:
                network_names = exchange_mapping[chain_code]
                logger.info(f"✓ Using dynamic API mapping: {exchange_name}/{chain_code} -> {network_names}")
                return network_names
            else:
                logger.debug(f"Chain {chain_code} not found in dynamic mappings for {exchange_name}")
        else:
            logger.debug(f"No dynamic mappings available for {exchange_name}, using fallbacks")

        # PRIORITY 2: Exchange-specific mappings (manually verified fallback)
        # These are used when dynamic mappings are not available
        exchange_specific_mappings = {
            'binance': {
                'aptos': ['APT'],         # Binance uses 'APT' for Aptos
                'arb': ['ARBITRUM'],      # Binance uses 'ARBITRUM'
                'avax_c': ['AVAXC'],      # Binance uses 'AVAXC'
                'polygon': ['MATIC'],     # Binance uses 'MATIC'
                'op': ['OPTIMISM'],       # Binance uses 'OPTIMISM'
                'sol': ['SOL'],           # Binance uses 'SOL'
                'trc': ['TRX'],           # Binance uses 'TRX' not 'TRC20'
                'bnb': ['BSC'],           # Binance uses 'BSC'
                'erc': ['ETH'],           # Binance uses 'ETH'
                'near': ['NEAR'],         # Binance uses 'NEAR'
                'plasma': ['POLYGON'],    # Binance uses 'POLYGON' for Plasma
                'kavaevm': ['KAVA'],      # Binance uses 'KAVA'
            },
            'okx': {
                'sol': ['Solana'],      # OKX uses 'Solana' not 'SOL'
                'aptos': ['Aptos'],     # OKX uses 'Aptos'
                'arb': ['Arbitrum One'], # OKX uses 'Arbitrum One'
                'avax_c': ['Avalanche C-Chain'], # OKX uses full name
                'polygon': ['Polygon'],  # OKX uses 'Polygon'
                'op': ['Optimism'],      # OKX uses 'Optimism'
                'bnb': ['BSC'],          # OKX uses 'BSC'
                'trc': ['TRON'],         # OKX uses 'TRON'
                'erc': ['ERC20'],        # OKX uses 'ERC20'
            },
            'gateio': {
                'avax_c': ['AVAXC'],    # Gate.io uses 'AVAXC' not 'AVALANCHE'
                'polygon': ['MATIC'],   # Gate.io uses 'MATIC' not 'POLYGON'
                'arb': ['ARBITRUM'],    # Gate.io uses 'ARBITRUM' not 'ARB'
                'aptos': ['APT'],       # Gate.io uses 'APT' for Aptos
                'sol': ['SOL'],         # Gate.io uses 'SOL'
                'trc': ['TRC20'],       # Gate.io uses 'TRC20'
                'bnb': ['BSC'],         # Gate.io uses 'BSC'
                'op': ['OPTIMISM'],     # Gate.io uses 'OPTIMISM'
                'erc': ['ETH'],         # Gate.io uses 'ETH'
            },
            'kucoin': {
                'arb': ['ARBITRUM'],  # KuCoin uses 'ARBITRUM'
            },
            'bitget': {
                'avax_c': ['AVAXC-Chain'],  # Bitget uses 'AVAXC-Chain' (case-sensitive!)
                'arb': ['ArbitrumOne'],      # Bitget uses 'ArbitrumOne'
                'polygon': ['Polygon'],      # Bitget uses 'Polygon'
                'op': ['Optimism'],          # Bitget uses 'Optimism'
                'erc': ['ERC20'],            # Bitget uses 'ERC20'
                'bnb': ['BEP20'],            # Bitget uses 'BEP20'
                'trc': ['TRC20'],            # Bitget uses 'TRC20'
                'sol': ['SOL'],              # Bitget uses 'SOL'
                'aptos': ['Aptos'],          # Bitget uses 'Aptos'
                'ton': ['TON'],              # Bitget uses 'TON'
                'plasma': ['Plasma'],        # Bitget uses 'Plasma'
                'morph': ['Morph'],          # Bitget uses 'Morph'
            },
            'bybit': {
                'avax_c': ['AVAXC'],  # Bybit uses 'AVAXC'
                'arb': ['ARBITRUM'],  # Bybit uses 'ARBITRUM'
            }
        }

        # Check for exchange-specific mapping as fallback
        if exchange_name in exchange_specific_mappings:
            if chain_code in exchange_specific_mappings[exchange_name]:
                specific_networks = exchange_specific_mappings[exchange_name][chain_code]
                logger.info(f"✓ Using verified exchange-specific mapping: {exchange_name}/{chain_code} -> {specific_networks}")
                return specific_networks

        # PRIORITY 3: Fallback to legacy hardcoded mappings
        legacy_mapping = {
            'bnb': ['BSC', 'BEP20', 'BNB'],
            'trc': ['TRC20', 'TRON'],
            'sol': ['SOL', 'SOLANA'],
            'erc': ['ERC20', 'ETH'],
            'polygon': ['MATIC', 'POLYGON'],
            'arb': ['ARB', 'ARBITRUM'],
            'op': ['OP', 'OPTIMISM'],
            'avax_c': ['AVAXC', 'AVALANCHE'],
            'base': ['BASE'],
            'ftm': ['FTM', 'FANTOM'],
        }

        # Use generic legacy mapping
        network_names = legacy_mapping.get(chain_code, [chain_code.upper()])
        logger.debug(f"Using legacy mapping: {exchange_name}/{chain_code} -> {network_names}")
        return network_names

    def __init__(self, cache_duration_minutes: int = 5):
        """
        Initialize health checker

        Args:
            cache_duration_minutes: How long to cache health checks
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.health_cache: Dict[str, ChainHealth] = {}
        self.exchanges: Dict[str, ccxt.Exchange] = {}

        # Try to populate dynamic_chain_mappings from FeeFetcher's network analyzer
        self._populate_dynamic_mappings()

        logger.info("Chain health checker initialized")

    def _populate_dynamic_mappings(self):
        """
        Populate dynamic chain mappings from FeeFetcher's NetworkAnalyzer
        This provides exchange-specific network names from CCXT API
        """
        try:
            from .fee_fetcher import FeeFetcher

            # Create temporary FeeFetcher to get network mappings
            fee_fetcher = FeeFetcher()

            # Get network mappings from NetworkAnalyzer
            if hasattr(fee_fetcher, 'network_analyzer'):
                analyzer = fee_fetcher.network_analyzer
                if hasattr(analyzer, 'chain_mapping') and analyzer.chain_mapping:
                    # Convert NetworkAnalyzer format to ChainHealthChecker format
                    # NetworkAnalyzer: {exchange: {network_name: standard_code}}
                    # ChainHealthChecker: {exchange: {standard_code: [network_names]}}

                    for exchange_id, network_map in analyzer.chain_mapping.items():
                        if exchange_id not in self.dynamic_chain_mappings:
                            self.dynamic_chain_mappings[exchange_id] = {}

                        # Reverse the mapping
                        for network_name, standard_code in network_map.items():
                            if standard_code not in self.dynamic_chain_mappings[exchange_id]:
                                self.dynamic_chain_mappings[exchange_id][standard_code] = []

                            # Add network name if not already present
                            if network_name not in self.dynamic_chain_mappings[exchange_id][standard_code]:
                                self.dynamic_chain_mappings[exchange_id][standard_code].append(network_name)

                    logger.debug(f"Populated dynamic mappings for {len(self.dynamic_chain_mappings)} exchanges")

        except Exception as e:
            logger.debug(f"Could not populate dynamic mappings: {e}")

    async def fetch_network_mappings_from_api(self, exchange_name: str) -> Dict[str, List[str]]:
        """
        Fetch network mappings directly from exchange API

        Uses fetchCurrencies() or fetchDepositWithdrawFees() depending on exchange support.
        This provides real-time network names for each chain from the exchange API.

        Args:
            exchange_name: Exchange identifier (e.g., 'bitget', 'bybit')

        Returns:
            Dict mapping standard chain codes to list of network names
            Example: {'avax_c': ['AVAX_CCHAIN'], 'sol': ['SOL'], ...}
        """
        try:
            if exchange_name not in self.exchanges:
                logger.warning(f"Exchange {exchange_name} not initialized, cannot fetch currencies")
                return {}

            exchange = self.exchanges[exchange_name]

            # Try fetchDepositWithdrawFees first (more accurate for withdrawal info)
            currencies = None
            if hasattr(exchange, 'fetch_deposit_withdraw_fees'):
                try:
                    logger.debug(f"Fetching network info via fetchDepositWithdrawFees for {exchange_name}")
                    currencies = await exchange.fetch_deposit_withdraw_fees(['USDT'])
                except Exception as e:
                    logger.debug(f"fetchDepositWithdrawFees failed for {exchange_name}: {e}")

            # Fallback to fetchCurrencies
            if not currencies and hasattr(exchange, 'fetch_currencies'):
                try:
                    logger.debug(f"Fetching network info via fetchCurrencies for {exchange_name}")
                    currencies = await exchange.fetch_currencies()
                except Exception as e:
                    logger.debug(f"fetchCurrencies failed for {exchange_name}: {e}")

            # Check if we got data
            if not currencies:
                logger.warning(f"No currency data fetched for {exchange_name}")
                return {}

            # Build mapping: standard_code -> [network_names]
            mapping = {}

            # Normalize network names to standard codes
            standard_codes_map = {
                'avax': 'avax_c',
                'avaxc': 'avax_c',
                'avalanche': 'avax_c',
                'cchain': 'avax_c',
                'bnb': 'bnb',
                'bsc': 'bnb',
                'bep20': 'bnb',
                'trc': 'trc',
                'trc20': 'trc',
                'tron': 'trc',
                'sol': 'sol',
                'solana': 'sol',
                'arb': 'arb',
                'arbitrum': 'arb',
                'arbone': 'arb',
                'op': 'op',
                'optimism': 'op',
                'polygon': 'polygon',
                'matic': 'polygon',
                'eth': 'erc',
                'erc20': 'erc',
                'ethereum': 'erc',
            }

            # Process USDT networks (most common for transfers)
            if 'USDT' in currencies:
                currency_info = currencies['USDT']

                if 'networks' in currency_info and currency_info['networks']:
                    for network_id, network_data in currency_info['networks'].items():
                        # Check if network is active
                        withdraw_enabled = network_data.get('withdraw', False)
                        deposit_enabled = network_data.get('deposit', False)

                        if not (withdraw_enabled or deposit_enabled):
                            continue

                        # Normalize network ID to standard code
                        network_lower = network_id.lower().replace('_', '').replace('-', '')
                        standard_code = standard_codes_map.get(network_lower)

                        if standard_code:
                            if standard_code not in mapping:
                                mapping[standard_code] = []

                            if network_id not in mapping[standard_code]:
                                mapping[standard_code].append(network_id)

            logger.info(f"✅ Fetched {len(mapping)} network mappings for {exchange_name} from API")

            # Update dynamic_chain_mappings
            self.dynamic_chain_mappings[exchange_name] = mapping

            return mapping

        except Exception as e:
            logger.error(f"Failed to fetch currencies for {exchange_name}: {e}")
            return {}
    
    async def initialize_exchange(self, exchange_name: str, api_credentials: Optional[Dict] = None):
        """
        Initialize exchange connection

        Args:
            exchange_name: Exchange name
            api_credentials: Optional API credentials (api_key, api_secret, password, etc.)
        """
        if exchange_name in self.exchanges:
            return

        try:
            exchange_class = getattr(ccxt, exchange_name)
            config = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }

            # Add API credentials if provided
            if api_credentials:
                if 'apiKey' in api_credentials:
                    config['apiKey'] = api_credentials['apiKey']
                if 'secret' in api_credentials:
                    config['secret'] = api_credentials['secret']
                if 'password' in api_credentials:
                    config['password'] = api_credentials['password']

                # Add any other credentials that might be passed
                for key, value in api_credentials.items():
                    if key not in ['api_key', 'api_secret', 'password'] and not key.startswith('_'):
                        config[key] = value

            exchange = exchange_class(config)
            await exchange.load_markets()

            self.exchanges[exchange_name] = exchange
            logger.info(f"✓ Initialized {exchange_name} for health checks" +
                       (" (with credentials)" if api_credentials else " (public only)"))

            # Fetch real-time network mappings from API
            await self.fetch_network_mappings_from_api(exchange_name)

        except Exception as e:
            logger.error(f"✗ Failed to initialize {exchange_name}: {e}")
    
    async def _fetch_currencies_with_retry(
        self,
        exchange,
        exchange_name: str,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Fetch currencies with retry logic for transient API failures

        Args:
            exchange: Exchange instance
            exchange_name: Exchange name for logging
            max_retries: Maximum number of retry attempts

        Returns:
            Dict of currency info or None if failed after all retries
        """
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Fetching currencies from {exchange_name} (attempt {attempt + 1}/{max_retries + 1})")
                currencies = await exchange.fetch_currencies()

                if currencies is not None:
                    logger.debug(f"Successfully fetched currencies from {exchange_name}")
                    return currencies
                else:
                    logger.debug(f"Exchange {exchange_name} returned None (attempt {attempt + 1})")

            except Exception as e:
                error_str = str(e).lower()

                # Check if error is retryable (network issues, timeouts, rate limits)
                is_retryable = any(keyword in error_str for keyword in [
                    'timeout', 'network', 'connection', 'rate limit',
                    'temporarily', 'unavailable', '429', '503', '502', '504'
                ])

                if is_retryable:
                    logger.warning(f"Retryable error from {exchange_name} (attempt {attempt + 1}): {str(e)[:100]}")
                else:
                    logger.warning(f"Non-retryable error from {exchange_name}: {str(e)[:100]}")
                    # For non-retryable errors, fail fast
                    return None

            # Wait before retry (exponential backoff)
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        logger.warning(f"Failed to fetch currencies from {exchange_name} after {max_retries + 1} attempts")
        return None

    async def _fetch_currencies_okx_fallback(self, exchange, max_retries: int = 2) -> Optional[Dict]:
        """
        OKX-specific fallback for fetch_currencies() when it returns None

        Uses private_get_asset_balances() to get currency info with retry logic

        Args:
            exchange: OKX exchange instance
            max_retries: Maximum number of retry attempts

        Returns:
            Dict of currency info or None if failed
        """
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"OKX fallback attempt {attempt + 1}/{max_retries + 1}")

                # Use OKX's asset balances endpoint as fallback
                response = await exchange.private_get_asset_balances()

                if not response or not isinstance(response, dict):
                    logger.debug(f"OKX asset balances response is invalid (attempt {attempt + 1})")
                    if attempt < max_retries:
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    return None

                data = response.get('data', [])
                if not data:
                    logger.debug(f"OKX asset balances data is empty (attempt {attempt + 1})")
                    if attempt < max_retries:
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    return None

                # Build currencies dict from asset balances
                currencies = {}

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    currency_code = item.get('ccy')
                    if not currency_code:
                        continue

                    # Create minimal currency info structure
                    currencies[currency_code] = {
                        'id': currency_code,
                        'code': currency_code,
                        'name': currency_code,
                        'active': True,
                        'fee': 0.0,
                        'precision': {'amount': 8, 'price': 8},
                        'limits': {
                            'amount': {'min': 0.000001, 'max': None},
                            'price': {'min': 0.000001, 'max': None}
                        },
                        'networks': {}  # Will be populated if needed
                    }

                logger.debug(f"OKX fallback succeeded: retrieved {len(currencies)} currencies")
                return currencies

            except Exception as e:
                logger.debug(f"OKX fallback attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.warning(f"OKX fallback failed after {max_retries + 1} attempts: {e}")
                return None

    async def check_chain_health(
        self,
        exchange_name: str,
        chain_code: str,
        force_refresh: bool = False
    ) -> ChainHealth:
        """
        Check if a chain is healthy for transfers

        Args:
            exchange_name: Exchange name
            chain_code: Chain code (e.g., 'bnb', 'trc')
            force_refresh: Force refresh cache

        Returns:
            ChainHealth object
        """
        cache_key = f"{exchange_name}:{chain_code}"

        # Check cache first
        if not force_refresh and cache_key in self.health_cache:
            cached = self.health_cache[cache_key]
            age = datetime.now(timezone.utc) - cached.last_checked

            if age < self.cache_duration:
                logger.debug(f"Using cached health for {cache_key}")
                return cached

        # Initialize exchange if needed
        await self.initialize_exchange(exchange_name)

        if exchange_name not in self.exchanges:
            # Exchange not available - but don't block withdrawals
            # Return healthy status with warning to allow attempts
            logger.warning(f"Exchange {exchange_name} not initialized, allowing withdrawal attempt")
            return ChainHealth(
                chain_code=chain_code,
                exchange=exchange_name,
                is_healthy=True,  # Changed: Allow withdrawal attempt
                is_deposit_enabled=True,
                is_withdraw_enabled=True,
                last_checked=datetime.now(timezone.utc),
                error_message="Exchange not initialized (allowing attempt)"
            )

        exchange = self.exchanges[exchange_name]

        try:
            # Get possible chain/network names (exchanges use different conventions)
            # SAFE CHAIN MAPPING ACCESS: Handle invalid mapping data
            possible_chain_names = self.CHAIN_MAPPING.get(chain_code, [chain_code.upper()])

            # Validate chain mapping result
            if possible_chain_names is None:
                possible_chain_names = [chain_code.upper()]
            elif not isinstance(possible_chain_names, list):
                possible_chain_names = [possible_chain_names]
            else:
                # Filter out None values from the list
                possible_chain_names = [name for name in possible_chain_names if name is not None]

            # Ensure we have at least one valid chain name
            if not possible_chain_names:
                possible_chain_names = [chain_code.upper()]

            # Check deposit/withdrawal status for USDT on this chain
            currency_code = 'USDT'

            # Fetch currency info with RETRY LOGIC
            currencies = await self._fetch_currencies_with_retry(exchange, exchange_name, max_retries=3)

            # CRITICAL FIX: Some exchanges return None from fetch_currencies()
            if currencies is None:
                logger.warning(f"Exchange {exchange_name} returned None from fetch_currencies() after retries")

                # OKX-SPECIFIC FALLBACK: Try alternative method for OKX
                if exchange_name == 'okx':
                    logger.info(f"Trying OKX-specific fallback for fetch_currencies()")
                    currencies = await self._fetch_currencies_okx_fallback(exchange)

                    if currencies is None:
                        logger.warning("OKX fallback also failed - allowing withdrawal attempt anyway")
                        return self._create_healthy_with_warning(exchange_name, chain_code, "API check failed but allowing withdrawal attempt")
                    else:
                        logger.info(f"OKX fallback succeeded, retrieved {len(currencies)} currencies")
                else:
                    # Changed: Don't block withdrawal, just log warning
                    logger.warning(f"API check failed for {exchange_name} - allowing withdrawal attempt anyway")
                    return self._create_healthy_with_warning(exchange_name, chain_code, "API check failed but allowing withdrawal attempt")

            if not isinstance(currencies, dict):
                logger.warning(f"Exchange {exchange_name} returned invalid currencies type: {type(currencies)}")
                return self._create_unhealthy(exchange_name, chain_code, "Invalid currencies data structure")

            if currency_code not in currencies:
                logger.warning(f"{currency_code} not found on {exchange_name}")
                return self._create_unhealthy(exchange_name, chain_code, "Currency not found")

            # SAFE CURRENCY INFO EXTRACTION: Handle None values from currencies dict
            currency_info = currencies.get(currency_code)

            # Validate currency_info structure with comprehensive null checking
            if currency_info is None:
                logger.warning(f"Currency info is None for {currency_code} on {exchange_name}")
                return self._create_unhealthy(exchange_name, chain_code, "Invalid currency data")

            if not isinstance(currency_info, dict):
                logger.warning(f"Currency info is not a dict for {currency_code} on {exchange_name}: {type(currency_info)}")
                return self._create_unhealthy(exchange_name, chain_code, "Invalid currency data structure")

            # Additional validation: ensure currency_info has required structure
            if not currency_info:
                logger.warning(f"Empty currency info dict for {currency_code} on {exchange_name}")
                return self._create_unhealthy(exchange_name, chain_code, "Empty currency data")

            # Check network-specific status with ROBUST NULL CHECKING
            networks = currency_info.get('networks', {})

            # CRITICAL FIX: Ensure networks is a dict (some exchanges return None)
            if networks is None:
                logger.warning(f"Networks is None for {exchange_name}/{chain_code} - exchange returned invalid data")
                return self._create_unhealthy(exchange_name, chain_code, "Exchange returned None for networks")

            if not isinstance(networks, dict):
                logger.warning(f"Networks is not a dict for {exchange_name}/{chain_code}: {type(networks)}")
                return self._create_unhealthy(exchange_name, chain_code, "Invalid networks data structure")

            # Ensure we have valid data before proceeding
            if not networks:
                logger.warning(f"Empty networks dict for {exchange_name}/{chain_code}")
                return self._create_unhealthy(exchange_name, chain_code, "No networks available")

            # Try to find matching network by trying all possible names
            network_key = None

            # First try exact match with all possible names
            for chain_name in possible_chain_names:
                if chain_name is None:
                    continue
                try:
                    # SAFE DICT ACCESS: Check if chain_name exists in networks
                    if isinstance(chain_name, str) and chain_name in networks:
                        network_key = chain_name
                        logger.debug(f"Found exact match: {chain_name} on {exchange_name}")
                        break
                except (TypeError, KeyError) as e:
                    logger.debug(f"TypeError during exact match for {chain_name}: {e}")
                    continue

            # If no exact match, try fuzzy matching with SAFE ITERATION
            if not network_key:
                # SAFE KEYS EXTRACTION: Handle case where networks.keys() might fail
                try:
                    network_keys = list(networks.keys())
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Cannot get network keys for {exchange_name}/{chain_code}: {e}")
                    return self._create_unhealthy(exchange_name, chain_code, "Invalid networks structure")

                # SAFE ITERATION: Check each network key individually
                for net_key in network_keys:
                    # Skip None or empty keys (some exchanges have malformed data)
                    if net_key is None or net_key == '':
                        continue

                    try:
                        # SAFE STRING CONVERSION: Handle non-string keys
                        net_key_str = str(net_key) if net_key is not None else ''
                        if not net_key_str:
                            continue
                        net_key_lower = net_key_str.lower()

                        # Try matching against all possible chain names
                        for chain_name in possible_chain_names:
                            if chain_name is None:
                                continue

                            try:
                                # SAFE STRING CONVERSION: Ensure both are strings
                                chain_name_str = str(chain_name)
                                chain_name_lower = chain_name_str.lower()

                                # SAFE STRING COMPARISON: Check types before comparison
                                if isinstance(net_key_lower, str) and isinstance(chain_name_lower, str):
                                    # Perform fuzzy matching
                                    if (chain_name_lower in net_key_lower or
                                        net_key_lower in chain_name_lower):
                                        network_key = net_key
                                        logger.debug(f"Found fuzzy match: {net_key} matches {chain_name} on {exchange_name}")
                                        break
                            except (TypeError, AttributeError) as e:
                                logger.debug(f"Error comparing chain_name {chain_name}: {e}")
                                continue

                        if network_key:
                            break

                    except (TypeError, AttributeError) as e:
                        # Skip keys that can't be processed
                        logger.debug(f"Skipping malformed network key on {exchange_name}: {net_key} ({e})")
                        continue

            # Final check: validate we found a valid network
            if network_key:
                try:
                    if isinstance(networks, dict) and network_key in networks:
                        network_info = networks.get(network_key)
                    else:
                        network_info = None
                except (TypeError, KeyError):
                    network_info = None
            else:
                network_info = None

            if network_info:

                # Ensure network_info is a dict
                if not isinstance(network_info, dict):
                    logger.warning(f"Invalid network info for {exchange_name}/{chain_code} (network_key: {network_key}): {type(network_info)}")
                    return self._create_unhealthy(exchange_name, chain_code, "Invalid network data")

                # Safely get deposit/withdraw status with type checking
                is_deposit_enabled = bool(network_info.get('deposit', False))
                is_withdraw_enabled = bool(network_info.get('withdraw', False))
                is_healthy = is_deposit_enabled and is_withdraw_enabled

                # Estimate arrival time (if available)
                estimated_time = None
                try:
                    info_dict = network_info.get('info')
                    if info_dict and isinstance(info_dict, dict):
                        estimated_time = info_dict.get('estimatedArrivalTime')
                except (TypeError, AttributeError):
                    # Ignore if info structure is malformed
                    pass

                health = ChainHealth(
                    chain_code=chain_code,
                    exchange=exchange_name,
                    is_healthy=is_healthy,
                    is_deposit_enabled=is_deposit_enabled,
                    is_withdraw_enabled=is_withdraw_enabled,
                    last_checked=datetime.now(timezone.utc),
                    estimated_arrival_time=estimated_time
                )

                # Cache result
                self.health_cache[cache_key] = health

                if is_healthy:
                    logger.info(f"✓ {cache_key} is healthy")
                else:
                    logger.warning(f"⚠ {cache_key} is unhealthy (deposit: {is_deposit_enabled}, withdraw: {is_withdraw_enabled})")

                return health
            else:
                # Log available networks for debugging
                try:
                    available_networks = list(networks.keys()) if networks else []
                except (TypeError, AttributeError):
                    available_networks = []

                logger.warning(f"Network not found on {exchange_name} for chain code '{chain_code}'")
                logger.debug(f"Tried names: {possible_chain_names}")
                logger.debug(f"Available networks: {available_networks}")

                # Safely create error message
                try:
                    chain_names_str = ', '.join(str(name) for name in possible_chain_names if name is not None)
                except (TypeError, AttributeError):
                    chain_names_str = str(possible_chain_names)

                return self._create_unhealthy(exchange_name, chain_code, f"Network not found (tried: {chain_names_str})")

        except Exception as e:
            # Log error but allow withdrawal attempt (graceful degradation)
            error_str = str(e)
            logger.error(f"Error checking {exchange_name}/{chain_code}: {error_str}")

            # Check if error is API-related (network, timeout, etc.)
            # If so, allow withdrawal attempt anyway
            error_lower = error_str.lower()
            is_api_error = any(keyword in error_lower for keyword in [
                'timeout', 'network', 'connection', 'rate limit', 'api',
                'temporarily', 'unavailable', '429', '503', '502', '504',
                'fetch', 'request', 'http'
            ])

            if is_api_error:
                logger.warning(f"API error for {exchange_name}/{chain_code}, allowing withdrawal attempt")
                return self._create_healthy_with_warning(
                    exchange_name,
                    chain_code,
                    f"Health check API error (allowing attempt): {error_str[:100]}"
                )
            else:
                # For non-API errors (logic errors, etc.), mark as unhealthy
                return self._create_unhealthy(exchange_name, chain_code, error_str)
    
    def _create_unhealthy(self, exchange: str, chain_code: str, error: str) -> ChainHealth:
        """Create unhealthy chain health object"""
        return ChainHealth(
            chain_code=chain_code,
            exchange=exchange,
            is_healthy=False,
            is_deposit_enabled=False,
            is_withdraw_enabled=False,
            last_checked=datetime.now(timezone.utc),
            error_message=error
        )

    def _create_healthy_with_warning(self, exchange: str, chain_code: str, warning: str) -> ChainHealth:
        """
        Create healthy chain health object with warning

        Used when health check fails but we want to allow withdrawal attempt anyway.
        This provides graceful degradation when API checks fail.
        """
        logger.warning(f"⚠ {exchange}/{chain_code}: {warning} - marking as healthy to allow attempt")
        return ChainHealth(
            chain_code=chain_code,
            exchange=exchange,
            is_healthy=True,  # Allow withdrawal attempt
            is_deposit_enabled=True,
            is_withdraw_enabled=True,
            last_checked=datetime.now(timezone.utc),
            error_message=warning
        )
    
    async def check_multiple_chains(
        self,
        exchange_name: str,
        chain_codes: List[str]
    ) -> Dict[str, ChainHealth]:
        """
        Check health of multiple chains
        
        Args:
            exchange_name: Exchange name
            chain_codes: List of chain codes
        
        Returns:
            Dict mapping chain_code to ChainHealth
        """
        results = {}
        
        for chain_code in chain_codes:
            health = await self.check_chain_health(exchange_name, chain_code)
            results[chain_code] = health
            await asyncio.sleep(0.1)  # Rate limit
        
        return results
    
    def get_healthy_chains(
        self,
        exchange_name: str,
        chain_codes: List[str] = None
    ) -> List[str]:
        """
        Get list of healthy chain codes
        
        Args:
            exchange_name: Exchange name
            chain_codes: Optional list to filter
        
        Returns:
            List of healthy chain codes
        """
        healthy = []
        
        for cache_key, health in self.health_cache.items():
            cached_exchange, cached_chain = cache_key.split(':')
            
            if cached_exchange == exchange_name:
                if chain_codes is None or cached_chain in chain_codes:
                    if health.is_healthy:
                        healthy.append(cached_chain)
        
        return healthy
    
    async def wait_for_chain_recovery(
        self,
        exchange_name: str,
        chain_code: str,
        max_wait_minutes: int = 30,
        check_interval_seconds: int = 60
    ) -> bool:
        """
        Wait for a chain to become healthy
        
        Args:
            exchange_name: Exchange name
            chain_code: Chain code
            max_wait_minutes: Maximum wait time
            check_interval_seconds: How often to check
        
        Returns:
            True if chain recovered, False if timeout
        """
        start_time = datetime.now(timezone.utc)
        max_wait = timedelta(minutes=max_wait_minutes)
        
        logger.info(f"Waiting for {exchange_name}/{chain_code} to recover (max {max_wait_minutes}min)")
        
        while datetime.now(timezone.utc) - start_time < max_wait:
            health = await self.check_chain_health(exchange_name, chain_code, force_refresh=True)
            
            if health.is_healthy:
                logger.info(f"✓ {exchange_name}/{chain_code} recovered!")
                return True
            
            await asyncio.sleep(check_interval_seconds)
        
        logger.warning(f"✗ {exchange_name}/{chain_code} did not recover in {max_wait_minutes}min")
        return False
    
    async def close(self):
        """Close all exchange connections"""
        import asyncio
        close_tasks = []
        for exchange in self.exchanges.values():
            if exchange and hasattr(exchange, 'close'):
                close_tasks.append(exchange.close())

        if close_tasks:
            try:
                await asyncio.gather(*close_tasks, return_exceptions=True)
                # Wait for SSL cleanup
                await asyncio.sleep(0.5)
            except Exception:
                pass  # Ignore errors during cleanup


# Standalone test
async def test_health_checker():
    """Test the health checker"""
    checker = ChainHealthChecker()
    
    # Test: Check Binance chains
    print("\n" + "="*80)
    print("TEST: Chain Health Check")
    print("="*80)
    
    chains_to_check = ['bnb', 'trc', 'sol', 'arb', 'polygon']
    
    print("\nChecking Binance chains...")
    results = await checker.check_multiple_chains('binance', chains_to_check)
    
    print("\nResults:")
    for chain_code, health in results.items():
        status = "✓" if health.is_healthy else "✗"
        print(f"  {status} {chain_code.upper()}: "
              f"Deposit={health.is_deposit_enabled}, Withdraw={health.is_withdraw_enabled}")
        
        if health.error_message:
            print(f"    Error: {health.error_message}")
    
    # Get healthy chains
    healthy = checker.get_healthy_chains('binance', chains_to_check)
    print(f"\nHealthy chains: {', '.join(healthy)}")
    
    await checker.close()


if __name__ == "__main__":
    asyncio.run(test_health_checker())
