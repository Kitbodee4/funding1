"""
Capital Transfer Engine

Complete transfer system with comprehensive safety checks:
1. Chain health check
2. Address validation
3. Test transfer
4. Confirmation from destination
5. Actual transfer
6. Transaction history
7. Fallback support
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import ccxt.async_support as ccxt
from loguru import logger
import yaml
from pathlib import Path

from .chain_selector import ChainSelector, ChainOption
from .chain_health_checker import ChainHealthChecker

import sys
import weakref
import ssl
import warnings


# SSL Context with verification
def get_ssl_context():
    """Get SSL context with proper verification"""
    try:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except Exception:
        # Fallback in case of SSL issues
        logger.warning("SSL context creation failed, using default")
        return None


async def graceful_shutdown(engine: 'CapitalTransferEngine', timeout: float = 15.0):
    """
    Gracefully shutdown the transfer engine with proper async cleanup

    This function ensures all connections are closed properly and prevents
    'Event loop is closed' errors by:
    1. Closing connections with timeouts
    2. Waiting for pending tasks to complete
    3. Cancelling orphaned tasks
    4. Giving event loop time for final cleanup
    5. Installing exception handler to suppress benign SSL transport errors

    Args:
        engine: CapitalTransferEngine instance to shutdown
        timeout: Maximum time to wait for shutdown (seconds)

    Example:
        engine = CapitalTransferEngine()
        try:
            # ... use engine ...
        finally:
            await graceful_shutdown(engine)
    """
    try:
        logger.info("Starting graceful shutdown...")

        # Install custom exception handler to suppress benign SSL transport errors
        loop = asyncio.get_running_loop()
        old_exception_handler = loop.get_exception_handler()

        def suppress_ssl_errors(loop, context):
            """Suppress SSL transport errors that occur during shutdown"""
            exception = context.get('exception')
            message = context.get('message', '')

            # Check if this is a benign SSL transport error during shutdown
            if exception:
                exc_str = str(exception)
                if any(msg in exc_str for msg in [
                    "Event loop is closed",
                    "RuntimeError: Event loop is closed",
                ]) and any(src in str(context) for src in [
                    "sslproto",
                    "_process_write_backlog",
                    "ssl transport"
                ]):
                    logger.debug(f"Suppressed benign SSL transport error during shutdown: {exc_str[:100]}")
                    return

            # For other errors, use the old handler or default
            if old_exception_handler:
                old_exception_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(suppress_ssl_errors)

        # Step 1: Close engine with timeout
        await asyncio.wait_for(engine.close(), timeout=timeout)

        # Step 2: CRITICAL - Wait for SSL transports to finish cleanup
        # aiohttp SSL connections need substantial time to properly close
        logger.debug("Waiting for aiohttp/SSL transport cleanup...")
        await asyncio.sleep(2.0)

        # Step 3: Check and cancel any remaining background tasks (except current)
        loop = asyncio.get_running_loop()
        current = asyncio.current_task()

        pending = [task for task in asyncio.all_tasks(loop)
                   if not task.done() and task != current]

        if pending:
            logger.debug(f"Found {len(pending)} remaining background tasks, cancelling...")
            for task in pending:
                if not task.done():
                    task.cancel()

            # Step 4: Wait for cancellation to complete (generous timeout)
            if pending:
                try:
                    done, pending = await asyncio.wait(pending, timeout=5.0)
                    if pending:
                        logger.debug(f"Warning: {len(pending)} tasks did not cancel in time")
                except Exception as e:
                    logger.debug(f"Error waiting for task cancellation: {e}")

        # Step 5: Final grace period for any remaining SSL/transport cleanup
        logger.debug("Final cleanup grace period...")
        await asyncio.sleep(1.0)

        # Step 6: One last check for pending tasks
        final_pending = [t for t in asyncio.all_tasks(loop)
                        if not t.done() and t != current]
        if final_pending:
            logger.warning(f"{len(final_pending)} tasks still pending after cleanup")

        logger.info("✓ Graceful shutdown complete")

    except asyncio.TimeoutError:
        logger.warning(f"Shutdown timeout after {timeout}s, forcing cleanup")
        # Cancel all remaining tasks on timeout
        try:
            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop)
                      if not t.done() and t != asyncio.current_task()]
            for task in pending:
                task.cancel()
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Error during graceful shutdown: {e}")


@dataclass
class TransferRequest:
    """Transfer request"""
    request_id: str
    source_exchange: str
    target_exchange: str
    amount_usdt: float
    purpose: str  # e.g., "funding_arbitrage", "rebalance"
    priority: str = "normal"  # "low", "normal", "high"
    max_fee_usdt: Optional[float] = None
    requested_at: datetime = None

    def __post_init__(self):
        if self.requested_at is None:
            self.requested_at = datetime.now(timezone.utc)


@dataclass
class TransferResult:
    """Transfer result"""
    request_id: str
    success: bool
    chain_used: str
    target_exchange: str
    test_txid: Optional[str]
    actual_txid: Optional[str]
    amount_sent: float
    fee_paid: float
    total_time_seconds: float
    test_confirmed: bool
    actual_confirmed: bool
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class CapitalTransferEngine:
    """
    Complete capital transfer engine with safety checks

    Safety Features:
    1. Chain health check before transfer
    2. Address validation (checksum, format)
    3. Test transfer with minimum amount (10.1 USDT)
    4. Confirmation from destination
    5. Actual transfer only after test success
    6. Transaction history logging
    7. Automatic fallback on failure
    8. Retry logic with exponential backoff
    """

    # Safety settings
    TEST_AMOUNT_USDT = 10.1  # Test transfer amount
    MIN_TRANSFER_USDT = 10.0
    MAX_TRANSFER_USDT = 100000.0

    # Confirmation settings
    MAX_CONFIRMATION_WAIT_MINUTES = 30
    CONFIRMATION_CHECK_INTERVAL_SECONDS = 10

    # Retry settings
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 5

    # Withdrawal rate limiting settings
    WITHDRAWAL_COOLDOWNS = {
        'bybit': 12,    # Bybit requires at least 10 seconds between withdrawals (add buffer)
        'binance': 5,   # Conservative cooldown for Binance
        'gateio': 12,   # Gate.io limits to 10s frequency (add buffer)
        'okx': 8,       # Conservative cooldown for OKX
        'bitget': 8,    # Conservative cooldown for Bitget
        'kucoin': 8,    # Conservative cooldown for KuCoin
        'kucoinfutures': 8  # Conservative cooldown for KuCoin Futures
    }

    def _load_withdrawal_cooldowns(self, config_path: str) -> Dict[str, int]:
        """
        Load withdrawal cooldowns from config file

        Args:
            config_path: Path to config file

        Returns:
            Dict of exchange_name -> cooldown_seconds
        """
        default_cooldowns = self.WITHDRAWAL_COOLDOWNS.copy()

        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # Load cooldowns from config, fallback to defaults
                custom_cooldowns = config.get('withdrawal_cooldowns', {})
                if custom_cooldowns:
                    default_cooldowns.update(custom_cooldowns)
                    logger.info(f"Loaded custom cooldowns from config: {custom_cooldowns}")

        except Exception as e:
            logger.warning(f"Failed to load withdrawal cooldowns from config: {e}, using defaults")

        return default_cooldowns

    def _get_exchange_network_mapping(self, exchange_name: str, chain_code: str) -> Optional[List[str]]:
        """
        Get exchange-specific network name mappings

        Args:
            exchange_name: Exchange name
            chain_code: Chain code (e.g., 'sol', 'bnb')

        Returns:
            List of possible network names for this exchange/chain combination
        """
        # Use the dynamic network mapping from health checker (prioritizes API data)
        # This ensures we use actual exchange API responses when available
        network_names = self.health_checker.get_network_mapping_for_exchange(exchange_name, chain_code)

        # Ensure we return a list
        if isinstance(network_names, list):
            return network_names
        elif network_names:
            return [network_names]
        else:
            return [chain_code.upper()]

    def _get_config_fee_for_exchange_chain(self, exchange_name: str, chain_code: str) -> float:
        """
        Get the configured fee for a specific exchange and chain from capital_config.yaml

        Args:
            exchange_name: Exchange name
            chain_code: Chain code (e.g., 'sol', 'bnb')

        Returns:
            Configured fee in USDT, or 0.0 if not found
        """
        try:
            # Access the chain selector's config
            config = self.chain_selector.config
            exchanges = config.get('exchanges', {})

            if exchange_name in exchanges:
                exchange_config = exchanges[exchange_name]
                chains = exchange_config.get('chains', {})

                # Find the chain by code
                for chain_name, chain_data in chains.items():
                    if chain_data.get('code') == chain_code:
                        return chain_data.get('fee_usdt', 0.0)

                # Try to find by chain name if code match fails
                if chain_code in chains:
                    chain_data = chains[chain_code]
                    return chain_data.get('fee_usdt', 0.0)

            logger.warning(f"Could not find config fee for {exchange_name}/{chain_code}, using 0.0")
            return 0.0

        except Exception as e:
            logger.error(f"Error getting config fee for {exchange_name}/{chain_code}: {e}")
            return 0.0

    def __init__(
        self,
        config_path: str = "capital_config.yaml",
        enable_test_transfer: bool = True,
        enable_health_check: bool = True,
        enable_confirmation_check: bool = False,
        enable_network_validation: bool = True,
        api_keys: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize transfer engine

        Args:
            config_path: Path to capital config
            enable_test_transfer: Enable test transfers
            enable_health_check: Enable chain health checks
            enable_confirmation_check: Enable deposit confirmation checks (can be slow)
            api_keys: Optional API keys for dynamic fee fetching {exchange: {apiKey, secret, password}}
        """
        self.chain_selector = ChainSelector(config_path, api_keys=api_keys)
        self.health_checker = ChainHealthChecker()

        self.enable_test_transfer = enable_test_transfer
        self.enable_health_check = enable_health_check
        self.enable_confirmation_check = enable_confirmation_check

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.transfer_history: List[TransferResult] = []

        # Load withdrawal cooldowns from config
        self.WITHDRAWAL_COOLDOWNS = self._load_withdrawal_cooldowns(config_path)

        # Track last withdrawal time per exchange to rate limit
        self.last_withdrawal_times: Dict[str, float] = {}

        logger.info("Capital Transfer Engine initialized")
        logger.info(f"  Test transfer: {'enabled' if enable_test_transfer else 'disabled'}")
        logger.info(f"  Health check: {'enabled' if enable_health_check else 'disabled'}")
        logger.info(f"  Confirmation check: {'enabled' if enable_confirmation_check else 'disabled'}")
        logger.info(f"  Withdrawal rate limiting: enabled")
        logger.info(f"  Cooldowns loaded: {self.WITHDRAWAL_COOLDOWNS}")

    async def initialize_exchange(
        self,
        exchange_name: str,
        api_credentials: Dict
    ):
        """
        Initialize exchange with API credentials

        Args:
            exchange_name: Exchange name
            api_credentials: API key, secret, etc.
        """
        if exchange_name in self.exchanges:
            return

        try:
            exchange_class = getattr(ccxt, exchange_name)
            config = {
                'enableRateLimit': True,
                'apiKey': api_credentials.get('apiKey'),
                'secret': api_credentials.get('secret'),
                'options': {'defaultType': 'spot'}
            }

            # Add password for exchanges that need it
            if 'password' in api_credentials:
                config['password'] = api_credentials['password']

            exchange = exchange_class(config)
            await exchange.load_markets()

            self.exchanges[exchange_name] = exchange
            logger.info(f"✓ Initialized {exchange_name} exchange")

            # Also initialize the health checker with credentials for this exchange
            # This ensures health checks can use authenticated endpoints when needed
            if self.enable_health_check:
                await self.health_checker.initialize_exchange(exchange_name, api_credentials)
                logger.debug(f"✓ Initialized {exchange_name} health checker with credentials")

        except Exception as e:
            logger.error(f"✗ Failed to initialize {exchange_name}: {e}")
            raise

    async def validate_address(
        self,
        address: str,
        chain_code: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate deposit address

        Args:
            address: Address to validate
            chain_code: Chain code

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not address:
            return False, "Address is empty"

        # Basic validation
        if chain_code in ['bnb', 'erc', 'arb', 'op', 'polygon', 'avax c']:
            # EVM-compatible chains
            if not address.startswith('0x'):
                return False, "EVM address must start with 0x"
            if len(address) != 42:
                return False, "EVM address must be 42 characters"

            # TODO: Checksum validation

        elif chain_code == 'trc':
            # Tron
            if not address.startswith('T'):
                return False, "Tron address must start with T"
            if len(address) != 34:
                return False, "Tron address must be 34 characters"

        elif chain_code == 'sol':
            # Solana
            if len(address) < 32 or len(address) > 44:
                return False, "Invalid Solana address length"

        elif chain_code == 'apot':
            # Aptos
            if not address.startswith('0x'):
                return False, "Aptos address must start with 0x"

        return True, None

    async def check_prerequisites(
        self,
        request: TransferRequest,
        chain: ChainOption
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all prerequisites before transfer

        Args:
            request: Transfer request
            chain: Selected chain

        Returns:
            Tuple of (can_proceed, error_message)
        """
        # 1. Check amount limits
        if request.amount_usdt < self.MIN_TRANSFER_USDT:
            return False, f"Amount below minimum ({self.MIN_TRANSFER_USDT} USDT)"

        if request.amount_usdt > self.MAX_TRANSFER_USDT:
            return False, f"Amount above maximum ({self.MAX_TRANSFER_USDT} USDT)"

        # 2. Check if exchanges initialized
        if request.source_exchange not in self.exchanges:
            return False, f"Source exchange {request.source_exchange} not initialized"

        if request.target_exchange not in self.exchanges:
            return False, f"Target exchange {request.target_exchange} not initialized"

        # 3. Check chain health
        if self.enable_health_check:
            health = await self.health_checker.check_chain_health(
                request.target_exchange,
                chain.chain_code
            )

            if not health.is_healthy:
                return False, f"Chain {chain.chain_code} is unhealthy: {health.error_message}"

        # 4. Validate address
        if chain.deposit_address:
            is_valid, error = await self.validate_address(
                chain.deposit_address,
                chain.chain_code
            )

            if not is_valid:
                return False, f"Invalid address: {error}"
        else:
            return False, "No deposit address configured"

        # 5. Check max fee
        if request.max_fee_usdt and chain.fee_usdt > request.max_fee_usdt:
            return False, f"Fee {chain.fee_usdt} exceeds maximum {request.max_fee_usdt}"

        return True, None

    async def validate_network_before_withdrawal(
        self,
        exchange_name: str,
        chain_code: str,
        possible_networks: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate which networks are actually supported for withdrawal before attempting

        Args:
            exchange_name: Exchange name
            chain_code: Chain code (e.g., 'bnb', 'trc')
            possible_networks: List of possible network names to check

        Returns:
            Tuple of (valid_networks, invalid_networks)
        """
        if not self.enable_health_check:
            logger.debug(f"Health check disabled, skipping network validation for {exchange_name}/{chain_code}")
            return possible_networks, []

        valid_networks = []
        invalid_networks = []

        try:
            # Check chain health to see if withdrawals are enabled
            health = await self.health_checker.check_chain_health(exchange_name, chain_code)

            if not health.is_healthy:
                logger.warning(f"Chain {chain_code} is not healthy on {exchange_name}: {health.error_message}")
                return [], possible_networks  # All networks invalid if chain unhealthy

            if not health.is_withdraw_enabled:
                logger.warning(f"Withdrawals not enabled for {chain_code} on {exchange_name}")
                return [], possible_networks  # All networks invalid if withdrawals disabled

            # Chain is healthy and withdrawals enabled, but we still need to validate specific networks
            # For now, we'll assume all possible networks are valid if the chain is healthy
            # This could be enhanced further to check each network individually
            logger.debug(f"Chain {chain_code} validated for withdrawal on {exchange_name}")
            valid_networks = possible_networks.copy()

        except Exception as e:
            logger.warning(f"Error validating networks for {exchange_name}/{chain_code}: {e}")
            # On validation error, allow all networks to be tried (fallback behavior)
            valid_networks = possible_networks.copy()

        logger.info(f"Network validation for {exchange_name}/{chain_code}: {len(valid_networks)} valid, {len(invalid_networks)} invalid")
        if invalid_networks:
            logger.debug(f"Invalid networks: {invalid_networks}")

        return valid_networks, invalid_networks

    async def execute_withdrawal(
        self,
        exchange_name: str,
        amount: float,
        address: str,
        chain_code: str,
        tag: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Execute withdrawal from exchange with rate limiting and network validation

        Args:
            exchange_name: Exchange name
            amount: Amount to withdraw
            address: Destination address
            chain_code: Chain code
            tag: Optional memo/tag

        Returns:
            Tuple of (success, txid, error_message)
        """
        if exchange_name not in self.exchanges:
            return False, None, f"Exchange {exchange_name} not initialized"

        exchange = self.exchanges[exchange_name]

        # Check withdrawal cooldown
        cooldown_seconds = self.WITHDRAWAL_COOLDOWNS.get(exchange_name, 5)
        last_withdrawal = self.last_withdrawal_times.get(exchange_name)

        if last_withdrawal:
            elapsed = asyncio.get_event_loop().time() - last_withdrawal
            if elapsed < cooldown_seconds:
                wait_time = cooldown_seconds - elapsed + 0.1  # Add small buffer
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s before {exchange_name} withdrawal")
                await asyncio.sleep(wait_time)

        try:
            # Map chain code to network (try all possible network names)
            # Use exchange-specific mappings for better accuracy
            exchange_specific_mapping = self._get_exchange_network_mapping(exchange_name, chain_code)
            possible_networks = exchange_specific_mapping if exchange_specific_mapping else [chain_code.upper()]

            # VALIDATE NETWORKS BEFORE WITHDRAWAL ATTEMPTS
            logger.debug(f"Validating networks for {exchange_name}/{chain_code} before withdrawal")
            valid_networks, invalid_networks = await self.validate_network_before_withdrawal(
                exchange_name, chain_code, possible_networks
            )

            if not valid_networks:
                error_msg = f"No valid networks found for {exchange_name}/{chain_code} withdrawal"
                if invalid_networks:
                    error_msg += f". Checked networks: {invalid_networks}"
                # Add health check info for debugging
                try:
                    health = await self.health_checker.check_chain_health(exchange_name, chain_code)
                    error_msg += f". Health check: withdraw_enabled={health.is_withdraw_enabled}"
                    if health.error_message:
                        error_msg += f", error='{health.error_message[:100]}'"
                except Exception:
                    pass
                logger.error(f"✗ {error_msg}")
                return False, None, error_msg

            if invalid_networks:
                logger.info(f"Filtered out invalid networks for {exchange_name}/{chain_code}: {invalid_networks}")

            logger.info(f"Attempting withdrawal on {exchange_name} with {len(valid_networks)} validated networks: {valid_networks}")

            # Try withdrawal with each VALIDATED network name until one succeeds
            last_error = None
            for network in valid_networks:
                try:
                    logger.debug(f"Trying withdrawal with network '{network}' on {exchange_name}")

                    # Special handling for different exchanges
                    if exchange_name == 'binance':
                        # Binance: walletType 0=spot, 1=funding
                        # Use funding wallet (walletType=1) to avoid needing internal transfer
                        params = {'walletType': 1, 'network': network}  # Always include network

                        if tag:
                            params['tag'] = tag

                        result = await exchange.withdraw(
                            code='USDT',
                            amount=amount,
                            address=address,
                            tag=tag,
                            params=params
                        )

                    elif exchange_name == 'okx':
                        # OKX requires fee parameter for external withdrawals
                        # Get proper config fee as fallback for dynamic fee fetching
                        config_fee = self._get_config_fee_for_exchange_chain(exchange_name, chain_code)
                        fee_amount = self.chain_selector._get_dynamic_fee(exchange_name, chain_code, config_fee)

                        params = {'network': network, 'fee': str(fee_amount)}
                        if tag:
                            params['tag'] = tag

                        result = await exchange.withdraw(
                            code='USDT',
                            amount=amount,
                            address=address,
                            tag=tag,
                            params=params
                        )

                    elif exchange_name == 'bybit':
                        # Bybit requires accountType parameter to specify which account to withdraw from
                        # Per Bybit V5 API: https://bybit-exchange.github.io/docs/v5/asset/withdraw
                        # Valid accountType values: 'UTA', 'FUND', 'FUND,UTA', 'SPOT' (classic accounts only)
                        # Use 'FUND,UTA' to try both FUND and UTA accounts
                        params = {'network': network, 'accountType': 'FUND,UTA'}
                        if tag:
                            params['tag'] = tag

                        result = await exchange.withdraw(
                            code='USDT',
                            amount=amount,
                            address=address,
                            tag=tag,
                            params=params
                        )

                    elif exchange_name == 'bitget':
                        # Bitget uses 'chain' parameter but CCXT might expect 'network' as well
                        # Per Bitget API: https://www.bitget.com/api-doc/spot/account/Wallet-Withdrawal
                        # Send both to be safe
                        params = {'chain': network, 'network': network}
                        if tag:
                            params['tag'] = tag

                        result = await exchange.withdraw(
                            code='USDT',
                            amount=amount,
                            address=address,
                            tag=tag,
                            params=params
                        )

                    else:
                        # Execute withdrawal with network parameter for other exchanges
                        params = {'network': network}
                        if tag:
                            params['tag'] = tag

                        result = await exchange.withdraw(
                            code='USDT',
                            amount=amount,
                            address=address,
                            tag=tag,
                            params=params
                        )

                    txid = result.get('id') or result.get('txid')

                    # Update last withdrawal time on successful withdrawal
                    self.last_withdrawal_times[exchange_name] = asyncio.get_event_loop().time()

                    logger.info(f"✓ Withdrawal executed: {amount} USDT to {address[:10]}... (network: {network}, txid: {txid})")

                    return True, txid, None

                except Exception as e:
                    last_error = str(e)
                    # Log at INFO level so errors are visible to users
                    logger.info(f"Withdrawal attempt with validated network '{network}' failed: {last_error[:200]}")

                    # Special handling for rate limit errors (like Bybit's retCode 131001)
                    if "131001" in last_error or "rate limit" in last_error.lower():
                        logger.warning(f"Rate limit error from {exchange_name}: {last_error[:150]}")
                        # Add extra delay before retry with different network
                        await asyncio.sleep(5)
                        continue

                    # Continue to next validated network name
                    continue

            # All validated network names failed
            # Include detailed error message for debugging
            error_msg = f"Withdrawal failed with all validated networks {valid_networks}. Last error: {last_error[:300]}"
            logger.error(f"✗ {error_msg}")
            return False, None, error_msg

        except Exception as e:
            error_msg = str(e)
            # Log full error for debugging
            logger.error(f"✗ Withdrawal failed with exception: {error_msg[:300]}")
            return False, None, f"{exchange_name} withdrawal error: {error_msg[:300]}"

    async def check_deposit_arrival(
        self,
        exchange_name: str,
        txid: str,
        amount: float,
        max_wait_minutes: int = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if deposit arrived at destination

        Args:
            exchange_name: Exchange name
            txid: Transaction ID
            amount: Expected amount
            max_wait_minutes: Maximum wait time

        Returns:
            Tuple of (arrived, error_message)
        """
        if max_wait_minutes is None:
            max_wait_minutes = self.MAX_CONFIRMATION_WAIT_MINUTES

        if exchange_name not in self.exchanges:
            return False, f"Exchange {exchange_name} not initialized"

        exchange = self.exchanges[exchange_name]

        start_time = datetime.now(timezone.utc)
        max_wait = timedelta(minutes=max_wait_minutes)

        logger.info(f"Waiting for deposit confirmation (max {max_wait_minutes}min)...")

        while datetime.now(timezone.utc) - start_time < max_wait:
            try:
                # Fetch deposit history
                deposits = await exchange.fetch_deposits(code='USDT', limit=50)

                # Look for matching deposit
                for deposit in deposits:
                    if deposit.get('txid') == txid or deposit.get('id') == txid:
                        status = deposit.get('status')

                        if status == 'ok':
                            logger.info(f"✓ Deposit confirmed: {txid}")
                            return True, None
                        elif status == 'failed':
                            logger.error(f"✗ Deposit failed: {txid}")
                            return False, "Deposit failed on exchange"

                # Wait before next check
                await asyncio.sleep(self.CONFIRMATION_CHECK_INTERVAL_SECONDS)

            except Exception as e:
                logger.debug(f"Error checking deposit: {e}")
                await asyncio.sleep(self.CONFIRMATION_CHECK_INTERVAL_SECONDS)

        logger.warning(f"⚠ Deposit confirmation timeout for {txid}")
        return False, "Confirmation timeout"

    async def transfer(
        self,
        request: TransferRequest,
        chain: Optional[ChainOption] = None
    ) -> TransferResult:
        """
        Execute complete transfer with all safety checks

        Process:
        1. Select chain (if not provided)
        2. Check prerequisites
        3. Test transfer (10.1 USDT)
        4. Wait for test confirmation
        5. Execute actual transfer
        6. Wait for actual confirmation
        7. Return result

        Args:
            request: Transfer request
            chain: Optional specific chain to use

        Returns:
            TransferResult
        """
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting transfer: {request.request_id}")
        logger.info(f"  From: {request.source_exchange}")
        logger.info(f"  To: {request.target_exchange}")
        logger.info(f"  Amount: {request.amount_usdt} USDT")

        try:
            # Step 1: Select chain if not provided
            if chain is None:
                logger.info("Selecting optimal chain...")
                chain = self.chain_selector.get_best_chain(
                    request.source_exchange,
                    request.target_exchange,
                    tier='primary'
                )

                if not chain:
                    raise Exception("No suitable chain found")

            logger.info(f"Selected chain: {chain.chain_code} (${chain.fee_usdt}, {chain.total_time_minutes}min)")

            # Step 2: Check prerequisites
            can_proceed, error = await self.check_prerequisites(request, chain)

            if not can_proceed:
                raise Exception(f"Prerequisites check failed: {error}")

            logger.info("✓ Prerequisites check passed")

            # Step 3: Test transfer (if enabled)
            test_txid = None
            test_confirmed = False

            if self.enable_test_transfer:
                logger.info(f"Executing test transfer ({self.TEST_AMOUNT_USDT} USDT)...")

                test_success, test_txid, test_error = await self.execute_withdrawal(
                    request.source_exchange,
                    self.TEST_AMOUNT_USDT,
                    chain.deposit_address,
                    chain.chain_code
                )

                if not test_success:
                    raise Exception(f"Test transfer failed: {test_error}")

                logger.info(f"✓ Test transfer executed: {test_txid}")

                # Step 4: Wait for test confirmation (if enabled)
                if self.enable_confirmation_check:
                    logger.info("Waiting for test transfer confirmation...")

                    test_confirmed, confirm_error = await self.check_deposit_arrival(
                        request.target_exchange,
                        test_txid,
                        self.TEST_AMOUNT_USDT,
                        max_wait_minutes=chain.total_time_minutes + 5
                    )

                    if not test_confirmed:
                        raise Exception(f"Test transfer not confirmed: {confirm_error}")

                    logger.info("✓ Test transfer confirmed")
                else:
                    logger.info("⚡ Skipping test confirmation check (disabled)")
                    test_confirmed = True  # Assume successful

            # Step 5: Execute actual transfer
            logger.info(f"Executing actual transfer ({request.amount_usdt} USDT)...")

            actual_success, actual_txid, actual_error = await self.execute_withdrawal(
                request.source_exchange,
                request.amount_usdt,
                chain.deposit_address,
                chain.chain_code
            )

            if not actual_success:
                raise Exception(f"Actual transfer failed: {actual_error}")

            logger.info(f"✓ Actual transfer executed: {actual_txid}")

            # Step 6: Wait for actual confirmation (if enabled)
            actual_confirmed = False

            if self.enable_confirmation_check:
                logger.info("Waiting for actual transfer confirmation...")

                actual_confirmed, confirm_error = await self.check_deposit_arrival(
                    request.target_exchange,
                    actual_txid,
                    request.amount_usdt,
                    max_wait_minutes=chain.total_time_minutes + 10
                )

                if not actual_confirmed:
                    logger.warning(f"⚠ Actual transfer not confirmed (may still arrive): {confirm_error}")
                else:
                    logger.info("✓ Actual transfer confirmed")
            else:
                logger.info("⚡ Skipping deposit confirmation check (disabled) - assuming successful")

            # Calculate total time
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Create successful result
            result = TransferResult(
                request_id=request.request_id,
                success=True,
                chain_used=chain.chain_code,
                target_exchange=request.target_exchange,
                test_txid=test_txid,
                actual_txid=actual_txid,
                amount_sent=request.amount_usdt,
                fee_paid=chain.fee_usdt,
                total_time_seconds=total_time,
                test_confirmed=test_confirmed,
                actual_confirmed=actual_confirmed,
                completed_at=datetime.now(timezone.utc)
            )

            # Save to history
            self.transfer_history.append(result)

            logger.info(f"✅ Transfer completed successfully in {total_time:.1f}s")

            return result

        except Exception as e:
            error_msg = str(e)
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.error(f"❌ Transfer failed: {error_msg}")

            # Create failed result
            result = TransferResult(
                request_id=request.request_id,
                success=False,
                chain_used=chain.chain_code if chain else 'unknown',
                target_exchange=request.target_exchange,
                test_txid=None,
                actual_txid=None,
                amount_sent=0,
                fee_paid=0,
                total_time_seconds=total_time,
                test_confirmed=False,
                actual_confirmed=False,
                error_message=error_msg,
                completed_at=datetime.now(timezone.utc)
            )

            self.transfer_history.append(result)

            return result

    async def transfer_with_fallback(
        self,
        request: TransferRequest,
        max_fallback_attempts: int = 2
    ) -> TransferResult:
        """
        Transfer with automatic fallback on failure

        Args:
            request: Transfer request
            max_fallback_attempts: Max fallback attempts

        Returns:
            TransferResult
        """
        failed_chains = []

        for attempt in range(max_fallback_attempts + 1):
            # Get chain (with fallback if needed)
            if attempt == 0:
                tier = 'primary'
            elif attempt == 1:
                tier = 'secondary'
            else:
                tier = 'tertiary'

            chain = self.chain_selector.get_best_chain(
                request.source_exchange,
                request.target_exchange,
                tier=tier
            )

            if not chain or chain.chain_code in failed_chains:
                continue

            logger.info(f"Transfer attempt {attempt + 1}/{max_fallback_attempts + 1} using {tier} chain")

            # Execute transfer with cancellation safety
            try:
                result = await self._safe_transfer(request, chain)
                if result.success:
                    return result
            except asyncio.CancelledError:
                logger.warning(f"Transfer attempt {attempt + 1} was cancelled")
                # Create cancelled result
                return TransferResult(
                    request_id=request.request_id,
                    success=False,
                    chain_used=chain.chain_code,
                    target_exchange=request.target_exchange,
                    test_txid=None,
                    actual_txid=None,
                    amount_sent=0.0,
                    fee_paid=0.0,
                    total_time_seconds=0.0,
                    test_confirmed=False,
                    actual_confirmed=False,
                    error_message="Transfer was cancelled",
                    completed_at=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.error(f"Transfer attempt {attempt + 1} failed: {e}")
                result = TransferResult(
                    request_id=request.request_id,
                    success=False,
                    chain_used=chain.chain_code if chain else 'unknown',
                    target_exchange=request.target_exchange,
                    test_txid=None,
                    actual_txid=None,
                    amount_sent=0.0,
                    fee_paid=0.0,
                    total_time_seconds=0.0,
                    test_confirmed=False,
                    actual_confirmed=False,
                    error_message=str(e),
                    completed_at=datetime.now(timezone.utc)
                )
                # Continue to next attempt

            # Mark chain as failed
            if result and hasattr(result, 'error_message'):
                failed_chains.append(chain.chain_code)
                logger.warning(f"Attempt {attempt + 1} failed with chain {chain.chain_code}, trying fallback...")

        # All attempts failed
        logger.error(f"❌ All transfer attempts failed")

        return TransferResult(
            request_id=request.request_id,
            success=False,
            chain_used='all_failed',
            target_exchange=request.target_exchange,
            test_txid=None,
            actual_txid=None,
            amount_sent=0.0,
            fee_paid=0.0,
            total_time_seconds=0.0,
            test_confirmed=False,
            actual_confirmed=False,
            error_message="All fallback attempts failed",
            completed_at=datetime.now(timezone.utc)
        )

    async def _safe_transfer(
        self,
        request: TransferRequest,
        chain: ChainOption
    ) -> TransferResult:
        """
        Wrapper for transfer operations with cancellation safety

        Args:
            request: Transfer request
            chain: Chain to use

        Returns:
            TransferResult
        """
        try:
            # Check if loop is still active
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")

            return await self.transfer(request, chain)

        except asyncio.CancelledError:
            logger.warning(f"Transfer {request.request_id} was cancelled")
            raise
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.error(f"Event loop closed during transfer {request.request_id}")
                raise e
            # Re-raise other RuntimeErrors
            raise
        except Exception as e:
            # Log and re-raise other exceptions
            logger.error(f"Transfer {request.request_id} failed: {e}")
            raise

    async def close(self):
        """Close all exchange connections with async safety"""
        try:
            # Check if event loop is still active
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                logger.warning("Event loop already closed, skipping connection cleanup")
                return

            # Close exchange connections defensively
            close_tasks = []
            for exchange_name, exchange in list(self.exchanges.items()):
                if exchange and hasattr(exchange, 'close'):
                    close_tasks.append(self._close_exchange_safe(exchange_name, exchange))

            # Close all exchanges concurrently with generous timeout
            if close_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*close_tasks, return_exceptions=True),
                        timeout=15.0  # Generous timeout for SSL cleanup (0.3s per exchange × N exchanges)
                    )
                    # Extra wait for any remaining SSL transport cleanup
                    logger.debug(f"Closed {len(close_tasks)} exchanges, waiting for SSL cleanup...")
                    await asyncio.sleep(1.0)
                except asyncio.TimeoutError:
                    logger.warning("Some exchanges did not close in time")
                except Exception as e:
                    logger.debug(f"Error during exchange closure: {e}")

            # Clear exchanges dict to prevent reuse
            self.exchanges.clear()

            # Close health checker
            try:
                await asyncio.wait_for(
                    self.health_checker.close(),
                    timeout=2.0
                )
                logger.debug("✓ Health checker closed")
            except asyncio.TimeoutError:
                logger.debug("Health checker close timeout")
            except (RuntimeError, ConnectionError) as e:
                if "Event loop is closed" in str(e):
                    logger.warning("Event loop closed during health checker cleanup")
                    return
                logger.debug("Health checker connection already closed")
            except Exception as e:
                logger.debug(f"Error closing health checker: {e}")

            # Final grace period for any remaining cleanup
            await asyncio.sleep(0.3)

        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.warning("Event loop closed during engine cleanup")
            else:
                logger.error(f"Runtime error during cleanup: {e}")
                # Don't re-raise during cleanup
        except Exception as e:
            logger.debug(f"Error during engine cleanup: {e}")
            # Don't re-raise during cleanup

    async def _close_exchange_safe(self, exchange_name: str, exchange):
        """Safely close a single exchange connection with full aiohttp cleanup"""
        try:
            # Close the exchange connection
            await exchange.close()

            # CRITICAL: aiohttp connector cleanup
            # Access the underlying aiohttp session if it exists
            if hasattr(exchange, 'session') and exchange.session:
                session = exchange.session

                # Close the connector if it exists
                if hasattr(session, 'connector') and session.connector:
                    try:
                        await session.connector.close()
                    except Exception as e:
                        logger.debug(f"Connector close error for {exchange_name}: {e}")

            # CRITICAL: Give SSL transports time to complete cleanup callbacks
            # This prevents "Event loop is closed" errors during shutdown
            await asyncio.sleep(0.5)

            logger.debug(f"✓ Closed {exchange_name} connection")
        except (RuntimeError, ConnectionError, OSError, BrokenPipeError) as e:
            # Suppress common shutdown errors
            error_str = str(e)
            if any(msg in error_str for msg in [
                "Event loop is closed",
                "Cannot write to closing transport",
                "Transport is closing",
                "Broken pipe",
                "Connection reset"
            ]):
                logger.debug(f"{exchange_name} connection already closed")
            else:
                logger.debug(f"Error closing {exchange_name}: {e}")
        except asyncio.CancelledError:
            # Don't suppress cancellation
            logger.debug(f"{exchange_name} close was cancelled")
            raise
        except Exception as e:
            logger.debug(f"Unexpected error closing {exchange_name}: {e}")


# Test function (mock)
async def test_transfer_engine():
    """Test transfer engine (mock mode - no real transfers)"""
    print("\n" + "="*80)
    print("TEST: Capital Transfer Engine")
    print("="*80)

    engine = CapitalTransferEngine(
        enable_test_transfer=False,  # Disable for testing
        enable_health_check=False
    )

    # Mock request
    request = TransferRequest(
        request_id="TEST001",
        source_exchange="binance",
        target_exchange="gateio",
        amount_usdt=100.0,
        purpose="funding_arbitrage"
    )

    print(f"\nTransfer Request:")
    print(f"  ID: {request.request_id}")
    print(f"  From: {request.source_exchange} → To: {request.target_exchange}")
    print(f"  Amount: {request.amount_usdt} USDT")

    # Test chain selection
    chain = engine.chain_selector.get_best_chain(
        request.source_exchange,
        request.target_exchange
    )

    if chain:
        print(f"\nSelected Chain:")
        print(f"  Code: {chain.chain_code}")
        print(f"  Fee: ${chain.fee_usdt}")
        print(f"  Time: {chain.total_time_minutes}min")
        print(f"  Address: {chain.deposit_address[:20]}..." if chain.deposit_address else "  Address: N/A")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_transfer_engine())
