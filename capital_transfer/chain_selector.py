"""
Chain Selector

Intelligent chain selection system with 3-tier strategy:
1. Primary: Cheapest (main)
2. Secondary: Fast & cheap (backup)
3. Tertiary: Reliable (emergency)
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# Import fee fetching infrastructure
try:
    from .fee_fetcher import FeeFetcher
except ImportError:
    # Fallback for direct execution
    from fee_fetcher import FeeFetcher


@dataclass
class ChainOption:
    """Chain transfer option"""
    exchange: str
    chain_name: str
    chain_code: str
    fee_usdt: float
    withdraw_time_minutes: int
    deposit_time_minutes: int
    total_time_minutes: int
    deposit_address: Optional[str]
    whitelisted: bool
    enabled: bool
    tier: str  # 'primary', 'secondary', 'tertiary'
    score: float
    
    def __repr__(self):
        return (f"ChainOption({self.exchange}/{self.chain_code}: "
                f"${self.fee_usdt} in {self.total_time_minutes}min)")


class ChainSelector:
    """
    Intelligent chain selection with 3-tier fallback strategy
    
    Features:
    - 3-tier selection (primary, secondary, tertiary)
    - Score-based ranking
    - Fallback support
    - Health check integration
    """
    
    # Scoring weights
    SCORE_WEIGHTS = {
        'fee': 0.50,  # 50% - most important
        'speed': 0.30,  # 30% - important
        'reliability': 0.20  # 20% - important for tertiary
    }
    
    # Tier definitions
    TIER_CRITERIA = {
        'primary': {
            'fee_weight': 0.70,
            'speed_weight': 0.20,
            'reliability_weight': 0.10
        },
        'secondary': {
            'fee_weight': 0.40,
            'speed_weight': 0.40,
            'reliability_weight': 0.20
        },
        'tertiary': {
            'fee_weight': 0.20,
            'speed_weight': 0.20,
            'reliability_weight': 0.60
        }
    }
    
    def __init__(self, config_path: str = "capital_config.yaml", api_keys: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize chain selector

        Args:
            config_path: Path to capital config YAML
            api_keys: Optional API keys for fee fetching {exchange: {api_key, secret, password}}
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.chains: List[ChainOption] = []

        # Initialize fee fetcher for dynamic fee updates
        self.fee_fetcher = FeeFetcher(
            cache_file="fee_cache.json",
            cache_ttl_hours=24,
            api_keys=api_keys
        )

        logger.info(f"Chain selector initialized with {len(self.config['exchanges'])} exchanges")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {'exchanges': {}}

    def _get_dynamic_fee(self, exchange: str, chain_code: str, fallback_fee: float) -> float:
        """
        Get dynamic fee from API with intelligent fallback

        Priority:
        1. API fetch (real-time)
        2. Historical fallback fees (from fee_fetcher)
        3. Config fallback (last resort)

        Args:
            exchange: Exchange name
            chain_code: Chain code (arb, op, polygon, etc.)
            fallback_fee: Fallback fee from config

        Returns:
            Fee in USDT
        """
        try:
            # Try to fetch fee from API
            fee, from_cache = self.fee_fetcher.fetch_fee_from_exchange(
                exchange_id=exchange,
                chain_code=chain_code,
                coin='USDT'
            )

            if fee is not None:
                source = "cache" if from_cache else "API"
                logger.debug(f"💰 {exchange}/{chain_code}: ${fee} ({source})")
                return fee
            else:
                # API failed, try historical fallback from fee_fetcher
                historical_fee = self.fee_fetcher._get_fallback_fee(exchange, chain_code)
                if historical_fee is not None:
                    logger.debug(f"💾 {exchange}/{chain_code}: ${historical_fee} (historical fallback)")
                    return historical_fee
                else:
                    # Last resort: use config fallback
                    logger.debug(f"⚠️  {exchange}/{chain_code}: ${fallback_fee} (config fallback)")
                    return fallback_fee

        except Exception as e:
            logger.warning(f"Error fetching fee for {exchange}/{chain_code}: {e}")
            # Try historical fallback before using config
            historical_fee = self.fee_fetcher._get_fallback_fee(exchange, chain_code)
            if historical_fee is not None:
                logger.debug(f"💾 {exchange}/{chain_code}: ${historical_fee} (historical fallback after error)")
                return historical_fee
            return fallback_fee
    
    def get_all_chains(self, source_exchange: str, target_exchange: str) -> List[ChainOption]:
        """
        Get all available chains for transfer between two exchanges
        
        Args:
            source_exchange: Source CEX
            target_exchange: Target CEX
        
        Returns:
            List of ChainOption
        """
        chains = []
        
        # Get chains available on both exchanges
        source_chains = self.config['exchanges'].get(source_exchange, {}).get('chains', {})
        target_chains = self.config['exchanges'].get(target_exchange, {}).get('chains', {})
        
        # Find common chains
        common_chain_codes = set()
        
        for chain_name, chain_data in source_chains.items():
            if chain_data.get('enabled', False):
                common_chain_codes.add(chain_data['code'])
        
        for chain_name, chain_data in target_chains.items():
            if chain_data.get('enabled', False) and chain_data['code'] in common_chain_codes:
                # Get source chain data
                source_chain = source_chains[chain_name]
                target_chain = chain_data
                
                # Calculate total time and fee
                total_time = (
                    source_chain['withdraw_time_minutes'] +
                    target_chain['deposit_time_minutes']
                )

                # Get dynamic fee from API with config fallback
                config_fee = source_chain['fee_usdt']
                total_fee = self._get_dynamic_fee(source_exchange, chain_data['code'], config_fee)
                
                # Create chain option
                option = ChainOption(
                    exchange=target_exchange,
                    chain_name=chain_name,
                    chain_code=chain_data['code'],
                    fee_usdt=total_fee,
                    withdraw_time_minutes=source_chain['withdraw_time_minutes'],
                    deposit_time_minutes=target_chain['deposit_time_minutes'],
                    total_time_minutes=total_time,
                    deposit_address=target_chain.get('deposit_address'),
                    whitelisted=target_chain.get('whitelisted', False),
                    enabled=True,
                    tier='',  # Will be set later
                    score=0.0  # Will be calculated
                )
                
                chains.append(option)
        
        logger.info(f"Found {len(chains)} common chains for {source_exchange} → {target_exchange}")

        # Debug: Log which chains were included
        if chains:
            chain_codes = [c.chain_code for c in chains]
            logger.debug(f"Common chains: {', '.join(chain_codes)}")

            # Debug: Check if BNB was filtered out
            if 'bnb' in [c['code'] for c in source_chains.values() if c.get('enabled')]:
                if 'bnb' not in chain_codes:
                    logger.warning(f"⚠️  BNB is enabled in {source_exchange} but not in common chains!")
                    # Check target
                    target_has_bnb = any(c.get('code') == 'bnb' and c.get('enabled') for c in target_chains.values())
                    logger.debug(f"Target {target_exchange} has BNB enabled: {target_has_bnb}")

        return chains
    
    def calculate_score(
        self,
        chain: ChainOption,
        tier: str = 'primary',
        max_fee: float = 10.0,
        max_time: int = 30
    ) -> float:
        """
        Calculate score for a chain option
        
        Args:
            chain: Chain option
            tier: Tier type
            max_fee: Maximum fee for normalization
            max_time: Maximum time for normalization
        
        Returns:
            Score (0-100)
        """
        # Get tier weights
        weights = self.TIER_CRITERIA.get(tier, self.TIER_CRITERIA['primary'])
        
        # Fee score (lower is better)
        fee_score = max(0, 100 - (chain.fee_usdt / max_fee) * 100)
        
        # Speed score (lower is better)
        speed_score = max(0, 100 - (chain.total_time_minutes / max_time) * 100)
        
        # Reliability score
        reliability_score = 100 if chain.whitelisted else 50
        
        # Calculate weighted score
        score = (
            fee_score * weights['fee_weight'] +
            speed_score * weights['speed_weight'] +
            reliability_score * weights['reliability_weight']
        )
        
        return score
    
    def select_chains_3tier(
        self,
        source_exchange: str,
        target_exchange: str
    ) -> Dict[str, Optional[ChainOption]]:
        """
        Select chains using 3-tier strategy
        
        Args:
            source_exchange: Source CEX
            target_exchange: Target CEX
        
        Returns:
            Dict with 'primary', 'secondary', 'tertiary' chains
        """
        # Get all available chains
        chains = self.get_all_chains(source_exchange, target_exchange)
        
        if not chains:
            logger.warning(f"No common chains found for {source_exchange} → {target_exchange}")
            return {'primary': None, 'secondary': None, 'tertiary': None}
        
        # Calculate scores for each tier
        for tier in ['primary', 'secondary', 'tertiary']:
            for chain in chains:
                chain.tier = tier
                chain.score = self.calculate_score(chain, tier)
        
        # Select primary (cheapest)
        primary_chains = sorted(chains, key=lambda x: (x.fee_usdt, x.total_time_minutes))

        # Debug: Log all chains being considered
        logger.debug(f"[CHAIN SELECTION] {source_exchange} -> {target_exchange}: {len(chains)} chains")
        for chain in primary_chains[:5]:  # Show top 5
            logger.debug(f"  - {chain.chain_code}: ${chain.fee_usdt} ({chain.total_time_minutes}min)")

        primary = primary_chains[0] if primary_chains else None
        
        if primary:
            primary.tier = 'primary'
            primary.score = self.calculate_score(primary, 'primary')
        
        # Select secondary (balanced)
        secondary_chains = sorted(chains, 
                                  key=lambda x: (x.fee_usdt + x.total_time_minutes/10, x.fee_usdt))
        
        # Exclude primary
        secondary_chains = [c for c in secondary_chains if c != primary]
        secondary = secondary_chains[0] if secondary_chains else None
        
        if secondary:
            secondary.tier = 'secondary'
            secondary.score = self.calculate_score(secondary, 'secondary')
        
        # Select tertiary (most reliable/whitelisted)
        tertiary_chains = sorted(chains,
                                key=lambda x: (not x.whitelisted, x.total_time_minutes, x.fee_usdt))

        # Exclude primary and secondary
        tertiary_chains = [c for c in tertiary_chains if c not in [primary, secondary]]
        tertiary = (tertiary_chains[0] if tertiary_chains else
                   secondary or
                   primary)
        
        if tertiary:
            tertiary.tier = 'tertiary'
            tertiary.score = self.calculate_score(tertiary, 'tertiary')
        
        result = {
            'primary': primary,
            'secondary': secondary,
            'tertiary': tertiary
        }
        
        logger.info(f"Selected chains for {source_exchange} → {target_exchange}:")
        for tier, chain in result.items():
            if chain:
                logger.info(f"  {tier.upper()}: {chain.chain_code} (${chain.fee_usdt}, {chain.total_time_minutes}min)")
        
        return result
    
    def get_best_chain(
        self,
        source_exchange: str,
        target_exchange: str,
        tier: str = 'primary'
    ) -> Optional[ChainOption]:
        """
        Get best chain for specific tier
        
        Args:
            source_exchange: Source CEX
            target_exchange: Target CEX
            tier: Desired tier
        
        Returns:
            ChainOption or None
        """
        chains = self.select_chains_3tier(source_exchange, target_exchange)
        return chains.get(tier)
    
    def get_fallback_chain(
        self,
        source_exchange: str,
        target_exchange: str,
        failed_chains: List[str] = None
    ) -> Optional[ChainOption]:
        """
        Get fallback chain when primary fails
        
        Args:
            source_exchange: Source CEX
            target_exchange: Target CEX
            failed_chains: List of chain codes that already failed
        
        Returns:
            Next best ChainOption or None
        """
        failed_chains = failed_chains or []
        
        # Get all tier selections
        chains = self.select_chains_3tier(source_exchange, target_exchange)
        
        # Try in order: primary → secondary → tertiary
        for tier in ['primary', 'secondary', 'tertiary']:
            chain = chains.get(tier)
            if chain and chain.chain_code not in failed_chains:
                logger.info(f"Fallback: Using {tier} chain {chain.chain_code}")
                return chain
        
        logger.error(f"No fallback chains available for {source_exchange} → {target_exchange}")
        return None

    def refresh_fees(self, source_exchange: str, target_exchange: str):
        """
        Force refresh fees for all chains between two exchanges

        Args:
            source_exchange: Source CEX
            target_exchange: Target CEX
        """
        logger.info(f"🔄 Refreshing fees for {source_exchange} → {target_exchange}")

        # Get all common chains
        source_chains = self.config['exchanges'].get(source_exchange, {}).get('chains', {})
        target_chains = self.config['exchanges'].get(target_exchange, {}).get('chains', {})

        common_chain_codes = set()
        for chain_name, chain_data in source_chains.items():
            if chain_data.get('enabled', False):
                common_chain_codes.add(chain_data['code'])

        refreshed_count = 0
        for chain_name, chain_data in target_chains.items():
            if chain_data.get('enabled', False) and chain_data['code'] in common_chain_codes:
                chain_code = chain_data['code']
                try:
                    # Force fresh fetch (bypass cache by clearing it first)
                    cache_key = self.fee_fetcher._get_cache_key(source_exchange, chain_code, 'USDT')
                    if cache_key in self.fee_fetcher.cache:
                        del self.fee_fetcher.cache[cache_key]

                    fee, from_cache = self.fee_fetcher.fetch_fee_from_exchange(
                        exchange_id=source_exchange,
                        chain_code=chain_code,
                        coin='USDT'
                    )

                    if fee is not None:
                        logger.info(f"✅ Refreshed {chain_code}: ${fee}")
                        refreshed_count += 1
                    else:
                        logger.warning(f"⚠️  Failed to refresh {chain_code}")

                except Exception as e:
                    logger.error(f"Error refreshing fee for {chain_code}: {e}")

        logger.info(f"🔄 Fee refresh complete: {refreshed_count} chains updated")

    def clear_fee_cache(self):
        """Clear all cached fees"""
        self.fee_fetcher.clear_cache()
        logger.info("🧹 Fee cache cleared")

    def print_chain_comparison(self, source_exchange: str, target_exchange: str):
        """Print detailed comparison of all chains"""
        chains = self.get_all_chains(source_exchange, target_exchange)
        
        if not chains:
            print(f"\nNo chains available for {source_exchange} → {target_exchange}")
            return
        
        print(f"\n{'='*100}")
        print(f"CHAIN COMPARISON: {source_exchange.upper()} → {target_exchange.upper()}")
        print(f"{'='*100}")
        print(f"{'Chain':<12} {'Fee':<10} {'Time':<12} {'Whitelisted':<15} {'Address':<45}")
        print(f"{'-'*100}")
        
        for chain in sorted(chains, key=lambda x: x.fee_usdt):
            address_short = chain.deposit_address[:42] + "..." if chain.deposit_address and len(chain.deposit_address) > 45 else (chain.deposit_address or "N/A")
            whitelisted = "✓" if chain.whitelisted else "✗"
            
            print(f"{chain.chain_code:<12} ${chain.fee_usdt:<9.2f} {chain.total_time_minutes:<11}min {whitelisted:<15} {address_short:<45}")
        
        print(f"{'='*100}\n")


# Standalone test
if __name__ == "__main__":
    selector = ChainSelector("capital_config.yaml")
    
    # Test: Binance → Gate.io
    print("\n" + "="*80)
    print("TEST: Chain Selection")
    print("="*80)
    
    chains = selector.select_chains_3tier('binance', 'gateio')
    
    print("\nSelected chains:")
    for tier, chain in chains.items():
        if chain:
            print(f"\n{tier.upper()}:")
            print(f"  Chain: {chain.chain_code}")
            print(f"  Fee: ${chain.fee_usdt}")
            print(f"  Time: {chain.total_time_minutes}min")
            print(f"  Whitelisted: {chain.whitelisted}")
            print(f"  Score: {chain.score:.1f}")
    
    # Print comparison
    selector.print_chain_comparison('binance', 'gateio')
