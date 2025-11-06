"""
Capital Config Parser

Parses Excel file containing transfer configuration and generates capital_config.yaml
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

try:
    from .fee_fetcher import FeeFetcher
    FEE_FETCHER_AVAILABLE = True
except ImportError:
    FEE_FETCHER_AVAILABLE = False
    logger.warning("FeeFetcher not available - fees will not be auto-fetched")


class CapitalConfigParser:
    """
    Parse Excel file and generate capital_config.yaml
    
    Features:
    - Extracts chain configurations from Excel
    - Generates structured YAML config
    - Supports multiple CEXs
    - Handles whitelisted addresses
    """
    
    # Chain name mapping
    CHAIN_MAPPING = {
        'bnb': 'BSC',
        'trc': 'TRC20',
        'sol': 'SOL',
        'op': 'OPTIMISM',
        'polygon': 'POLYGON',
        'avax c': 'AVALANCHE_C',
        'apot': 'APTOS',
        'arb': 'ARBITRUM',
        'erc': 'ERC20'
    }
    
    def __init__(self, excel_path: str, fetch_fees: bool = False, api_keys: Optional[Dict] = None):
        """
        Initialize parser

        Args:
            excel_path: Path to Excel file
            fetch_fees: If True, fetch real-time fees from exchanges
            api_keys: Optional API keys for exchanges
        """
        self.excel_path = Path(excel_path)
        self.fetch_fees = fetch_fees
        self.api_keys = api_keys or {}

        # Initialize fee fetcher if enabled
        self.fee_fetcher = None
        if fetch_fees and FEE_FETCHER_AVAILABLE:
            self.fee_fetcher = FeeFetcher(api_keys=api_keys)
            logger.info("💰 Dynamic fee fetching enabled")
        elif fetch_fees and not FEE_FETCHER_AVAILABLE:
            logger.warning("⚠️  Fee fetching requested but FeeFetcher not available")

        self.config: Dict = {
            'exchanges': {},
            'main_hub': 'binance',  # Binance as main hub
            'metadata': {
                'generated_at': None,
                'source': str(excel_path),
                'fees_fetched_from_api': fetch_fees
            }
        }
    
    def parse_exchange_data(
        self,
        sheet_name: str,
        exchange_name: str,
        start_row: int = 0
    ) -> Dict:
        """
        Parse data for a single exchange
        
        Args:
            sheet_name: Excel sheet name
            exchange_name: CEX name
            start_row: Starting row in Excel
        
        Returns:
            Exchange configuration dict
        """
        try:
            # Read Excel
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name, header=None)
            
            # Find data start
            chains = {}
            
            # Parse each row
            for i in range(start_row, len(df)):
                # Check if it's a valid chain row
                chain_code = df.iloc[i, 0] if pd.notna(df.iloc[i, 0]) else None
                
                if not chain_code or chain_code in ['NaN', '', exchange_name.lower()]:
                    continue
                
                # Stop if we hit another exchange or empty section
                if pd.isna(df.iloc[i, 1]):
                    break
                
                # Extract data
                fee = df.iloc[i, 1]
                withdraw_time = df.iloc[i, 2]
                address = df.iloc[i, 3] if len(df.columns) > 3 else None
                deposit_time = df.iloc[i, 5] if len(df.columns) > 5 else None
                
                # Skip if fee is '-' (chain not available)
                if fee == '-' or pd.isna(fee):
                    continue
                
                # Normalize chain name
                chain_name = self.CHAIN_MAPPING.get(str(chain_code).lower(), str(chain_code).upper())
                
                # Create chain config
                chains[chain_name] = {
                    'code': str(chain_code).lower(),
                    'fee_usdt': float(fee) if pd.notna(fee) else 999,
                    'withdraw_time_minutes': int(withdraw_time) if pd.notna(withdraw_time) else 999,
                    'deposit_time_minutes': int(deposit_time) if pd.notna(deposit_time) else 999,
                    'deposit_address': str(address) if pd.notna(address) and address != '-' else None,
                    'enabled': True,
                    'whitelisted': True if address and address != '-' else False
                }
            
            return {
                'name': exchange_name,
                'chains': chains
            }
            
        except Exception as e:
            logger.error(f"Error parsing {exchange_name}: {e}")
            return {'name': exchange_name, 'chains': {}}
    
    def parse_all_exchanges(self) -> Dict:
        """
        Parse all exchanges from Excel
        
        Returns:
            Complete config dict
        """
        from datetime import datetime
        
        logger.info(f"Parsing config from {self.excel_path}")
        
        # Parse Gate.io (ชีต5)
        gateio_config = self.parse_exchange_data('ชีต5', 'gateio', start_row=1)
        self.config['exchanges']['gateio'] = gateio_config
        
        # Parse OKX (ชีต4, starts at row 14)
        okx_config = self.parse_exchange_data('ชีต4', 'okx', start_row=15)
        self.config['exchanges']['okx'] = okx_config
        
        # Parse other exchanges from ชีต4 (Binance, Bybit, etc.)
        # You can add more parsers here based on your Excel structure
        
        # For now, add placeholder configs for other exchanges
        self._add_default_exchanges()
        
        # Add metadata
        self.config['metadata']['generated_at'] = datetime.now().isoformat()
        
        logger.info(f"Parsed {len(self.config['exchanges'])} exchanges")
        
        return self.config
    
    def _add_default_exchanges(self):
        """Add default configs for exchanges not in Excel"""
        
        # Binance (main hub) - add common chains
        if 'binance' not in self.config['exchanges']:
            self.config['exchanges']['binance'] = {
                'name': 'binance',
                'is_main_hub': True,
                'chains': {
                    'BSC': {
                        'code': 'bnb',
                        'fee_usdt': 0.5,
                        'withdraw_time_minutes': 1,
                        'deposit_time_minutes': 1,
                        'enabled': True,
                        'whitelisted': False
                    },
                    'TRC20': {
                        'code': 'trc',
                        'fee_usdt': 1.0,
                        'withdraw_time_minutes': 1,
                        'deposit_time_minutes': 2,
                        'enabled': True,
                        'whitelisted': False
                    }
                }
            }
        
        # Bybit
        if 'bybit' not in self.config['exchanges']:
            self.config['exchanges']['bybit'] = {
                'name': 'bybit',
                'chains': {}
            }
        
        # Bitget
        if 'bitget' not in self.config['exchanges']:
            self.config['exchanges']['bitget'] = {
                'name': 'bitget',
                'chains': {}
            }
        
        # KuCoin
        if 'kucoin' not in self.config['exchanges']:
            self.config['exchanges']['kucoin'] = {
                'name': 'kucoin',
                'chains': {}
            }
    
    def generate_yaml(self, output_path: str = "capital_config.yaml", update_fees: bool = None):
        """
        Generate capital_config.yaml file

        Args:
            output_path: Output file path
            update_fees: If True, fetch real fees. If None, uses self.fetch_fees
        """
        # Parse all data
        config = self.parse_all_exchanges()

        # Update fees if requested
        should_update_fees = update_fees if update_fees is not None else self.fetch_fees
        if should_update_fees and self.fee_fetcher:
            logger.info("🔄 Fetching real-time fees from exchanges...")
            config = self.fee_fetcher.update_config_fees(config)
        elif should_update_fees:
            logger.warning("⚠️  Fee update requested but fee fetcher not available")
        
        # Add chain selection tiers
        config['chain_selection'] = {
            'strategy': '3_tier',
            'tiers': {
                'primary': {
                    'description': 'Cheapest option (main)',
                    'criteria': 'lowest_fee',
                    'weight': 1.0
                },
                'secondary': {
                    'description': 'Fast and cheap (backup)',
                    'criteria': 'balanced',
                    'weight': 0.7
                },
                'tertiary': {
                    'description': 'Reliable (emergency)',
                    'criteria': 'reliable',
                    'weight': 0.5
                }
            }
        }
        
        # Add safety settings
        config['safety'] = {
            'test_transfer': {
                'enabled': True,
                'amount_usdt': 10.1,  # Minimum test amount
                'confirmation_required': True
            },
            'chain_health_check': {
                'enabled': True,
                'check_before_transfer': True,
                'max_retry': 3
            },
            'address_validation': {
                'enabled': True,
                'checksum_validation': True
            },
            'fallback': {
                'enabled': True,
                'auto_switch_chain': True,
                'max_fallback_attempts': 2
            }
        }
        
        # Add transfer limits
        config['limits'] = {
            'min_transfer_usdt': 10,
            'max_transfer_usdt': 100000,
            'daily_transfer_limit': 500000
        }
        
        # Save to YAML
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"✓ Generated {output_path}")
        
        return output_path
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("CAPITAL CONFIG SUMMARY")
        print("="*80)
        
        for exchange_name, exchange_data in self.config['exchanges'].items():
            chains = exchange_data.get('chains', {})
            print(f"\n{exchange_name.upper()}:")
            print(f"  Total chains: {len(chains)}")
            
            if chains:
                # Find cheapest chain
                cheapest = min(chains.values(), key=lambda x: x['fee_usdt'])
                print(f"  Cheapest: {cheapest['code']} (${cheapest['fee_usdt']})")
                
                # Find fastest chain
                fastest = min(chains.values(), 
                            key=lambda x: x['withdraw_time_minutes'] + x['deposit_time_minutes'])
                total_time = fastest['withdraw_time_minutes'] + fastest['deposit_time_minutes']
                print(f"  Fastest: {fastest['code']} ({total_time}min)")
                
                # Whitelisted count
                whitelisted = sum(1 for c in chains.values() if c.get('whitelisted', False))
                print(f"  Whitelisted: {whitelisted}/{len(chains)}")
        
        print("\n" + "="*80)


# Standalone test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = "สเปรดช_ตไม_ม_ช__อ__1_.xlsx"
    
    parser = CapitalConfigParser(excel_file)
    parser.generate_yaml("capital_config.yaml")
    parser.print_summary()
