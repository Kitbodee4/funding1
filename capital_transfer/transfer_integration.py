"""
Capital Transfer Integration

Integrates capital transfer system with funding scanner for automated transfers
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger
import uuid

# Import from funding scanner (Part 1)
try:
    from ..funding_scanner import EnhancedFundingScanner, FundingOpportunity
    FUNDING_SCANNER_AVAILABLE = True
except ImportError:
    logger.warning("Funding scanner not available")
    FUNDING_SCANNER_AVAILABLE = False
    EnhancedFundingScanner = None
    FundingOpportunity = None

# Import from capital transfer
from .transfer_engine import CapitalTransferEngine, TransferRequest, TransferResult
from .transaction_history import TransactionHistoryDB


class CapitalTransferIntegration:
    """
    Integration between capital transfer and funding scanner
    
    Features:
    - Auto-transfer when opportunity detected
    - Calculate optimal transfer amount
    - Coordinate transfers for both sides (long & short)
    - Return funds to Binance after closing
    - Complete lifecycle management
    """
    
    def __init__(
        self,
        transfer_engine: CapitalTransferEngine,
        history_db: TransactionHistoryDB,
        main_hub_exchange: str = "binance"
    ):
        """
        Initialize integration

        Args:
            transfer_engine: Transfer engine instance
            history_db: Transaction history database instance
            main_hub_exchange: Main hub exchange (default: binance)
        """
        self.transfer_engine = transfer_engine
        self.history_db = history_db
        self.main_hub = main_hub_exchange
        
        logger.info(f"Capital Transfer Integration initialized (hub: {main_hub_exchange})")
    
    def calculate_transfer_amounts(
        self,
        opportunity: FundingOpportunity,
        total_capital: float,
        reserve_ratio: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate how much to transfer to each exchange

        Args:
            opportunity: Funding opportunity
            total_capital: Capital PER SIDE (not total for both sides)
            reserve_ratio: Reserve ratio (10% default)

        Returns:
            Dict with transfer amounts per exchange
        """
        # NOTE: total_capital is the amount per side, not the total for both sides
        # So for capital = $100, we need $100 for long and $100 for short = $200 total

        # Calculate total capital needed for both sides
        total_needed = total_capital * 2

        # Reserve some capital
        reserved_amount = total_needed * reserve_ratio
        usable_capital = total_needed - reserved_amount

        # Split equally for long and short sides
        per_side = usable_capital / 2

        amounts = {
            opportunity.exchange_long: per_side,
            opportunity.exchange_short: per_side
        }

        logger.info(f"Calculated transfer amounts:")
        logger.info(f"  {opportunity.exchange_long}: ${per_side:,.2f}")
        logger.info(f"  {opportunity.exchange_short}: ${per_side:,.2f}")
        logger.info(f"  Reserved: ${reserved_amount:,.2f}")
        logger.info(f"  Total (including reserve): ${total_needed:,.2f}")

        return amounts
    
    async def transfer_capital_for_opportunity(
        self,
        opportunity: FundingOpportunity,
        capital: float
    ) -> Dict[str, TransferResult]:
        """
        Transfer capital to exchanges for opening positions
        
        Process:
        1. Calculate amounts for each side
        2. Transfer from Binance to long exchange
        3. Transfer from Binance to short exchange
        4. Return results
        
        Args:
            opportunity: Funding opportunity
            capital: Total capital to allocate
        
        Returns:
            Dict with transfer results
        """
        logger.info(f"Starting capital transfer for opportunity: {opportunity.symbol}")
        
        # Calculate amounts
        amounts = self.calculate_transfer_amounts(opportunity, capital)
        
        results = {}
        
        # Transfer to long exchange (if not already there)
        if opportunity.exchange_long != self.main_hub:
            logger.info(f"Transferring ${amounts[opportunity.exchange_long]:,.2f} to {opportunity.exchange_long} (long side)")
            
            request = TransferRequest(
                request_id=f"LONG_{uuid.uuid4().hex[:8]}",
                source_exchange=self.main_hub,
                target_exchange=opportunity.exchange_long,
                amount_usdt=amounts[opportunity.exchange_long],
                purpose=f"funding_arb_{opportunity.symbol}_long"
            )
            
            result = await self.transfer_engine.transfer_with_fallback(request)
            results['long'] = result
            
            # Record to history
            if result.success:
                logger.info(f"✓ Long transfer completed: {result.actual_txid}")
            else:
                logger.error(f"✗ Long transfer failed: {result.error_message}")
        else:
            logger.info(f"Long exchange is main hub, no transfer needed")
            results['long'] = None
        
        # Transfer to short exchange (if not already there)
        if opportunity.exchange_short != self.main_hub:
            logger.info(f"Transferring ${amounts[opportunity.exchange_short]:,.2f} to {opportunity.exchange_short} (short side)")
            
            request = TransferRequest(
                request_id=f"SHORT_{uuid.uuid4().hex[:8]}",
                source_exchange=self.main_hub,
                target_exchange=opportunity.exchange_short,
                amount_usdt=amounts[opportunity.exchange_short],
                purpose=f"funding_arb_{opportunity.symbol}_short"
            )
            
            result = await self.transfer_engine.transfer_with_fallback(request)
            results['short'] = result
            
            # Record to history
            if result.success:
                logger.info(f"✓ Short transfer completed: {result.actual_txid}")
            else:
                logger.error(f"✗ Short transfer failed: {result.error_message}")
        else:
            logger.info(f"Short exchange is main hub, no transfer needed")
            results['short'] = None
        
        # Check if both transfers successful
        all_success = all(r is None or r.success for r in results.values())
        
        if all_success:
            logger.info(f"✅ All capital transfers completed successfully")
        else:
            logger.error(f"❌ Some capital transfers failed")
        
        return results
    
    async def return_capital_to_hub(
        self,
        from_exchanges: List[str],
        amounts: Dict[str, float]
    ) -> Dict[str, TransferResult]:
        """
        Return capital from exchanges back to main hub
        
        Args:
            from_exchanges: List of exchanges to return from
            amounts: Amount to return from each exchange
        
        Returns:
            Dict with transfer results
        """
        logger.info(f"Returning capital to {self.main_hub}")
        
        results = {}
        
        for exchange in from_exchanges:
            if exchange == self.main_hub:
                continue
            
            amount = amounts.get(exchange, 0)
            if amount <= 0:
                continue
            
            logger.info(f"Returning ${amount:,.2f} from {exchange}")
            
            request = TransferRequest(
                request_id=f"RETURN_{uuid.uuid4().hex[:8]}",
                source_exchange=exchange,
                target_exchange=self.main_hub,
                amount_usdt=amount,
                purpose="return_to_hub"
            )
            
            result = await self.transfer_engine.transfer_with_fallback(request)
            results[exchange] = result
            
            if result.success:
                logger.info(f"✓ Return completed from {exchange}: {result.actual_txid}")
            else:
                logger.error(f"✗ Return failed from {exchange}: {result.error_message}")
        
        return results
    
    async def complete_funding_arbitrage_lifecycle(
        self,
        opportunity: FundingOpportunity,
        capital: float,
        hold_duration_hours: int = 24
    ) -> Dict:
        """
        Complete funding arbitrage lifecycle:
        1. Transfer capital to exchanges
        2. Open positions (simulated)
        3. Wait for holding period
        4. Close positions (simulated)
        5. Return capital to hub
        
        Args:
            opportunity: Funding opportunity
            capital: Total capital
            hold_duration_hours: How long to hold
        
        Returns:
            Complete lifecycle report
        """
        logger.info(f"="*80)
        logger.info(f"Starting complete funding arbitrage lifecycle")
        logger.info(f"  Symbol: {opportunity.symbol}")
        logger.info(f"  Long: {opportunity.exchange_long}")
        logger.info(f"  Short: {opportunity.exchange_short}")
        logger.info(f"  Capital: ${capital:,.2f}")
        logger.info(f"="*80)
        
        lifecycle_report = {
            'opportunity': opportunity.to_dict(),
            'capital': capital,
            'transfers': {},
            'positions': {},
            'returns': {},
            'total_fees_paid': 0.0,
            'net_profit': 0.0
        }
        
        # Phase 1: Transfer capital
        logger.info("\n📤 PHASE 1: Transferring capital to exchanges")
        transfer_results = await self.transfer_capital_for_opportunity(opportunity, capital)
        lifecycle_report['transfers'] = {
            k: v.to_dict() if v else None for k, v in transfer_results.items()
        }
        
        # Calculate total transfer fees
        total_transfer_fees = sum(
            r.fee_paid for r in transfer_results.values() if r and r.success
        )
        lifecycle_report['total_fees_paid'] += total_transfer_fees
        
        logger.info(f"✓ Transfer fees: ${total_transfer_fees:.2f}")
        
        # Phase 2: Open positions (simulated)
        logger.info("\n📊 PHASE 2: Opening positions")
        logger.info("  [Simulated - would open actual positions here]")
        
        amounts = self.calculate_transfer_amounts(opportunity, capital, reserve_ratio=0.1)
        
        lifecycle_report['positions'] = {
            'long': {
                'exchange': opportunity.exchange_long,
                'amount': amounts[opportunity.exchange_long],
                'funding_rate': opportunity.long_rate
            },
            'short': {
                'exchange': opportunity.exchange_short,
                'amount': amounts[opportunity.exchange_short],
                'funding_rate': opportunity.short_rate
            }
        }
        
        # Phase 3: Hold period (simulated)
        logger.info(f"\n⏳ PHASE 3: Holding for {hold_duration_hours} hours")
        logger.info("  [Simulated - would monitor positions here]")
        
        # Calculate expected funding earnings
        periods = hold_duration_hours / opportunity.long_interval_hours
        expected_earnings = (
            abs(opportunity.short_rate) * amounts[opportunity.exchange_short] * periods +
            abs(opportunity.long_rate) * amounts[opportunity.exchange_long] * periods
        ) * 100  # Convert to USD
        
        lifecycle_report['expected_earnings'] = expected_earnings
        
        logger.info(f"  Expected funding earnings: ${expected_earnings:.2f}")
        
        # Phase 4: Close positions (simulated)
        logger.info("\n📉 PHASE 4: Closing positions")
        logger.info("  [Simulated - would close actual positions here]")
        
        # Phase 5: Return capital
        logger.info("\n📥 PHASE 5: Returning capital to hub")
        return_results = await self.return_capital_to_hub(
            [opportunity.exchange_long, opportunity.exchange_short],
            amounts
        )
        lifecycle_report['returns'] = {
            k: v.to_dict() for k, v in return_results.items()
        }
        
        # Calculate total return fees
        total_return_fees = sum(
            r.fee_paid for r in return_results.values() if r.success
        )
        lifecycle_report['total_fees_paid'] += total_return_fees
        
        logger.info(f"✓ Return fees: ${total_return_fees:.2f}")
        
        # Calculate net profit
        net_profit = expected_earnings - lifecycle_report['total_fees_paid']
        lifecycle_report['net_profit'] = net_profit
        lifecycle_report['roi_pct'] = (net_profit / capital) * 100 if capital > 0 else 0
        
        # Final report
        logger.info("\n" + "="*80)
        logger.info("LIFECYCLE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total Fees Paid: ${lifecycle_report['total_fees_paid']:.2f}")
        logger.info(f"Expected Earnings: ${expected_earnings:.2f}")
        logger.info(f"Net Profit: ${net_profit:.2f}")
        logger.info(f"ROI: {lifecycle_report['roi_pct']:.2f}%")
        logger.info("="*80)
        
        return lifecycle_report


# Test function
async def test_integration():
    """Test capital transfer integration"""
    from enhanced_funding_scanner import FundingOpportunity
    from transfer_engine import CapitalTransferEngine
    from transaction_history import TransactionHistoryManager
    
    print("\n" + "="*80)
    print("TEST: Capital Transfer Integration")
    print("="*80)
    
    # Create mock opportunity
    opportunity = FundingOpportunity(
        symbol='BTC/USDT:USDT',
        exchange_long='gateio',
        exchange_short='okx',
        long_rate=0.0001,
        short_rate=-0.0005,
        rate_diff=0.0006,
        rate_diff_pct=0.06,
        long_volume_24h=100_000_000,
        short_volume_24h=80_000_000,
        long_oi=500_000_000,
        short_oi=450_000_000,
        long_oi_rank=1,
        short_oi_rank=2
    )
    
    # Initialize components
    engine = CapitalTransferEngine(
        enable_test_transfer=False,  # Disable for testing
        enable_health_check=False
    )
    
    history = TransactionHistoryManager("sqlite:///test_integration.db")
    
    integration = CapitalTransferIntegration(engine, history)
    
    # Calculate transfer amounts
    print("\nCalculating transfer amounts for $10,000 capital...")
    amounts = integration.calculate_transfer_amounts(opportunity, 10000)
    
    for exchange, amount in amounts.items():
        print(f"  {exchange}: ${amount:,.2f}")
    
    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_integration())
