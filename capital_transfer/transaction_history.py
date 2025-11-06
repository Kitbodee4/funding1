"""
Transaction History Database

SQLite database for logging all capital transfers with reconciliation support.

Tables:
- transfers: Main transfer records
- transactions: Individual transaction details (test + actual)
- fees: Fee tracking
- errors: Error logging
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger


@dataclass
class TransferRecord:
    """Complete transfer record"""
    request_id: str
    source_exchange: str
    target_exchange: str
    amount_usdt: float
    chain_code: str
    test_txid: Optional[str]
    actual_txid: Optional[str]
    test_confirmed: bool
    actual_confirmed: bool
    success: bool
    fee_paid: float
    total_time_seconds: float
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class TransactionDetail:
    """Individual transaction detail"""
    txid: str
    transfer_request_id: str
    transaction_type: str  # 'test' or 'actual'
    amount: float
    status: str  # 'pending', 'confirmed', 'failed'
    created_at: datetime
    confirmed_at: Optional[datetime] = None


class TransactionHistoryDB:
    """
    SQLite database for transaction history

    Features:
    - Complete transfer logging
    - Transaction tracking
    - Fee recording
    - Error logging
    - Reconciliation queries
    - Performance analytics
    """

    def __init__(self, db_path: str = "transaction_history.db"):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
        logger.info(f"Transaction history database initialized: {self.db_path}")

    def _initialize_db(self):
        """Initialize database and create tables"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Table 1: Transfers (main records)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                source_exchange TEXT NOT NULL,
                target_exchange TEXT NOT NULL,
                amount_usdt REAL NOT NULL,
                chain_code TEXT NOT NULL,
                test_txid TEXT,
                actual_txid TEXT,
                test_confirmed BOOLEAN DEFAULT 0,
                actual_confirmed BOOLEAN DEFAULT 0,
                success BOOLEAN DEFAULT 0,
                fee_paid REAL DEFAULT 0,
                total_time_seconds REAL DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                CONSTRAINT positive_amount CHECK (amount_usdt > 0),
                CONSTRAINT positive_fee CHECK (fee_paid >= 0)
            )
        """)

        # Table 2: Transactions (individual tx details)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                txid TEXT NOT NULL,
                transfer_request_id TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL,
                confirmed_at TIMESTAMP,
                FOREIGN KEY (transfer_request_id) REFERENCES transfers(request_id),
                CONSTRAINT valid_type CHECK (transaction_type IN ('test', 'actual')),
                CONSTRAINT valid_status CHECK (status IN ('pending', 'confirmed', 'failed'))
            )
        """)

        # Table 3: Fees (detailed fee tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transfer_request_id TEXT NOT NULL,
                fee_type TEXT NOT NULL,
                amount_usdt REAL NOT NULL,
                chain_code TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL,
                FOREIGN KEY (transfer_request_id) REFERENCES transfers(request_id),
                CONSTRAINT valid_fee_type CHECK (fee_type IN ('withdrawal', 'network', 'total'))
            )
        """)

        # Table 4: Errors (error logging)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transfer_request_id TEXT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                occurred_at TIMESTAMP NOT NULL,
                FOREIGN KEY (transfer_request_id) REFERENCES transfers(request_id)
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_request_id ON transfers(request_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_source ON transfers(source_exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_target ON transfers(target_exchange)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_created ON transfers(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_txid ON transactions(txid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_request ON transactions(transfer_request_id)")

        self.conn.commit()
        logger.debug("Database tables created successfully")

    def record_transfer(self, record: TransferRecord) -> bool:
        """
        Record a transfer

        Args:
            record: Transfer record

        Returns:
            Success status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO transfers (
                    request_id, source_exchange, target_exchange, amount_usdt,
                    chain_code, test_txid, actual_txid, test_confirmed, actual_confirmed,
                    success, fee_paid, total_time_seconds, error_message,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.request_id,
                record.source_exchange,
                record.target_exchange,
                record.amount_usdt,
                record.chain_code,
                record.test_txid,
                record.actual_txid,
                record.test_confirmed,
                record.actual_confirmed,
                record.success,
                record.fee_paid,
                record.total_time_seconds,
                record.error_message,
                record.created_at.isoformat(),
                record.completed_at.isoformat() if record.completed_at else None
            ))

            self.conn.commit()
            logger.info(f"✓ Transfer recorded: {record.request_id}")
            return True

        except sqlite3.IntegrityError as e:
            logger.error(f"✗ Duplicate transfer record: {record.request_id}")
            return False
        except Exception as e:
            logger.error(f"✗ Error recording transfer: {e}")
            self.conn.rollback()
            return False

    def record_transaction(self, detail: TransactionDetail) -> bool:
        """
        Record a transaction detail

        Args:
            detail: Transaction detail

        Returns:
            Success status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (
                    txid, transfer_request_id, transaction_type, amount,
                    status, created_at, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                detail.txid,
                detail.transfer_request_id,
                detail.transaction_type,
                detail.amount,
                detail.status,
                detail.created_at.isoformat(),
                detail.confirmed_at.isoformat() if detail.confirmed_at else None
            ))

            self.conn.commit()
            logger.debug(f"Transaction recorded: {detail.txid}")
            return True

        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
            self.conn.rollback()
            return False

    def update_transaction_status(
        self,
        txid: str,
        status: str,
        confirmed_at: Optional[datetime] = None
    ) -> bool:
        """
        Update transaction status

        Args:
            txid: Transaction ID
            status: New status
            confirmed_at: Confirmation timestamp

        Returns:
            Success status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE transactions
                SET status = ?, confirmed_at = ?
                WHERE txid = ?
            """, (status, confirmed_at.isoformat() if confirmed_at else None, txid))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating transaction: {e}")
            return False

    def record_fee(
        self,
        request_id: str,
        fee_type: str,
        amount: float,
        chain_code: str
    ) -> bool:
        """
        Record a fee

        Args:
            request_id: Transfer request ID
            fee_type: Type of fee
            amount: Fee amount
            chain_code: Chain code

        Returns:
            Success status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fees (
                    transfer_request_id, fee_type, amount_usdt, chain_code, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (request_id, fee_type, amount, chain_code, datetime.now(timezone.utc).isoformat()))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error recording fee: {e}")
            return False

    def record_error(
        self,
        request_id: Optional[str],
        error_type: str,
        error_message: str
    ) -> bool:
        """
        Record an error

        Args:
            request_id: Transfer request ID (if applicable)
            error_type: Type of error
            error_message: Error message

        Returns:
            Success status
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO errors (
                    transfer_request_id, error_type, error_message, occurred_at
                ) VALUES (?, ?, ?, ?)
            """, (request_id, error_type, error_message, datetime.now(timezone.utc).isoformat()))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error recording error: {e}")
            return False

    def get_transfer(self, request_id: str) -> Optional[Dict]:
        """
        Get transfer by request ID

        Args:
            request_id: Transfer request ID

        Returns:
            Transfer record dict or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transfers WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_transfers_by_exchange(
        self,
        exchange: str,
        direction: str = 'both'
    ) -> List[Dict]:
        """
        Get transfers by exchange

        Args:
            exchange: Exchange name
            direction: 'source', 'target', or 'both'

        Returns:
            List of transfer records
        """
        cursor = self.conn.cursor()

        if direction == 'source':
            cursor.execute("SELECT * FROM transfers WHERE source_exchange = ? ORDER BY created_at DESC", (exchange,))
        elif direction == 'target':
            cursor.execute("SELECT * FROM transfers WHERE target_exchange = ? ORDER BY created_at DESC", (exchange,))
        else:
            cursor.execute("""
                SELECT * FROM transfers
                WHERE source_exchange = ? OR target_exchange = ?
                ORDER BY created_at DESC
            """, (exchange, exchange))

        return [dict(row) for row in cursor.fetchall()]

    def get_transfers_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get transfers by date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of transfer records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM transfers
            WHERE created_at BETWEEN ? AND ?
            ORDER BY created_at DESC
        """, (start_date.isoformat(), end_date.isoformat()))

        return [dict(row) for row in cursor.fetchall()]

    def get_successful_transfers(self) -> List[Dict]:
        """Get all successful transfers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transfers WHERE success = 1 ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_failed_transfers(self) -> List[Dict]:
        """Get all failed transfers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transfers WHERE success = 0 ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get transfer statistics

        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()

        # Total transfers
        cursor.execute("SELECT COUNT(*) FROM transfers")
        total_transfers = cursor.fetchone()[0]

        # Successful transfers
        cursor.execute("SELECT COUNT(*) FROM transfers WHERE success = 1")
        successful_transfers = cursor.fetchone()[0]

        # Failed transfers
        cursor.execute("SELECT COUNT(*) FROM transfers WHERE success = 0")
        failed_transfers = cursor.fetchone()[0]

        # Total volume
        cursor.execute("SELECT SUM(amount_usdt) FROM transfers WHERE success = 1")
        total_volume = cursor.fetchone()[0] or 0

        # Total fees
        cursor.execute("SELECT SUM(fee_paid) FROM transfers WHERE success = 1")
        total_fees = cursor.fetchone()[0] or 0

        # Average transfer time
        cursor.execute("SELECT AVG(total_time_seconds) FROM transfers WHERE success = 1")
        avg_time = cursor.fetchone()[0] or 0

        # Success rate
        success_rate = (successful_transfers / total_transfers * 100) if total_transfers > 0 else 0

        return {
            'total_transfers': total_transfers,
            'successful_transfers': successful_transfers,
            'failed_transfers': failed_transfers,
            'success_rate': success_rate,
            'total_volume_usdt': total_volume,
            'total_fees_usdt': total_fees,
            'avg_transfer_time_seconds': avg_time,
            'avg_fee_percent': (total_fees / total_volume * 100) if total_volume > 0 else 0
        }

    def reconcile(self, exchange: str) -> Dict[str, Any]:
        """
        Reconcile transfers for an exchange

        Args:
            exchange: Exchange name

        Returns:
            Reconciliation report
        """
        cursor = self.conn.cursor()

        # Sent from exchange
        cursor.execute("""
            SELECT SUM(amount_usdt) FROM transfers
            WHERE source_exchange = ? AND success = 1
        """, (exchange,))
        total_sent = cursor.fetchone()[0] or 0

        # Received by exchange
        cursor.execute("""
            SELECT SUM(amount_usdt) FROM transfers
            WHERE target_exchange = ? AND success = 1
        """, (exchange,))
        total_received = cursor.fetchone()[0] or 0

        # Net flow
        net_flow = total_received - total_sent

        return {
            'exchange': exchange,
            'total_sent': total_sent,
            'total_received': total_received,
            'net_flow': net_flow
        }

    def print_statistics(self):
        """Print statistics"""
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("TRANSFER STATISTICS")
        print("="*80)
        print(f"Total Transfers:      {stats['total_transfers']}")
        print(f"Successful:           {stats['successful_transfers']}")
        print(f"Failed:               {stats['failed_transfers']}")
        print(f"Success Rate:         {stats['success_rate']:.1f}%")
        print(f"Total Volume:         ${stats['total_volume_usdt']:,.2f}")
        print(f"Total Fees:           ${stats['total_fees_usdt']:,.2f}")
        print(f"Avg Fee:              {stats['avg_fee_percent']:.2f}%")
        print(f"Avg Transfer Time:    {stats['avg_transfer_time_seconds']:.1f}s")
        print("="*80 + "\n")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Test function
def test_transaction_history():
    """Test transaction history database"""
    print("\n" + "="*80)
    print("TEST: Transaction History Database")
    print("="*80)

    # Create database
    db = TransactionHistoryDB("test_transactions.db")

    # Test record
    record = TransferRecord(
        request_id="TEST001",
        source_exchange="binance",
        target_exchange="gateio",
        amount_usdt=100.0,
        chain_code="arb",
        test_txid="test_tx_123",
        actual_txid="actual_tx_456",
        test_confirmed=True,
        actual_confirmed=True,
        success=True,
        fee_paid=1.0,
        total_time_seconds=120.5,
        error_message=None,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc)
    )

    # Record transfer
    success = db.record_transfer(record)
    print(f"\nTransfer recorded: {success}")

    # Get statistics
    db.print_statistics()

    # Get transfer
    retrieved = db.get_transfer("TEST001")
    if retrieved:
        print(f"Retrieved transfer: {retrieved['request_id']}")
        print(f"  Amount: ${retrieved['amount_usdt']}")
        print(f"  Success: {bool(retrieved['success'])}")

    db.close()


if __name__ == "__main__":
    test_transaction_history()
