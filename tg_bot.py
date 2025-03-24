"""
This module creates a telegram bot which parses SMS messages about transactions in ADCB bank
passed to it and automatically updates the user's budgetting google sheet used in Rich Secret app.
"""

import os
import re
from datetime import datetime

import gspread
from dotenv import load_dotenv

load_dotenv()


def parse_transaction(transaction: str) -> dict[str]:
    # Check transaction type
    transaction_type = "income" if "has been credited" in transaction else "expense"

    # Extract amount
    amount_pattern = r"([A-Z]{3})([\d,]+\.\d{2})"
    amount_match = re.search(amount_pattern, transaction)

    if amount_match:
        currency = amount_match.group(1)
        amount = float(amount_match.group(2).replace(",", ""))
    else:
        currency = None
        amount = None

    # Extract date and time
    # Try for first date format (20/03/2025 11:14:39)
    date_pattern1 = r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"
    # Try for second date format (Jan 12 2025  5:33PM)
    date_pattern2 = r"([A-Za-z]{3}\s+\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}[AP]M)"

    date_match = re.search(date_pattern1, transaction) or re.search(date_pattern2, transaction)

    if date_match:
        date_str = date_match.group(1)
        try:
            # Try first format
            date = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            # Try second format
            date = datetime.strptime(date_str, "%b %d %Y %I:%M%p")
    else:
        date = None

    # Extract location
    if transaction_type == "expense":
        # Look for location between "at" and the next comma or period
        location_pattern = r"at\s+([^,\.]+)"
        location_match = re.search(location_pattern, transaction)
        location = location_match.group(1).strip() if location_match else None
    else:
        location = None  # No location for income transactions

    return {
        "type": transaction_type,
        "amount": amount,
        "currency": currency,
        "date": date,
        "name": location,
    }


def parse_user_message(user_message: str) -> list[dict]:
    patterns = [
        "Your Cr.Card",
        "Your debit card",
        "Your salary",
    ]
    pattern = "|".join(f"({p})" for p in patterns)

    transactions = re.split(f"(?=({pattern}))", user_message)
    transactions = [t.strip() for t in transactions if t and (t not in patterns)]

    return [parse_transaction(t) for t in transactions]


def get_sheet() -> gspread.spreadsheet.Spreadsheet:
    gc = gspread.service_account()
    return gc.open_by_url(os.getenv("SHEETS_LINK"))


def update_sheet(sheet: gspread.spreadsheet.Spreadsheet, transactions: list[dict]):
    all_rows = sheet.worksheet("expenses").get_all_values()
