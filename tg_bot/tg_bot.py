"""
This module creates a telegram bot which parses SMS messages about transactions in ADCB bank
passed to it and automatically updates the user's budgetting google sheet used in Rich Secret app.
"""

import json
import os
import re
from datetime import datetime
from functools import partial

import gspread
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler

load_dotenv()


def parse_transaction(transaction: str) -> dict[str]:
    transaction_type = "income" if "has been credited" in transaction else "expense"

    amount_pattern = r"([A-Z]{3})([\d,]+\.\d{2})"
    amount_match = re.search(amount_pattern, transaction)

    if amount_match:
        currency = amount_match.group(1)
        amount = float(amount_match.group(2).replace(",", ""))
    else:
        currency = None
        amount = None

    date_pattern1 = r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"
    date_pattern2 = r"([A-Za-z]{3}\s+\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}[AP]M)"

    date_match = re.search(date_pattern1, transaction) or re.search(date_pattern2, transaction)

    if date_match:
        date_str = date_match.group(1)
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            date = datetime.strptime(date_str, "%b %d %Y %I:%M%p")
    else:
        date = None

    if transaction_type == "expense":
        location_pattern = r"at\s+([^,\.]+)"
        location_match = re.search(location_pattern, transaction)
        location = location_match.group(1).strip() if location_match else None
    else:
        location = None

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


def get_sheet(url: str) -> gspread.spreadsheet.Spreadsheet:
    gc = gspread.service_account()
    return gc.open_by_url(url)


def convert_transaction_to_row(transaction: dict[str], keys_order: list) -> list[str]:
    return [
        transaction.get(field, "").strftime("%d %b, %Y")
        if field == "date" and transaction.get(field, "") != ""
        else str(transaction.get(field, ""))
        for field in keys_order
    ]


def update_sheet(sheet: gspread.spreadsheet.Spreadsheet, transactions: list[dict]) -> int:
    expenses_rows = sheet.worksheet("expenses").get_values()
    income_rows = sheet.worksheet("income").get_values()

    new_expenses_rows = []
    new_income_rows = []

    for transaction in transactions:
        if transaction["type"] == "expense":
            row = convert_transaction_to_row(transaction, expenses_rows[0])
            if row not in expenses_rows:
                new_expenses_rows.append(row)
        else:
            row = convert_transaction_to_row(transaction, income_rows[0])
            if row not in income_rows:
                new_income_rows.append(row)

    sheet.worksheet("expenses").append_rows(new_expenses_rows)
    sheet.worksheet("income").append_rows(new_income_rows)

    return len(new_income_rows) + len(new_expenses_rows)


class AuthorizationError(Exception):
    pass


def check_authorization(update):
    if authorized_chats := os.getenv("AUTHORIZED_CHATS"):
        authorized_chats = json.loads(authorized_chats)
        if update.message.chat_id not in [int(chat_id) for chat_id in authorized_chats]:
            raise AuthorizationError("You are not authorized to use this bot")


async def bot_respond(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG001
    sheet: gspread.spreadsheet.Spreadsheet,
) -> None:
    try:
        check_authorization(update)
        transactions = parse_user_message(update.message.text)
        if transactions[0]["amount"]:
            n_transactions = update_sheet(sheet, transactions)
            most_recent_date = max(transactions, key=lambda x: x["date"])["date"]
            await update.message.reply_text(
                text=(
                    rf"Added *{n_transactions}* transactions to [sheet]({os.getenv('SHEETS_LINK')})\."
                    f"\nMost recent transaction was made\non *{most_recent_date.strftime('%d %b, %Y')}*"
                ),
                parse_mode="MarkdownV2",
            )
        else:
            await update.message.reply_text(
                text="Couldn't parse any transactions from your message",
            )

    except Exception as e:
        await update.message.reply_text(
            text=f"Encountered an error:\n*{e!s}*",
            parse_mode="MarkdownV2",
        )


def main():
    sheet = get_sheet(os.getenv("SHEETS_LINK"))

    application = ApplicationBuilder().token(os.getenv("API_TOKEN")).build()

    handler = MessageHandler(None, partial(bot_respond, sheet=sheet))
    application.add_handler(handler)

    application.run_polling()


if __name__ == "__main__":
    main()
