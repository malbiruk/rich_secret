"""
This module creates a telegram bot which parses SMS messages about transactions in ADCB bank
passed to it and automatically updates the user's budgetting google sheet used in Rich Secret app.
"""

import json
import os
import re
from datetime import datetime
from functools import partial
from pathlib import Path

import gspread
import pytz
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

with Path("categories.json").open() as f:
    CATEGORIES = json.load(f)


def guess_category(name: str) -> str:
    name_lower = name.lower()

    for category, keywords in CATEGORIES.items():
        if any(keyword in name_lower for keyword in keywords):
            return category

    return "Other"


def parse_transaction(transaction: str) -> dict[str]:
    income_patterns = [
        "has been credited",
        "on your account",
        "has been deposited",
    ]

    transaction_type = (
        "income" if any(pattern in transaction for pattern in income_patterns) else "expense"
    )

    amount_pattern = r"((?!XXX)[A-Z]{3})\s?([\d,]+(?:\.\d{2})?)"
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
        date = datetime.now(pytz.timezone("Asia/Dubai"))

    if transaction_type == "expense":
        location_pattern = r"at\s+([^,\.]+)"
        location_match = re.search(location_pattern, transaction)
        location = location_match.group(1).strip() if location_match else None
    else:
        location = ""

    if transaction_type == "expense" and not location:
        return None

    return {
        "type": transaction_type,
        "amount": amount,
        "currency": currency,
        "date": date,
        "name": location,
        "category": guess_category(location),
    }


def parse_user_message(user_message: str) -> list[dict]:
    patterns = [
        "Your Cr.Card",
        "Your debit card",
        "Your salary",
        "A Cr. transaction",
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
        else transaction.get(field, "")
        for field in keys_order
    ]


def update_sheet(sheet: gspread.spreadsheet.Spreadsheet, transactions: list[dict]) -> int:
    expenses_rows = sheet.worksheet("expenses").get_values()
    income_rows = sheet.worksheet("income").get_values()

    new_expenses_rows = []
    new_income_rows = []

    for transaction in transactions:
        if transaction is None:
            continue
        if transaction["type"] == "expense":
            row = convert_transaction_to_row(transaction, expenses_rows[0])
            if row not in expenses_rows:
                new_expenses_rows.append(row)
        else:
            if transaction["amount"] == 16000:
                transaction["category"] = "Paycheck"
                transaction["name"] = "Paycheck"
            row = convert_transaction_to_row(transaction, income_rows[0])
            if row not in income_rows:
                new_income_rows.append(row)

    sheet.worksheet("expenses").append_rows(new_expenses_rows)
    sheet.worksheet("income").append_rows(new_income_rows)

    return len(new_income_rows) + len(new_expenses_rows)


class AuthorizationError(Exception):
    pass


async def check_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if authorized_chats := os.getenv("AUTHORIZED_CHATS"):
        authorized_chats = json.loads(authorized_chats)
        if update.effective_chat.id not in [int(chat_id) for chat_id in authorized_chats]:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="*You are not authorized to use this bot.* "
                f"This chat id is: `{update.effective_chat.id}`",
            )
            return


async def bot_respond(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    sheet: gspread.spreadsheet.Spreadsheet,
) -> None:
    try:
        await check_authorization(update, context)
        user_message = update.message if update.message else update.channel_post
        transactions = parse_user_message(user_message.text)
        if (transaction := transactions[0]) and transaction.get("amount"):
            n_transactions = update_sheet(sheet, transactions)
            most_recent_date = max(transactions, key=lambda x: x["date"])["date"]
            if len(transactions) > 1:
                text = (
                    rf"Added *{n_transactions}* transactions to [sheet]({os.getenv('SHEETS_LINK')})\."
                    f"\nThe most recent transaction was made\non *{most_recent_date.strftime('%d %b, %Y')}*"
                )
            elif len(transactions) == 1:
                transaction["date"] = transaction["date"].strftime("%d %b, %Y")
                text = (
                    rf"Added *1* transaction to [sheet]({os.getenv('SHEETS_LINK')}):"
                    f"\n```json\n{json.dumps(transaction, indent=2)}\n```"
                    f"\nThe most recent transaction was made\non *{most_recent_date.strftime('%d %b, %Y')}*"
                )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=text,
                parse_mode="MarkdownV2",
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Couldn't parse any transactions from your message",
            )

    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Encountered an error:\n*{e!s}*",
            parse_mode="MarkdownV2",
        )


def main():
    sheet = get_sheet(os.getenv("SHEETS_LINK"))

    application = ApplicationBuilder().token(os.getenv("PARSER_API_TOKEN")).build()

    handler = MessageHandler(filters.TEXT & ~filters.COMMAND, partial(bot_respond, sheet=sheet))
    application.add_handler(handler)

    application.run_polling()


if __name__ == "__main__":
    main()
