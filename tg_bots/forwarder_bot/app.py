"""
This module creates a telegram bot which forwards SMS messages about transactions in ADCB bank
into a group with parser bot.
"""

import os

from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder

load_dotenv()


def main():
    application = ApplicationBuilder().token(os.getenv("FORWARDER_API_TOKEN")).build()
    application.run_polling()


if __name__ == "__main__":
    main()
