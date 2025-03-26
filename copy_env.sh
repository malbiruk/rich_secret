#!/bin/bash

# Path to the original .env file
ENV_FILE="$(pwd)/.env"

# Ensure the .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create the .env file first."
    exit 1
fi

# Create directories if they don't exist
mkdir -p tg_bots/parser_bot
mkdir -p tg_bots/forwarder_bot

# Create symlinks
cp "$ENV_FILE" "tg_bots/parser_bot/.env"
cp "$ENV_FILE" "tg_bots/forwarder_bot/.env"

# Verify symlinks were created
if [ -f "tg_bots/parser_bot/.env" ] && [ -f "tg_bots/forwarder_bot/.env" ]; then
    echo "Files copied successfully:"
    echo "- tg_bots/parser_bot/.env -> $ENV_FILE"
    echo "- tg_bots/forwarder_bot/.env -> $ENV_FILE"
else
    echo "Error: Failed to create one or more symlinks."
    exit 1
fi

exit 0
