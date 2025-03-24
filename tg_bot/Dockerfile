FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /root/.config/gspread
COPY service_account.json /root/.config/gspread/service_account.json

RUN pip install --no-cache-dir python-telegram-bot python-dotenv gspread

COPY . .

CMD ["python", "tg_bot.py"]
