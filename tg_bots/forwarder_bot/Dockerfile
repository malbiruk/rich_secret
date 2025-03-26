FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir python-telegram-bot python-dotenv

COPY . .

CMD ["python", "app.py"]
