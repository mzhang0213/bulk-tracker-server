FROM python:3.10

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# more lang support: RUN apt-get install -y tesseract-ocr-kor tesseract-ocr-eng

WORKDIR /app

COPY . .

RUN pip install flask flask-socketio opencv-python pytesseract matplotlib numpy eventlet Pillow

EXPOSE 3005

CMD ["python", "app.py"]
