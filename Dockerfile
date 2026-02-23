FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV FLASK_APP=src/app.py
ENV MODEL_PATH=/app/models/vegetable_model.h5
EXPOSE 5000
CMD ["python","src/app.py"]
