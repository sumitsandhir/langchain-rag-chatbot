FROM python:3.10

WORKDIR /app

COPY app.py .

RUN pip install streamlit requests

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
