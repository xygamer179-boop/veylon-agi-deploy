FROM python:3.12

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

# Ensure port 7860 is available for Hugging Face Spaces
EXPOSE 7860

CMD ["python", "api.py"]