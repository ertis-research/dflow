FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port uvicorn will run on
EXPOSE 8000

# Command to run the application (replace main:app if your file/app is named differently)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]