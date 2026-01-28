# Dockerfile for testing Delta Lake with deletion vectors
FROM python:3.11-slim

WORKDIR /app

# Install Java (required for PySpark to create Delta tables with deletion vectors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Install Python dependencies
# Note: delta-spark 3.2.x is compatible with PySpark 3.5.x
# delta-spark 4.0.x requires Spark 4.0
RUN pip install --no-cache-dir \
    deltalake>=1.0.0 \
    pyspark==3.5.3 \
    delta-spark==3.2.1 \
    pandas \
    pyarrow

# Copy the test script and source module
COPY test_delta_lake.py /app/
COPY delta_lake_dataset.py /app/

CMD ["python", "test_delta_lake.py"]

