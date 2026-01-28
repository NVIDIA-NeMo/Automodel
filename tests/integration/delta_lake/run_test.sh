#!/bin/bash
# Run the Delta Lake deletion vectors test in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Delta Lake Deletion Vectors Integration Test"
echo "=============================================="
echo ""

# Copy the delta_lake_dataset.py module for testing
echo "Copying delta_lake_dataset.py for testing..."
cp "$PROJECT_ROOT/nemo_automodel/components/datasets/llm/delta_lake_dataset.py" ./delta_lake_dataset.py

echo "Building and running Delta Lake test container..."
echo ""

# Build and run with docker-compose
docker compose up --build --abort-on-container-exit
EXIT_CODE=$?

# Cleanup
rm -f ./delta_lake_dataset.py

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Test completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Test FAILED with exit code $EXIT_CODE"
    echo "=============================================="
    exit $EXIT_CODE
fi

