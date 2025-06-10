# Smart Expo Chatbot Mockup

## Ollama settings
- EMBEDDING_MODEL_NAME=qwen3:14b
- LANGUAGE_MODEL_NAME=qwen3:14b

## RAG & Memory Settings
- CHUNK_SIZE=1000
- CHUNK_OVERLAP=100
- SEARCH_RESULTS_K=5
- MEMORY_RETRIEVAL_K=3

## PostgreSQL settings
- PG_HOST=raspberrypi
- PG_PORT=5432
- PG_DATABASE=odoo
- PG_USER=postgres
- PG_PASSWORD=xxxx
- ENABLE_POSTGRES=True

## InfluxDB settings
- INFLUXDB_URL=http://raspberrypi:8086
- INFLUXDB_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxx
- INFLUXDB_ORG=WMG
- INFLUXDB_BUCKET=Expo
- ENABLE_INFLUXDB=True

## Web search
- TAVILY_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxx
- TAVILY_AVAILABLE=True

## Logging level
- LOG_LEVEL=INFO
