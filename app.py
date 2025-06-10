import streamlit as st
import os
import time
import json
import logging
import re
import uuid
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as _connection
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import WriteOptions

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

BACKEND_RAG_SOURCE_DIR = os.path.join(BASE_DIR, "backend_kb_docs")
BACKEND_RAG_INDEX_DIR = os.path.join(DATA_DIR, "backend_rag_index")
HISTORY_INDEX_DIR = os.path.join(DATA_DIR, "chat_history_index")
SESSIONS_METADATA_FILE = os.path.join(DATA_DIR, "sessions_metadata.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKEND_RAG_SOURCE_DIR, exist_ok=True)
os.makedirs(BACKEND_RAG_INDEX_DIR, exist_ok=True)
os.makedirs(HISTORY_INDEX_DIR, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large:latest")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:14b")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

POSTGRES_HOST = os.getenv("PG_HOST", "raspberrypi")
POSTGRES_PORT = int(os.getenv("PG_PORT", "5432"))
POSTGRES_DATABASE = os.getenv("PG_DATABASE", "")
POSTGRES_USER = os.getenv("PG_USER", "")
POSTGRES_PASSWORD = os.getenv("PG_PASSWORD", "")
POSTGRES_CONFIGURED = os.getenv("ENABLE_POSTGRES", "False").lower() == "true"

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://raspberrypi:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "")
INFLUXDB_CONFIGURED = os.getenv("ENABLE_INFLUXDB", "False").lower() == "true"

class PostgresConnector:
    def __init__(self):
        self.conn_params = {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "database": POSTGRES_DATABASE,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
        }
        self._is_connected = False
        self.connection: Optional[_connection] = None

        if not POSTGRES_CONFIGURED:
            logger.error("PostgreSQL configuration is incomplete or disabled. Skipping connection attempt.")
            return

        try:
            logger.info(f"Attempting to connect to PostgreSQL at {self.conn_params.get('host')}:{self.conn_params.get('port')}...")
            self.connection = psycopg2.connect(**self.conn_params)
            self.connection.autocommit = True
            self._is_connected = True
            logger.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}", exc_info=True)
            self.connection = None
            self._is_connected = False

    def is_connected(self) -> bool:
        if self.connection is None:
            return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return self._is_connected
        except Exception:
            self._is_connected = False
            return False

    # In class PostgresConnector:
    def get_schema_description(self) -> str:
        if not self.is_connected():
            return "PostgreSQL database not connected, schema unavailable."

        tables_of_interest_desc = {
            "public.mrp_workorder": "Work Orders (Details about manufacturing work orders)",
            "public.mrp_production": "Production Orders (Manufacturing production orders)",
            "public.product_product": "Product Variants (Specific product variants)",
            "public.product_template": "Product Templates (General product information)",
            "public.crm_lead": "CRM Leads/Opportunities (Sales leads and opportunities)",
            "public.res_partner": "Partners (Customer, vendor, and contact information)",
            "cobot_data.rtde_logs": "Cobot RTDE Logs (Historical sensor data from cobot, including joints, forces, temperatures, and modes)"
        }

        rtde_logs_columns_definition = [
            {"column_name": "timestamp", "data_type": "timestamp with time zone"},
            {"column_name": "joint1", "data_type": "double precision"}, {"column_name": "joint2", "data_type": "double precision"},
            {"column_name": "joint3", "data_type": "double precision"}, {"column_name": "joint4", "data_type": "double precision"},
            {"column_name": "joint5", "data_type": "double precision"}, {"column_name": "joint6", "data_type": "double precision"},
            {"column_name": "fx", "data_type": "double precision"}, {"column_name": "fy", "data_type": "double precision"},
            {"column_name": "fz", "data_type": "double precision"}, {"column_name": "tx", "data_type": "double precision"},
            {"column_name": "ty", "data_type": "double precision"}, {"column_name": "tz", "data_type": "double precision"},
            {"column_name": "temp1", "data_type": "double precision"}, {"column_name": "temp2", "data_type": "double precision"},
            {"column_name": "temp3", "data_type": "double precision"}, {"column_name": "temp4", "data_type": "double precision"},
            {"column_name": "temp5", "data_type": "double precision"}, {"column_name": "temp6", "data_type": "double precision"},
            {"column_name": "robot_mode", "data_type": "integer"}, {"column_name": "safety_mode", "data_type": "integer"}
        ]

        final_schema_representation = {}
        try:
            with self.connection.cursor() as cursor:
                for full_table_name_str, friendly_description in tables_of_interest_desc.items():
                    schema_name, table_name_simple = full_table_name_str.split('.')
                    
                    current_table_columns = []
                    if full_table_name_str == "cobot_data.rtde_logs":
                        current_table_columns = rtde_logs_columns_definition
                    else:
                        cursor.execute(sql.SQL("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s;
                        """), [schema_name, table_name_simple])
                        db_columns = cursor.fetchall()
                        
                        if not db_columns:
                            logger.warning(f"Table {full_table_name_str} not found or has no columns in information_schema. Will be excluded from schema description.")
                            continue
                        for col in db_columns:
                            current_table_columns.append({"column_name": col[0], "data_type": col[1]})
                    
                    final_schema_representation[full_table_name_str] = {
                        "description": friendly_description,
                        "columns": current_table_columns
                    }
            
            if not final_schema_representation:
                return "No schema information could be retrieved for the specified tables of interest. Ensure tables exist and are accessible."
            return json.dumps(final_schema_representation, indent=2)

        except Exception as e:
            logger.error(f"Error fetching PostgreSQL schema for specified tables: {e}", exc_info=True)
            return f"Error fetching PostgreSQL schema: {e}"
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[Dict[str, Any]]]:
        if not self.is_connected():
            logger.error("PostgreSQL connection is not available to execute query.")
            return [{"error": "PostgreSQL connection not available."}]

        if not query.strip().upper().startswith("SELECT"):
            logger.warning(f"Attempted to execute non-SELECT query: {query[:100]}...")
            return [{"error": "Only read-only SELECT queries are allowed."}]

        logger.info(f"Executing PostgreSQL query: {query[:200]}...")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    logger.info(f"PostgreSQL query executed successfully. Returned {len(results)} rows.")
                    logger.debug(f"PostgreSQL query full output: {results}")
                    return results
                else:
                    logger.info("PostgreSQL SELECT query executed but returned no description (possibly empty result set).")
                    return []

        except Exception as e:
            logger.error(f"Error executing PostgreSQL query: {e}", exc_info=True)
            return [{"error": str(e)}]

    def get_recent_work_orders(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        work_orders_table = sql.Identifier("mrp_workorder")
        production_table = sql.Identifier("mrp_production")
        product_variant_table = sql.Identifier("product_product")
        products_table = sql.Identifier("product_template")

        query = sql.SQL("""
            SELECT w.name AS work_order, p.name AS product, w.state, w.date_planned_start
            FROM {work_orders_table} w
            JOIN {production_table} pr ON w.production_id = pr.id
            JOIN {product_variant_table} pp ON w.product_id = pp.id
            JOIN {products_table} p ON pp.product_tmpl_id = p.id
            ORDER BY w.date_planned_start DESC
            LIMIT %s;
        """).format(
            work_orders_table=work_orders_table,
            production_table=production_table,
            product_variant_table=product_variant_table,
            products_table=products_table
        )
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_recent_opportunities_with_orders(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        crm_table = sql.Identifier("crm_lead")
        partner_table = sql.Identifier("res_partner")
        production_table = sql.Identifier("mrp_production")

        query = sql.SQL("""
            SELECT crm.name AS opportunity, res.name AS customer, pr.name AS production_order
            FROM {crm_table} crm
            JOIN {partner_table} res ON crm.partner_id = res.id
            LEFT JOIN {production_table} pr ON crm.name = pr.origin
            ORDER BY crm.create_date DESC
            LIMIT %s;
        """).format(
            crm_table=crm_table,
            partner_table=partner_table,
            production_table=production_table
        )
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_recent_customers(self, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        partner_table = sql.Identifier("res_partner")

        query = sql.SQL("""
            SELECT name, email, phone
            FROM {partner_table}
            ORDER BY create_date DESC
            LIMIT %s;
        """).format(partner_table=partner_table)
        return self.execute_query(query.as_string(self.connection), (limit,))

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        tools = []
        if self.is_connected():
            tools.append({
                "name": "postgres_query",
                "description": (
                    "Use this tool to execute read-only SQL queries against the PostgreSQL database "
                    "containing Odoo ERP data (work orders, production, CRM, customers, etc.). "
                    "Input should be a valid SQL SELECT query string. "
                    "Only use SELECT queries. Do not attempt INSERT, UPDATE, DELETE, or DDL commands. "
                    "If you need schema information, request the 'get_postgres_schema' tool, then form your query based on that."
                ),
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The SQL SELECT query string."}}, "required": ["query"]}
            })
            tools.append({
                "name": "get_recent_work_orders",
                "description": (
                    "Retrieves the most recent work orders from the PostgreSQL database. "
                    "This tool is useful for getting an overview of manufacturing tasks. "
                    "It takes an optional integer `limit` parameter to specify the number of recent orders to retrieve (default is 5). "
                    "Example: 'Show me the last 10 work orders' or 'What are the recent work orders?'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of work orders to retrieve."}}}
            })
            tools.append({
                "name": "get_recent_opportunities_with_orders",
                "description": (
                    "Retrieves recent CRM opportunities along with associated production orders from PostgreSQL. "
                    "Useful for tracking sales pipeline and manufacturing linkage. "
                    "Takes an optional integer `limit` parameter (default is 5)."
                    "Example: 'What are our latest sales opportunities?' or 'Show me recent opportunities and their production orders.'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of opportunities to retrieve."}}}
            })
            tools.append({
                "name": "get_recent_customers",
                "description": (
                    "Fetches information about the most recently added customers from PostgreSQL. "
                    "Useful for reviewing new client acquisitions. "
                    "Takes an optional integer `limit` parameter (default is 5)."
                    "Example: 'Who are our newest customers?' or 'List the 5 most recent customers.'"
                ),
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "The maximum number of customers to retrieve."}}}
            })
            tools.append({
                "name": "get_postgres_schema",
                "description": (
                    "Retrieves a detailed description of the PostgreSQL database schema (tables and columns). "
                    "Use this when the user asks about database structure or when you need to form a complex SQL query. "
                    "This tool takes no parameters."
                ),
                "parameters": {"type": "object", "properties": {}}
            })

        return tools

    def get_tool_function(self, tool_name: str):
        if not self.is_connected():
            return None
        if tool_name == "postgres_query":
            return self.execute_query
        if tool_name == "get_recent_work_orders":
            return self.get_recent_work_orders
        if tool_name == "get_recent_opportunities_with_orders":
            return self.get_recent_opportunities_with_orders
        if tool_name == "get_recent_customers":
            return self.get_recent_customers
        if tool_name == "get_postgres_schema":
            return self.get_schema_description
        return None

class InfluxDBConnector:
    def __init__(self):
        self.config = {
            "url": INFLUXDB_URL,
            "token": INFLUXDB_TOKEN,
            "org": INFLUXDB_ORG,
            "bucket": INFLUXDB_BUCKET
        }
        self._is_connected = False
        self.client: Optional[InfluxDBClient] = None
        self.query_api = None

        if not INFLUXDB_CONFIGURED:
            logger.error("InfluxDB configuration is incomplete or disabled. Skipping connection attempt.")
            return

        try:
            logger.info(f"Attempting to connect to InfluxDB at {self.config.get('url')}...")
            self.client = InfluxDBClient(
                url=self.config["url"],
                token=self.config["token"],
                org=self.config["org"]
            )
            if not self.client.ping():
                raise ConnectionError("InfluxDB ping failed. Server might be down or unreachable.")

            self.query_api = self.client.query_api()
            self._is_connected = True
            logger.info("Connected to InfluxDB successfully.")
        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}", exc_info=True)
            self.client = None
            self.query_api = None
            self._is_connected = False

    def is_connected(self) -> bool:
        return self._is_connected

    # Inside class InfluxDBConnector:
    def get_schema_description(self) -> str:
        if not self.is_connected():
            return "InfluxDB database not connected, schema unavailable."
        
        org_name = self.config.get("org")
        bucket_name = self.config.get("bucket")
        if not org_name or not bucket_name:
            return "InfluxDB organization or bucket not configured, schema unavailable."

        schema_info = {"buckets": {}}
        try:
            flux_query_measurements = f'''
            import "influxdata/influxdb/schema"
            schema.measurements(bucket: "{bucket_name}")
            '''
            tables_measurements = self.query_api.query(query=flux_query_measurements, org=org_name)
            
            measurements = [record.get_value() for table in tables_measurements for record in table.records]
            
            schema_info["buckets"][bucket_name] = {"measurements": {}}
            for measurement in measurements:
                # Corrected queries for fields and tags per measurement
                fields_query = f'''
                import "influxdata/influxdb/schema"
                schema.measurementFieldKeys(bucket: "{bucket_name}", measurement: "{measurement}")
                '''
                tags_query = f'''
                import "influxdata/influxdb/schema"
                schema.measurementTagKeys(bucket: "{bucket_name}", measurement: "{measurement}")
                '''
                
                fields_tables = self.query_api.query(query=fields_query, org=org_name)
                tags_tables = self.query_api.query(query=tags_query, org=org_name)
                
                fields = [record.get_value() for table in fields_tables for record in table.records]
                tags = [record.get_value() for table in tags_tables for record in table.records]
                
                schema_info["buckets"][bucket_name]["measurements"][measurement] = {
                    "fields": fields,
                    "tags": tags
                }
            
            if "cobot_telemetry" in schema_info["buckets"][bucket_name]["measurements"]:
                 schema_info["buckets"][bucket_name]["measurements"]["cobot_telemetry"]["description"] = "Real-time data from the cobot, including joint angles, forces, temperatures, and robot/safety modes."
            else:
                 logger.warning(f"cobot_telemetry measurement not found in bucket {bucket_name} during schema description.")

            return json.dumps(schema_info, indent=2)

        except Exception as e:
            logger.error(f"Error fetching InfluxDB schema: {e}", exc_info=True)
            return f"Error fetching schema: {e}"

    def execute_flux_query(self, flux_query: str) -> Optional[List[Dict[str, Any]]]:
        if not self.is_connected() or self.query_api is None:
            logger.error("InfluxDB connection or query API not available to execute query.")
            return [{"error": "InfluxDB connection not available."}]

        bucket_name = self.config.get("bucket")
        org_name = self.config.get("org")

        if not bucket_name or not org_name:
            logger.error("InfluxDB bucket or organization name is missing in config.")
            return [{"error": "InfluxDB bucket or organization is not configured."}]

        if not re.search(r'from\s*\(\s*bucket:\s*["\']?.*?["\']?\s*\)', flux_query, re.IGNORECASE):
            default_prefix = f'from(bucket: "{bucket_name}") |> range(start: -15m)'
            if flux_query.strip().startswith('|'):
                flux_query = f'{default_prefix} {flux_query.strip()}'
                logger.warning(f"Prepended default bucket/range to Flux query: {flux_query[:200]}...")
            else:
                flux_query = f'{default_prefix}\n{flux_query.strip()}'
                logger.warning(f"Prepended default bucket/range to Flux query: {flux_query[:200]}...")

        logger.info(f"Executing InfluxDB Flux query: {flux_query[:200]}...")

        try:
            tables = self.query_api.query(query=flux_query, org=org_name)

            if not tables:
                logger.info("InfluxDB query returned no results.")
                return []

            results_list: List[Dict[str, Any]] = []
            for table in tables:
                for record in table.records:
                    results_list.append(record.values)

            logger.info(f"InfluxDB Flux query executed successfully. Returned {len(results_list)} records.")
            logger.debug(f"InfluxDB Flux query full output: {results_list}")
            return results_list

        except InfluxDBError as e:
            logger.error(f"InfluxDB API Error executing Flux query: {e}", exc_info=True)
            error_message = getattr(e, 'message', str(e))
            return [{"error": f"InfluxDB query failed: {error_message}"}]
        except Exception as e:
            logger.error(f"An unexpected error occurred during query execution: {e}", exc_info=True)
            return [{"error": f"An unexpected error occurred: {str(e)}"}]

    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        if not self.is_connected() or self.query_api is None:
            logger.error("InfluxDB connection or query API not available to get latest telemetry.")
            return {"error": "Not connected to InfluxDB"}

        bucket_name = self.config.get("bucket")
        org_name = self.config.get("org")

        if not bucket_name or not org_name:
            logger.error("InfluxDB bucket or organization name is missing in config for latest telemetry.")
            return {"error": "InfluxDB bucket or organization is not configured."}

        flux_query = f'''
        from(bucket: "{bucket_name}")
          |> range(start: -5m)
          |> filter(fn: (r) => r._measurement == "cobot_telemetry")
          |> filter(fn: (r) =>
              r._field == "joint1" or
              r._field == "fx" or
              r._field == "temp1" or
              r._field == "robot_mode" or
              r._field == "safety_mode"
          )
          |> last()
          |> yield(name: "last_values")
        '''

        logger.info("Executing InfluxDB query for latest telemetry...")

        try:
            result = self.query_api.query(org=org_name, query=flux_query)

            if not result:
                logger.info("InfluxDB latest telemetry query returned no results.")
                return {"info": "No recent telemetry available"}

            latest_data: Dict[str, Any] = {}
            for table in result:
                for record in table.records:
                    field = record.get_field()
                    value = record.get_value()

                    if field is not None:
                        latest_data[field] = value

            if not latest_data:
                logger.info("Processed InfluxDB latest telemetry results but no fields were found.")
                return {"info": "No recent telemetry available with specified fields"}

            logger.info(f"InfluxDB latest telemetry query executed successfully. Found data for fields: {list(latest_data.keys())}")
            logger.debug(f"InfluxDB latest telemetry data: {latest_data}")
            return latest_data

        except InfluxDBError as e:
            logger.error(f"InfluxDB API Error getting latest telemetry: {e}", exc_info=True)
            error_message = getattr(e, 'message', str(e))
            return {"error": f"InfluxDB query failed: {error_message}"}
        except Exception as e:
            logger.error(f"An unexpected error getting latest InfluxDB telemetry: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        tools = []
        if self.is_connected():
            tools.append({
                "name": "influxdb_query",
                "description": (
                    "Use this tool to execute Flux queries against the InfluxDB database "
                    "containing real-time Cobot telemetry data. Input should be a valid Flux query string. "
                    "This is useful for getting specific time series data, aggregations, or filtered results. "
                    "Example query: `from(bucket: \"your_cobot_bucket\") |> range(start: -15m) |> filter(fn: (r) => r[\"_measurement\"] == \"cobot_sensor\" and r[\"_field\"] == \"temperature\") |> last()`"
                    "If your query doesn't specify a bucket or range, the default configured bucket and a range of the last 15 minutes will be used automatically."
                    "Be specific about the time range and fields you are interested in."
                    "If you need schema information, request the 'get_influxdb_schema' tool, then form your query based on that."
                ),
                "parameters": {"type": "object", "properties": {"flux_query": {"type": "string", "description": "The Flux query string."}}, "required": ["flux_query"]}
            })
            tools.append({
                "name": "get_latest_telemetry",
                "description": (
                    "Retrieves the latest telemetry data for key cobot metrics (joint1, fx, temp1, robot_mode, safety_mode) "
                    "from the InfluxDB database. This is ideal for quick checks on current robot status. "
                    "This tool does not take any parameters."
                    "Example: 'What's the latest robot telemetry?' or 'Give me the current cobot status.'"
                ),
                "parameters": {"type": "object", "properties": {}}
            })
            tools.append({
                "name": "get_influxdb_schema",
                "description": (
                    "Retrieves a detailed description of the InfluxDB database schema (buckets, measurements, fields, and tags). "
                    "Use this when the user asks about database structure or when you need to form a complex Flux query. "
                    "This tool takes no parameters."
                ),
                "parameters": {"type": "object", "properties": {}}
            })

        return tools

    def get_tool_function(self, tool_name: str):
        if not self.is_connected():
            return None

        if tool_name == "influxdb_query":
            return self.execute_flux_query
        if tool_name == "get_latest_telemetry":
            return self.get_latest_telemetry
        if tool_name == "get_influxdb_schema":
            return self.get_schema_description
        return None

def load_sessions_metadata_util():
    if os.path.exists(SESSIONS_METADATA_FILE):
        try:
            with open(SESSIONS_METADATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Error decoding sessions_metadata.json. Starting with empty metadata.")
            return {}
    return {}

def save_sessions_metadata_util(metadata):
    with open(SESSIONS_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_chat_history_util(session_id):
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    if os.path.exists(history_file):
        try:
            # Add encoding='utf-8' here
            with open(history_file, 'r', encoding='utf-8') as f: 
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding chat_history_{session_id}.json. Returning empty history.")
            return []
        except Exception as e:
            logger.error(f"Error reading chat_history_{session_id}.json: {e}", exc_info=True)
            return []
    return []

def save_chat_history_util(session_id, chat_history, sessions_metadata):
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

    if session_id in sessions_metadata:
        sessions_metadata[session_id]['last_active'] = time.time()
        sessions_metadata[session_id]['name'] = get_session_title(chat_history)
        save_sessions_metadata_util(sessions_metadata)

def generate_session_id() -> str:
    return f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"

def sanitize_text_for_title(text: str) -> str:
    text = re.sub(r'[^\w\s-]', '', text).strip()
    text = re.sub(r'[-\s]+', ' ', text)
    return text[:50]

def get_session_title(history: list) -> str:
    if history:
        user_messages = [m['content'] for m in history if m['role'] == 'user']
        if user_messages:
            first_user_message_content = user_messages[0].strip()
            if first_user_message_content:
                title = sanitize_text_for_title(first_user_message_content)
                return title if title else DEFAULT_CHAT_TITLE
    return DEFAULT_CHAT_TITLE

def create_new_session_util(sessions_metadata):
    new_session_id = generate_session_id()
    sessions_metadata[new_session_id] = {
        "id": new_session_id,
        "name": DEFAULT_CHAT_TITLE,
        "created_at": time.time(),
        "last_active": time.time(),
        "chat_history_file": f"chat_history_{new_session_id}.json"
    }
    save_sessions_metadata_util(sessions_metadata)
    return new_session_id

def delete_session_util(session_id, sessions_metadata):
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    if os.path.exists(history_file):
        os.remove(history_file)
    if session_id in sessions_metadata:
        del sessions_metadata[session_id]
        save_sessions_metadata_util(sessions_metadata)

def load_faiss_store_util(index_dir, embeddings_model):
    if os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0:
        try:
            logger.info(f"Loading FAISS index from {index_dir}...")
            faiss_index = FAISS.load_local(index_dir, embeddings_model, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
            return faiss_index
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_dir}: {e}", exc_info=True)
            return None
    logger.info(f"No existing FAISS index found at {index_dir}.")
    return None

def save_faiss_store_util(faiss_index, index_dir):
    try:
        if faiss_index:
            logger.info(f"Saving FAISS index to {index_dir}...")
            os.makedirs(index_dir, exist_ok=True)
            faiss_index.save_local(index_dir)
            logger.info("FAISS index saved successfully.")
            return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {index_dir}: {e}", exc_info=True)
    return False

def build_backend_rag_index_util(source_dir, index_dir, embeddings_model, text_splitter):
    documents = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(source_dir, filename)
            logger.info(f"Loading document: {filename}")
            try:
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if documents:
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
        new_index = FAISS.from_documents(chunks, embeddings_model)
        save_faiss_store_util(new_index, index_dir)
        return new_index
    logger.warning("No documents found to build backend RAG index.")
    return None

def clean_llm_output(text: str) -> str:
    cleaned_text = re.sub(r"<tool_code>.*?</tool_code>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<execute_result>.*?</execute_result>", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<(?:think)>.*?</(?:think)>", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

class ChatbotCore:
    def __init__(self, embedding_model_name: str, llm_model_name: str, ollama_base_url: str, temperature: float):
        self.embedding_model = None
        self.language_model = None
        self.backend_rag_vector_store = None
        self.history_vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.postgres_connector = None
        self.influxdb_connector = None

        self._initialize_models(embedding_model_name, llm_model_name, ollama_base_url, temperature)
        self._initialize_connectors()
        self._initialize_vector_stores()

    def _initialize_models(self, embedding_model_name, llm_model_name, ollama_base_url, temperature):
        try:
            logger.info(f"Initializing embedding model: {embedding_model_name} from {ollama_base_url}")
            self.embedding_model = OllamaEmbeddings(model=embedding_model_name, base_url=ollama_base_url)
            self.embedding_model.embed_query("test query")
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            self.embedding_model = None

        try:
            logger.info(f"Initializing LLM: {llm_model_name} from {ollama_base_url}")
            self.language_model = OllamaLLM(
                model=llm_model_name,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            self.language_model.invoke("Hello") 
            logger.info("LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            self.language_model = None
    
    def _initialize_connectors(self):
        if POSTGRES_CONFIGURED:
            self.postgres_connector = PostgresConnector()
            if not self.postgres_connector.is_connected():
                logger.warning("PostgresConnector failed to connect or is not configured correctly.")
                self.postgres_connector = None
            else:
                logger.info("PostgresConnector initialized and connected.")
        else:
            logger.info("PostgreSQL is not configured, skipping PostgresConnector initialization.")

        if INFLUXDB_CONFIGURED:
            self.influxdb_connector = InfluxDBConnector()
            if not self.influxdb_connector.is_connected():
                logger.warning("InfluxDBConnector failed to connect or is not configured correctly.")
                self.influxdb_connector = None
            else:
                logger.info("InfluxDBConnector initialized and connected.")
        else:
            logger.info("InfluxDB is not configured, skipping InfluxDBConnector initialization.")

    def _initialize_vector_stores(self):
        if self.embedding_model:
            self.backend_rag_vector_store = load_faiss_store_util(BACKEND_RAG_INDEX_DIR, self.embedding_model)
            if not self.backend_rag_vector_store:
                self.backend_rag_vector_store = build_backend_rag_index_util(BACKEND_RAG_SOURCE_DIR, BACKEND_RAG_INDEX_DIR, self.embedding_model, self.text_splitter)
            
            self.history_vector_store = load_faiss_store_util(HISTORY_INDEX_DIR, self.embedding_model)
            if not self.history_vector_store:
                logger.info("Creating empty FAISS index for chat history.")
                self.history_vector_store = FAISS.from_texts(["initialization"], self.embedding_model)
                save_faiss_store_util(self.history_vector_store, HISTORY_INDEX_DIR)
        else:
            logger.error("Embedding model not initialized. RAG and history features will be disabled.")

    def _get_all_available_tools_description(self) -> str:
        all_tools_descriptions = []
        if self.postgres_connector:
            all_tools_descriptions.extend(self.postgres_connector.get_tool_descriptions())
        if self.influxdb_connector:
            all_tools_descriptions.extend(self.influxdb_connector.get_tool_descriptions())
        
        formatted_tools = []
        for tool in all_tools_descriptions:
            tool_info = f"  - Name: {tool['name']}\n    Description: {tool['description']}"
            if 'parameters' in tool and tool['parameters']['properties']:
                params = ", ".join([f"{k} ({v['type']})" for k, v in tool['parameters']['properties'].items()])
                tool_info += f"\n    Parameters: {params}"
                if 'required' in tool['parameters'] and tool['parameters']['required']:
                    tool_info += f" (Required: {', '.join(tool['parameters']['required'])})"
            formatted_tools.append(tool_info)
        
        if formatted_tools:
            return "Available Database Tools:\n" + "\n".join(formatted_tools)
        return "No database tools are currently available."

    def _get_tool_function_by_name(self, tool_name: str):
        if self.postgres_connector:
            func = self.postgres_connector.get_tool_function(tool_name)
            if func: return func
        if self.influxdb_connector:
            func = self.influxdb_connector.get_tool_function(tool_name)
            if func: return func
        return None

    def generate_answer(
        self,
        user_query: str,
        context_documents: Optional[str] = None,
        chat_history_str: Optional[str] = None,
        db_results: Optional[str] = None,
        current_mode: str = "Chat 💬"
    ) -> Generator[str, None, None]:
        if not self.language_model:
            yield "LLM is not initialized. Cannot generate answer."
            return

        system_message = ""
        user_message_content = user_query

        if current_mode == "Conversational 💬":
            system_message = (
                "You are a helpful and knowledgeable assistant."
            )
            if context_documents:
                system_message += f"\n\nContext:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        elif current_mode == "Knowledge Base 📚":
            system_message = (
                f"You are an expert assistant for retrieving information from the documents saved in the {BACKEND_RAG_SOURCE_DIR} folder in this chatbot_app. "
                f"Answer the user's questions based on the provided documents in the {BACKEND_RAG_SOURCE_DIR} folder. "
                f"The {BACKEND_RAG_SOURCE_DIR} folder is also refered to as the knowledge base or backend knowledge base. "
                "If the information is not found in the documents, tell the user the information is not found in the documents, but use your own general knowledge to the best of your ability to answer the question asked. "
                "Do not explicitly mention if information was found in the documents or not, just integrate it naturally if relevant."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."
            )
            if context_documents:
                system_message += f"\n\nBackend Knowledge Base Documents:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        elif current_mode == "Database Search 🗃️":
            tool_descriptions = self._get_all_available_tools_description()
            
            postgres_schema_desc_str = ""
            if self.postgres_connector and self.postgres_connector.is_connected():
                postgres_schema_raw = self.postgres_connector.get_schema_description()
                if postgres_schema_raw and not postgres_schema_raw.startswith("Error") and \
                   postgres_schema_raw != "No schema information could be retrieved for the specified tables of interest. Ensure tables exist and are accessible." and \
                   postgres_schema_raw != "PostgreSQL database not connected, schema unavailable.":
                    postgres_schema_desc_str = f"\n\nPostgreSQL Schema (Use table names like 'public.mrp_workorder' or 'cobot_data.rtde_logs' for queries):\n{postgres_schema_raw.replace('{', '{{').replace('}', '}}')}"
                else:
                    logger.warning(f"PostgreSQL schema description was empty or an error: '{postgres_schema_raw}'")
                    postgres_schema_desc_str = "\n\nPostgreSQL Schema: Currently unavailable or contains no relevant tables of interest."


            influxdb_schema_desc_str = ""
            if self.influxdb_connector and self.influxdb_connector.is_connected():
                influxdb_schema_raw = self.influxdb_connector.get_schema_description()
                if influxdb_schema_raw and not influxdb_schema_raw.startswith("Error") and \
                   influxdb_schema_raw != "InfluxDB database not connected, schema unavailable.":
                    influxdb_schema_desc_str = f"\n\nInfluxDB Schema (Bucket: '{self.influxdb_connector.config.get('bucket', 'Expo')}', Measurement for cobot data is typically 'cobot_telemetry'):\n{influxdb_schema_raw.replace('{', '{{').replace('}', '}}')}"
                else:
                    logger.warning(f"InfluxDB schema description was an error or unavailable: '{influxdb_schema_raw}'")
                    influxdb_schema_desc_str = "\n\nInfluxDB Schema: Currently unavailable or contains no relevant measurements."


            system_message = (
                "You are an expert database assistant for manufacturing ERP data (in PostgreSQL) and cobot telemetry (real-time/recent in InfluxDB, historical in PostgreSQL table `cobot_data.rtde_logs`). "
                "Your primary goal is to answer user questions by interacting with these databases using ONLY the provided tools and schemas. Adhere strictly to the following:\n"
                "1.  **Analyze Provided Schemas**: Your first step is to CAREFULLY review the PostgreSQL and InfluxDB schemas detailed below. All your database actions MUST be based EXCLUSIVELY on this information. DO NOT assume or hallucinate any other tables, measurements, fields, columns, or functions.\n"
                "    - **PostgreSQL**: Contains Odoo ERP tables (e.g., `public.mrp_workorder`, `public.crm_lead`, `public.res_partner` for Customers) and a crucial table `cobot_data.rtde_logs` for HISTORICAL cobot sensor data (joint positions, forces, temperatures, robot/safety modes over time).\n"
                "    - **InfluxDB**: Contains the `cobot_telemetry` measurement for RECENT or REAL-TIME cobot sensor data (joint positions, forces, temperatures, robot/safety modes). The bucket is 'Expo'.\n"
                
                "2.  **Prioritize Specialized Functions (Tools) & Map Common Terms**:\n"
                "    - For LATEST/CURRENT cobot status (e.g., 'latest temperature', 'current robot mode'), you MUST use the `get_latest_telemetry` function (InfluxDB).\n"
                "    - For common Odoo ERP requests, you MUST prioritize these specialized functions if they match the user's intent:\n"
                "        - User asks about 'newest customers', 'recent customers', 'list customers': Use the `get_recent_customers` function. (This queries the `public.res_partner` table which stores customer data).\n"
                "        - User asks about 'recent sales opportunities', 'latest leads', 'sales pipeline': Use the `get_recent_opportunities_with_orders` function. (This queries `public.crm_lead` and related tables).\n"
                "        - User asks about 'recent work orders', 'manufacturing tasks': Use the `get_recent_work_orders` function. (This queries `public.mrp_workorder`).\n"
                "    - If a specialized function perfectly fits, use it instead of trying to generate a raw SQL query from scratch for these common requests.\n"

                "3.  **Schema Inquiry Protocol**: If the user asks 'what tables exist?', 'describe measurements', 'what is the schema for customers?', or any similar question about database structure, you MUST call the `get_postgres_schema` function for PostgreSQL details or the `get_influxdb_schema` function for InfluxDB details. Provide the output of these functions. Do not invent schema; always use these functions for schema questions.\n"
                
                "4.  **Formulating Raw Queries (When Specialized Functions Don't Apply or for Other Tables)**:\n"
                "    - When formulating SQL queries for Odoo data not covered by a specialized function, or if the user asks for data from a table not covered by a specific tool, remember these mappings if the user uses common terms (always verify against the provided schema description below which lists full table names and their purpose):\n"
                "        - 'Customers', 'Partners', or 'Contacts' usually refers to the `public.res_partner` table.\n"
                "        - 'Production Orders' or 'Manufacturing Orders' usually refers to the `public.mrp_production` table.\n"
                "        - 'Products' can refer to `public.product_template` (general product definitions) or `public.product_product` (specific product variants).\n"
                "        - 'Work Orders' refers to `public.mrp_workorder`.\n"
                "        - 'Leads' or 'Opportunities' refers to `public.crm_lead`.\n"
                "    - Always refer to the full provided PostgreSQL schema for exact table names (like `public.res_partner`) and column names before constructing any query.\n"
                "    - For HISTORICAL cobot sensor data (e.g., temperature trend last hour, average force yesterday), generate a SQL SELECT query for the PostgreSQL table `cobot_data.rtde_logs`. Use the column definitions provided in its schema.\n"
                "    - For specific time-series analysis from InfluxDB not covered by `get_latest_telemetry` (e.g., average `temp3` over the last 30 minutes, or querying specific fields like `joint_temperatures_5`), generate a Flux query targeting the `cobot_telemetry` measurement in the 'Expo' bucket. Use field names like `temp1`-`temp6`, `joint_positions_0`-`joint_positions_5`, `tcp_forces_0`-`tcp_forces_5`, `robot_mode`, `safety_mode` as indicated by the data structure and schema.\n"

                "5.  **Non-Database Questions**: If the user's request is clearly unrelated to querying these specific databases, respond as a general helpful assistant without attempting database actions or outputting JSON.\n"
                f"\n\nAvailable Database Tools:\n{tool_descriptions.replace('{', '{{').replace('}', '}}') if tool_descriptions else 'No tools available.'}"
                f"{postgres_schema_desc_str}"
                f"{influxdb_schema_desc_str}"
                "\n\n"
                "**Response Format for Database Actions (MANDATORY - Adhere Strictly! Output ONLY the JSON block when performing an action):**\n"
                "To call a specific function: \n"
                "```json\n"
                "{{\n"
                "  \"action\": \"call_function\",\n"
                "  \"function_name\": \"<function_name_here>\",\n"
                "  \"arguments\": {{ <\"param1\": \"value1\", ...> }} \n"
                "}}\n"
                "```\n"
                "To execute a raw query: \n"
                "```json\n"
                "{{\n"
                "  \"action\": \"execute_query\",\n"
                "  \"query_type\": \"<postgres_sql_or_influxdb_flux>\",\n"
                "  \"query\": \"<your_SQL_SELECT_or_Flux_query_here>\"\n"
                "}}\n"
                "```\n"
                "For functions requiring no arguments (like schema functions), use `\"arguments\": {{}}`."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."
            )
            
            if db_results: 
                system_message = ( 
                    "You are an expert database assistant. You have just received the following results from a database query or function call. "
                    "Based SOLELY on these results, provide a concise and natural language answer to the user's original question. Do not refer to how you got the data (e.g., 'I ran a query'). Just present the information."
                    f"\n\n**Database Query/Function Results:**\n{db_results.replace('{', '{{').replace('}', '}}')}\n\n"
                    "Now, answer the user's original question using only this information."
                )
            
            if chat_history_str and not db_results: 
                system_message += f"\n\nRelevant Chat History:\n{chat_history_str.replace('{', '{{').replace('}', '}}')}"
            
        else:
            system_message = (
                "You are a helpful and knowledgeable assistant. "
                "Answer the user's questions based on the provided context and chat history. "
                "If the information is not available, use your own general knowledge to the best of your ability to answer the question asked."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."

            )
            if context_documents:
                system_message += f"\n\nContext:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message_content)
        ])

        try:
            chain = prompt | self.language_model
            for chunk in chain.stream({"user_query": user_query}):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}", exc_info=True)
            yield f"I'm sorry, I'm having trouble processing that request right now. Details: {e}"
            return

    def process_query(self, user_query: str, active_session_id: str, chat_history: List[Dict[str, str]], current_mode: str) -> Generator[str, None, None]:
        if not self.language_model:
            yield "LLM is not initialized. Please check model configuration."
            return
        if not self.embedding_model:
            yield "Embedding model not initialized. RAG features will be limited."

        retrieved_context = []
        retrieved_history = []
        
        
        if current_mode == "Knowledge Base 📚" and self.backend_rag_vector_store:
            try:
                retrieved_context_docs = self.backend_rag_vector_store.similarity_search(user_query, k=3)
                retrieved_context = [doc.page_content for doc in retrieved_context_docs]
                logger.info(f"Retrieved {len(retrieved_context)} context documents for Backend RAG.")
            except Exception as e:
                logger.error(f"Error retrieving from backend RAG: {e}", exc_info=True)
                retrieved_context = ["Error retrieving backend RAG documents."]

        if self.history_vector_store:
            try:
                retrieved_history_docs = self.history_vector_store.similarity_search(user_query, k=5)
                filtered_history_docs = [
                    doc for doc in retrieved_history_docs 
                    if doc.metadata.get('session_id') == active_session_id
                ]
                retrieved_history = [doc.page_content for doc in filtered_history_docs]
                logger.info(f"Retrieved {len(retrieved_history)} relevant chat history turns.")
            except Exception as e:
                logger.warning(f"Error retrieving from history vector store: {e}")
                retrieved_history = []

        retrieved_context_str = "\n".join(retrieved_context) if retrieved_context else None
        formatted_chat_history_str = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history]) if chat_history else None

        db_results_str = None
        
        if current_mode == "Database Search 🗃️":
            llm_suggestion_raw_chunks = list(self.generate_answer(
                user_query, 
                context_documents=retrieved_context_str, 
                chat_history_str=formatted_chat_history_str, 
                current_mode=current_mode
            ))
            llm_suggestion_raw = "".join(llm_suggestion_raw_chunks)

            logger.info(f"LLM raw suggestion for DB query: {llm_suggestion_raw[:500]}...")

            db_output_raw = None
            action_executed = False

            try:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_suggestion_raw, re.DOTALL)
                if json_match:
                    llm_json_output = json.loads(json_match.group(1))
                    
                    action_type = llm_json_output.get("action")

                    if action_type == "call_function":
                        func_name = llm_json_output.get("function_name")
                        func_args = llm_json_output.get("arguments", {})
                        
                        logger.info(f"LLM suggested calling function: {func_name} with args: {func_args}")
                        
                        target_func = self._get_tool_function_by_name(func_name)
                        if target_func:
                            try:
                                db_output_raw = target_func(**func_args)
                                action_executed = True
                                logger.info(f"Successfully executed function: {func_name}")
                            except TypeError as te:
                                db_results_str = f"Error: Function '{func_name}' called with incorrect arguments. Details: {te}"
                                logger.error(db_results_str, exc_info=True)
                            except Exception as e:
                                db_results_str = f"Error executing function '{func_name}': {e}"
                                logger.error(db_results_str, exc_info=True)
                        else:
                            db_results_str = f"Error: Unknown function suggested by LLM: '{func_name}'."
                            logger.warning(db_results_str)
                    
                    elif action_type == "execute_query":
                        query_type = llm_json_output.get("query_type")
                        query_str = llm_json_output.get("query")
                        
                        logger.info(f"LLM suggested executing raw query (type: {query_type}): {query_str[:200]}...")
                        
                        if query_type == "postgres_sql" and self.postgres_connector:
                            db_output_raw = self.postgres_connector.execute_query(query_str)
                            action_executed = True
                            logger.info("Successfully executed raw PostgreSQL query.")
                        elif query_type == "influxdb_flux" and self.influxdb_connector:
                            db_output_raw = self.influxdb_connector.execute_flux_query(query_str)
                            action_executed = True
                            logger.info("Successfully executed raw InfluxDB Flux query.")
                        else:
                            db_results_str = f"Error: Invalid query type '{query_type}' or connector not available."
                            logger.warning(db_results_str)

                if action_executed and db_output_raw is not None:
                    if isinstance(db_output_raw, list):
                        if not db_output_raw:
                            db_results_str = "Query executed, but no results found."
                        else:
                            def datetime_converter(o):
                                if isinstance(o, datetime):
                                    return o.isoformat()
                                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                            formatted_rows = []
                            for row in db_output_raw:
                                if isinstance(row, dict):
                                    try:
                                        formatted_rows.append(json.dumps(row, default=datetime_converter))
                                    except TypeError as e:
                                        logger.error(f"TypeError during json.dumps of row {row}: {e}")
                                        formatted_rows.append(str(row)) 
                                else:
                                    formatted_rows.append(str(row))
                            db_results_str = "\n".join(formatted_rows)
                    elif isinstance(db_output_raw, dict):
                        def datetime_converter(o):
                            if isinstance(o, datetime):
                                return o.isoformat()
                            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                        try:
                            db_results_str = json.dumps(db_output_raw, default=datetime_converter)
                        except TypeError as e:
                            logger.error(f"TypeError during json.dumps of dict {db_output_raw}: {e}")
                            db_results_str = str(db_output_raw)
                    else:
                        db_results_str = str(db_output_raw)
                    
                    error_in_results = False
                    if isinstance(db_output_raw, list) and db_output_raw and isinstance(db_output_raw[0], dict) and "error" in db_output_raw[0]:
                        error_in_results = True
                    elif isinstance(db_output_raw, dict) and "error" in db_output_raw:
                        error_in_results = True
                    
                    if error_in_results:
                         db_results_str = f"Database returned an error: {db_results_str}"
                         logger.error(f"Database operation returned an error based on 'error' key: {db_results_str}")

                elif not action_executed:
                    for chunk in llm_suggestion_raw_chunks:
                        yield chunk
                    return
            except json.JSONDecodeError as e:
                db_results_str = f"Could not parse LLM's database action suggestion (invalid JSON). Error: {e}. LLM Output: {llm_suggestion_raw[:200]}..."
                logger.error(db_results_str, exc_info=True)
            except Exception as e:
                db_results_str = f"An unexpected error occurred while processing LLM's database suggestion. Error: {e}. LLM Output: {llm_suggestion_raw[:200]}..."
                logger.error(db_results_str, exc_info=True)
            
            if db_results_str is not None:
                summary_generator = self.generate_answer(
                    user_query, 
                    context_documents=retrieved_context_str, 
                    chat_history_str=formatted_chat_history_str, 
                    db_results=db_results_str, 
                    current_mode=current_mode
                )
                yield from summary_generator
            else:
                yield "I couldn't perform a database action based on your request. Please try rephrasing."
                logger.warning("No database action was performed despite being in DB search mode.")
        else:
            answer_generator = self.generate_answer(
                user_query, 
                context_documents=retrieved_context_str, 
                chat_history_str=formatted_chat_history_str, 
                db_results=db_results_str, 
                current_mode=current_mode
            )
            yield from answer_generator

DEFAULT_CHAT_TITLE = "New Chat"

def switch_session(session_id):
    st.session_state.active_session_id = session_id
    st.session_state.chat_history = load_chat_history_util(st.session_state.active_session_id)
    st.session_state.current_mode = "Conversational 💬"

def delete_chat_session(session_id):
    delete_session_util(session_id, st.session_state.sessions_metadata)
    if st.session_state.active_session_id == session_id:
        sessions_meta = load_sessions_metadata_util()
        if sessions_meta:
            most_recent_session_id = None
            latest_timestamp = 0
            for sid, meta in sessions_meta.items():
                if isinstance(meta, dict) and meta.get('last_active', 0) > latest_timestamp:
                    latest_timestamp = meta['last_active']
                    most_recent_session_id = sid
            
            if most_recent_session_id:
                st.session_state.active_session_id = most_recent_session_id
                st.session_state.chat_history = load_chat_history_util(st.session_state.active_session_id)
                st.session_state.current_mode = "Conversational 💬"
                logger.info(f"Deleted active session, loaded most recent: {most_recent_session_id}")
            else:
                st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
                st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
                save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
                st.session_state.current_mode = "Conversational 💬"
                logger.info("Deleted last session, created a new one with initial greeting.")
        else:
            st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
            st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
            save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
            st.session_state.current_mode = "Conversational 💬"
            logger.info("No sessions found after deletion, created a new one with initial greeting.")
    st.rerun() 

def init_chatbot_core():
    if "chatbot_core" not in st.session_state or st.session_state.chatbot_core is None:
        with st.spinner("Initializing Chatbot Core (LLM, Embeddings, DB connections)... This may take a moment."):
            try:
                st.session_state.chatbot_core = ChatbotCore(
                    embedding_model_name=EMBEDDING_MODEL_NAME,
                    llm_model_name=LLM_MODEL_NAME,
                    ollama_base_url=OLLAMA_BASE_URL,
                    temperature=TEMPERATURE
                )
                st.success("Chatbot Core initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize Chatbot Core: {e}. Please check your Ollama server and model configuration.")
                st.session_state.chatbot_core = None
                logger.exception("Failed to initialize ChatbotCore")

st.set_page_config(page_title="Intelligent Cobot Assistant", layout="centered", initial_sidebar_state="expanded")

# --- Session State Initialization ---
if "chatbot_core" not in st.session_state:
    st.session_state.chatbot_core = None
if "sessions_metadata" not in st.session_state:
    st.session_state.sessions_metadata = load_sessions_metadata_util()
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Conversational 💬"
if "new_session_name" not in st.session_state:
    st.session_state.new_session_name = ""
if "current_session_loaded" not in st.session_state:
    st.session_state.current_session_loaded = None

# --- Load most recent session on startup if no session is active ---
if st.session_state.active_session_id is None:
    sessions_meta = load_sessions_metadata_util()
    if sessions_meta:
        most_recent_session_id = None
        latest_timestamp = 0
        for session_id, meta in sessions_meta.items():
            if isinstance(meta, dict) and meta.get('last_active', 0) > latest_timestamp:
                latest_timestamp = meta['last_active']
                most_recent_session_id = session_id
        
        if most_recent_session_id:
            st.session_state.active_session_id = most_recent_session_id
            st.session_state.chat_history = load_chat_history_util(st.session_state.active_session_id)
            st.session_state.current_mode = "Conversational 💬"
            st.session_state.current_session_loaded = most_recent_session_id
            logger.info(f"Loaded most recent session: {most_recent_session_id}")
        else:
            st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
            st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
            save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
            st.session_state.current_mode = "Conversational 💬"
            st.session_state.current_session_loaded = st.session_state.active_session_id
            logger.warning("No valid recent session found, created a new one with initial greeting.")
    else:
        st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
        st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
        save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        st.session_state.current_mode = "Conversational 💬"
        st.session_state.current_session_loaded = st.session_state.active_session_id
        logger.info("No sessions found, created a new one with initial greeting.")
# --- End Load most recent session on startup ---


init_chatbot_core()

with st.sidebar:
    try:
        st.image("wmg_logo.png", width=300)
    except Exception as e:
        st.error(f"Error loading logo: {e}")
    # --- New Chat button logic ---
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True, type="primary"):
        new_id = generate_session_id()
        st.session_state.active_session_id = new_id
        st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
        
        st.session_state.sessions_metadata[new_id] = {
            'id': new_id,
            'name': DEFAULT_CHAT_TITLE,
            'created_at': time.time(),
            'last_active': time.time(),
            'chat_history_file': f"chat_history_{new_id}.json"
        }
        
        st.session_state.current_session_loaded = new_id
        save_chat_history_util(new_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        logging.info(f"New chat started: {new_id}")
        st.rerun() 
    
    st.markdown("--- \n#### Chat History")
    if not isinstance(st.session_state.sessions_metadata, dict): st.session_state.sessions_metadata = {}
    try:
        sorted_sessions = sorted(st.session_state.sessions_metadata.items(), key=lambda item: item[1].get('last_active', 0) if isinstance(item[1], dict) else 0, reverse=True)
    except Exception: sorted_sessions = []
    
    history_container = st.container(height=200, border=False)
    with history_container:
        if not sorted_sessions: st.caption("No past chats yet.")
        for session_id, meta in sorted_sessions:
            if not isinstance(meta, dict): continue
            title = meta.get('name', session_id)
            display_title = (title[:20] + '...' if len(title) > 20 else title) if title else DEFAULT_CHAT_TITLE
            is_active = (session_id == st.session_state.active_session_id)
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if col1.button(f"**{display_title}**" if is_active else display_title, key=f"session_btn_{session_id}", use_container_width=True, help=title, disabled=is_active, on_click=lambda s_id=session_id: switch_session(s_id)):
                    pass
            with col2:
                if col2.button("🗑️", key=f"del_btn_{session_id}", help=f"Delete chat: {title}", type="secondary", use_container_width=True, on_click=lambda s_id=session_id: delete_chat_session(s_id)):
                    pass
    st.divider()
    available_modes = ["Conversational 💬", "Knowledge Base 📚", "Database Search 🗃️"]
    st.markdown("### ⚡ Mode Selection ⚡")
    default_mode_val = "Conversational 💬"
    if 'current_mode' not in st.session_state or st.session_state.current_mode not in available_modes: st.session_state.current_mode = default_mode_val
    try: current_mode_idx = available_modes.index(st.session_state.current_mode)
    except ValueError: current_mode_idx = 0; st.session_state.current_mode = available_modes[0]
    
    selected_mode_radio = st.radio("Choose interaction mode:", available_modes, key='mode_radio', index=current_mode_idx)
    if selected_mode_radio != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode_radio
        st.rerun()
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("💖 WMG SME Digital Team", unsafe_allow_html=True)

st.title("Intelligent Assistant")
st.caption("Your friendly AI assistant with a knowledge base and database query capabilities! 📚")

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        avatar = "🤖" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- Main Logic Loop for Chat Input ---
user_input_data = st.chat_input(
    "Let's chat or ask me anything!!!",
)

if user_input_data:
    active_session_id = st.session_state.active_session_id
    user_query_text = user_input_data
    display_message = user_query_text

    if display_message:
        st.session_state.chat_history.append({"role": "user", "content": display_message})

        with chat_container:
            chat_container.empty()
            for message in st.session_state.chat_history:
                avatar = "🤖" if message["role"] == "assistant" else None
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

    # --- Display Response with Streaming ---
    with st.spinner("🧠 Thinking..."):
        message_placeholder = st.empty()

        full_response_content = "" 
        final_display_response = ""

        if st.session_state.chatbot_core:
            try:
                response_stream_gen = st.session_state.chatbot_core.process_query(
                    user_query_text,
                    st.session_state.active_session_id,
                    st.session_state.chat_history,
                    st.session_state.current_mode,)

                for chunk in response_stream_gen:
                    full_response_content += chunk
                    message_placeholder.markdown(full_response_content + "▌")
                
                final_display_response = clean_llm_output(full_response_content)
                message_placeholder.markdown(final_display_response)

            except Exception as e:
                error_msg = f"An error occurred: {e}"
                logger.error(error_msg, exc_info=True)
                message_placeholder.markdown(error_msg)
                final_display_response = error_msg
        else:
            core_error_msg = "Chatbot core unavailable. Cannot process query."
            message_placeholder.markdown(core_error_msg)
            final_display_response = core_error_msg

        st.session_state.chat_history.append({"role": "assistant", "content": final_display_response})
        save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        
        if st.session_state.chatbot_core.history_vector_store and st.session_state.chatbot_core.embedding_model:
            try:
                user_turn_text = user_query_text
                ai_turn_text = final_display_response
                turn_docs = [
                    Document(page_content=user_turn_text, metadata={"role": "user", "session_id": st.session_state.active_session_id, "timestamp": time.time()}), 
                    Document(page_content=ai_turn_text, metadata={"role": "assistant", "session_id": st.session_state.active_session_id, "timestamp": time.time()})
                ]
                st.session_state.chatbot_core.history_vector_store.add_texts([doc.page_content for doc in turn_docs], metadatas=[doc.metadata for doc in turn_docs])
                if not save_faiss_store_util(st.session_state.chatbot_core.history_vector_store, HISTORY_INDEX_DIR): 
                    logger.warning("Failed to save history FAISS store.")
            except Exception as e: 
                logger.error(f"Error saving chat turn to history vector store: {e}", exc_info=True)

        st.rerun()
