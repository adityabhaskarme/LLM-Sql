# main.py - FastAPI Backend

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Dict, Any

import sqlite3
import os
import logging
import time
import re

from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv

# Hugging Face
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logger = logging.getLogger("chatbot")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)


# General log file
info_handler = logging.FileHandler("chatbot.log")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)


# Error log file
error_handler = logging.FileHandler("chatbot_error.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)


# Console logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


# Avoid duplicate handlers during reload
if not logger.handlers:
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Gemma-Powered Chatbot API",
    description=(
        "API for converting natural language queries "
        "into SQL using Google Gemma through Hugging Face"
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API SECURITY
# ============================================================

API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def get_api_key(
    api_key: str = Security(api_key_header)
):
    """
    Validate API key passed through X-API-Key header.
    """

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY not configured"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )

    return api_key


# ============================================================
# DATABASE CONNECTION
# ============================================================

@contextmanager
def get_db_connection():

    try:

        conn = sqlite3.connect(
            "customers.db",
            check_same_thread=False
        )

        conn.row_factory = sqlite3.Row

        try:
            yield conn

        finally:
            conn.close()

    except Exception as e:

        logger.error(
            f"Database connection failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Database connection failed"
        )


# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserQuery(BaseModel):

    query: str
    case_sensitive: bool = False


class QueryResponse(BaseModel):

    sql_query: str
    results: List[Dict[str, Any]]
    message: str
    execution_time: float


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

def get_hf_client():

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:

        raise ValueError(
            "HF_TOKEN not configured"
        )

    client = InferenceClient(
        api_key=hf_token,
        provider="auto"
    )

    return client


# ============================================================
# SQL VALIDATION
# ============================================================

def validate_sql_query(sql_query: str):

    # Remove trailing semicolon
    sql_query = sql_query.rstrip(";").strip()

    # Operations that the LLM is never allowed to execute
    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "replace",
        "attach",
        "detach"
    ]

    pattern = (
        r"\b("
        + "|".join(forbidden_keywords)
        + r")\b"
    )

    if re.search(
        pattern,
        sql_query,
        re.IGNORECASE
    ):

        raise ValueError(
            "Query contains forbidden SQL operation"
        )

    # Prevent multiple SQL statements
    if ";" in sql_query:

        raise ValueError(
            "Multiple queries are not allowed"
        )

    # Only SELECT / WITH queries
    if not re.match(
        r"^\s*select\s+",
        sql_query,
        re.IGNORECASE
    ):

        if not re.match(
            r"^\s*with\s+",
            sql_query,
            re.IGNORECASE
        ):

            raise ValueError(
                "Only SELECT queries are allowed"
            )


# ============================================================
# EXTRACT SQL FROM MODEL RESPONSE
# ============================================================

def extract_sql_from_response(
    response: str
) -> str:

    response = response.strip()

    # --------------------------------------------------------
    # ```sql code block
    # --------------------------------------------------------

    sql_code_block = re.search(
        r"```sql\s*(.*?)```",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if sql_code_block:

        return (
            sql_code_block
            .group(1)
            .strip()
            .rstrip(";")
        )

    # --------------------------------------------------------
    # Generic code block
    # --------------------------------------------------------

    generic_code_block = re.search(
        r"```\s*(.*?)```",
        response,
        re.DOTALL
    )

    if generic_code_block:

        possible_sql = (
            generic_code_block
            .group(1)
            .strip()
        )

        if re.match(
            r"^(select|with)\b",
            possible_sql,
            re.IGNORECASE
        ):

            return possible_sql.rstrip(";")

    # --------------------------------------------------------
    # Find SELECT statement inside response
    # --------------------------------------------------------

    sql_match = re.search(
        r"\b(SELECT|WITH)\b.*",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if sql_match:

        sql = sql_match.group(0).strip()

        return sql.rstrip(";")

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return response.rstrip(";").strip()


# ============================================================
# GENERATE SQL USING GEMMA
# ============================================================

def generate_sql_from_natural_language(
    query: str,
    case_sensitive: bool = False
) -> str:

    try:

        # ----------------------------------------------------
        # Hugging Face client
        # ----------------------------------------------------

        client = get_hf_client()

        # ----------------------------------------------------
        # Pre-process natural language
        # ----------------------------------------------------

        processed_query = query.lower()

        processed_query = re.sub(
            r"\bcustomer(s)?\b",
            "",
            processed_query
        )

        processed_query = re.sub(
            r"\bfrom location\b",
            "from",
            processed_query
        )

        processed_query = processed_query.strip()

        # ----------------------------------------------------
        # Matching instructions
        # ----------------------------------------------------

        if case_sensitive:

            matching_rule = """
Use case-sensitive matching.

Example:

WHERE location = 'Mumbai'
"""

        else:

            matching_rule = """
Use case-insensitive matching.

For text comparisons, prefer COLLATE NOCASE.

Example:

WHERE location = 'Mumbai' COLLATE NOCASE
"""

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are an expert SQLite query generator.

Your job is to convert the user's natural language request
into ONE valid SQLite SELECT query.

Return ONLY the SQL query.

Do not explain the query.
Do not use markdown.
Do not use ```sql blocks.
Do not add comments.
Do not add a semicolon.


DATABASE SCHEMA
===============

Table:

customers


Columns:

customer_id INTEGER PRIMARY KEY

name TEXT

gender TEXT

location TEXT


USER QUERY
==========

Original query:

"{query}"


Normalized query:

"{processed_query}"


MATCHING RULE
=============

{matching_rule}


IMPORTANT RULES
===============

1. Only generate SELECT statements.

2. Never generate:

INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
REPLACE


3. The only available table is:

customers


4. Available columns are:

customer_id
name
gender
location


5. Do not invent tables.

6. Do not invent columns.

7. "customer" and "customers" mean the customers table.

8. Gender-related phrases should use the gender column.

Examples:

male
man
men

should normally mean:

gender = 'Male'


female
woman
women

should normally mean:

gender = 'Female'


9. Location phrases should use the location column.

Examples:

"customers from Mumbai"

means:

location = 'Mumbai'


"customers in London"

means:

location = 'London'


10. Multiple locations connected by OR can use IN.

Example:

"customers in Mumbai or London"

becomes:

SELECT *
FROM customers
WHERE location IN ('Mumbai', 'London')


11. When multiple gender/location conditions are requested,
use parentheses correctly.

Example:

"female customers from Mumbai and male customers from Paris"

becomes:

SELECT *
FROM customers
WHERE
(
    gender = 'Female'
    AND location = 'Mumbai'
)
OR
(
    gender = 'Male'
    AND location = 'Paris'
)


12. If the user asks for all customers:

SELECT *
FROM customers


13. If the user asks for names only:

SELECT name
FROM customers


14. If the user asks "how many",
use COUNT(*).

Example:

"how many female customers"

becomes:

SELECT COUNT(*)
FROM customers
WHERE gender = 'Female'


15. Never include a trailing semicolon.


EXAMPLES
========


User:

show me all female customers from location Mumbai


SQL:

SELECT *
FROM customers
WHERE gender = 'Female'
AND location = 'Mumbai'


User:

show me all male customers from New York


SQL:

SELECT *
FROM customers
WHERE gender = 'Male'
AND location = 'New York'


User:

find customers in Mumbai or London


SQL:

SELECT *
FROM customers
WHERE location IN ('Mumbai', 'London')


User:

list female customers from Mumbai and male customers from Paris


SQL:

SELECT *
FROM customers
WHERE
(
    gender = 'Female'
    AND location = 'Mumbai'
)
OR
(
    gender = 'Male'
    AND location = 'Paris'
)


User:

show all customers


SQL:

SELECT *
FROM customers


User:

how many customers are from Mumbai


SQL:

SELECT COUNT(*)
FROM customers
WHERE location = 'Mumbai'


NOW GENERATE SQL
================

User:

"{query}"

SQL:
"""

        # ----------------------------------------------------
        # Call Gemma through Hugging Face
        # ----------------------------------------------------

        chat_completion = (
            client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=512
            )
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        llm_response = (
            chat_completion
            .choices[0]
            .message
            .content
        )

        if not llm_response:

            raise ValueError(
                "Gemma returned an empty response"
            )

        llm_response = llm_response.strip()

        logger.info(
            f"Gemma raw response: {llm_response}"
        )

        # ----------------------------------------------------
        # Extract SQL
        # ----------------------------------------------------

        sql_query = extract_sql_from_response(
            llm_response
        )

        logger.info(
            f"Extracted SQL: {sql_query}"
        )

        # ----------------------------------------------------
        # Validate SQL
        # ----------------------------------------------------

        validate_sql_query(
            sql_query
        )

        return sql_query

    except HTTPException:

        raise

    except Exception as e:

        logger.exception(
            "SQL generation using Gemma failed"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "SQL generation error: "
                + str(e)
            )
        )


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse
)
async def process_query(
    user_query: UserQuery,
    api_key: str = Depends(get_api_key)
):

    start_time = time.time()

    logger.info(
        f"Received query: {user_query.query}"
    )

    try:

        # ----------------------------------------------------
        # Generate SQL using Gemma
        # ----------------------------------------------------

        sql_query = (
            generate_sql_from_natural_language(
                user_query.query,
                user_query.case_sensitive
            )
        )

        logger.info(
            f"Generated SQL: {sql_query}"
        )

        # ----------------------------------------------------
        # Execute SQL
        # ----------------------------------------------------

        with get_db_connection() as conn:

            cursor = conn.cursor()

            try:

                cursor.execute(
                    sql_query
                )

                rows = cursor.fetchall()

                results = [
                    dict(row)
                    for row in rows
                ]

            except sqlite3.Error as e:

                logger.error(
                    f"SQL execution error: {str(e)}"
                )

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SQL execution error: "
                        + str(e)
                    )
                )

        # ----------------------------------------------------
        # Execution time
        # ----------------------------------------------------

        execution_time = (
            time.time()
            - start_time
        )

        logger.info(
            "Query completed successfully "
            f"in {execution_time:.2f}s"
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return QueryResponse(
            sql_query=sql_query,
            results=results,
            message=(
                f"Found {len(results)} results"
            ),
            execution_time=execution_time
        )

    except HTTPException as e:

        raise e

    except Exception as e:

        logger.exception(
            "Query processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health_check():

    try:

        with get_db_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM customers"
            )

            count = cursor.fetchone()[0]

        return {
            "status": "healthy",
            "llm": "google/gemma-3-27b-it",
            "provider": "Hugging Face",
            "database_records": count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:

        logger.exception(
            "Health check failed"
        )

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    if not os.path.exists(
        "customers.db"
    ):

        logger.info(
            "Creating customers database"
        )

        with sqlite3.connect(
            "customers.db"
        ) as conn:

            cursor = conn.cursor()

            # ------------------------------------------------
            # Create customers table
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE TABLE customers (
                    customer_id INTEGER
                        PRIMARY KEY AUTOINCREMENT,

                    name TEXT NOT NULL,

                    gender TEXT,

                    location TEXT
                )
                """
            )

            # ------------------------------------------------
            # Indexes
            # ------------------------------------------------

            cursor.execute(
                """
                CREATE INDEX
                idx_customers_location
                ON customers(location)
                """
            )

            cursor.execute(
                """
                CREATE INDEX
                idx_customers_gender
                ON customers(gender)
                """
            )

            # ------------------------------------------------
            # Sample data
            # ------------------------------------------------

            customers = [

                (
                    "John Doe",
                    "Male",
                    "New York"
                ),

                (
                    "Jane Smith",
                    "Female",
                    "Mumbai"
                ),

                (
                    "Alice Johnson",
                    "Female",
                    "London"
                ),

                (
                    "Bob Brown",
                    "Male",
                    "Mumbai"
                ),

                (
                    "Charlie Davis",
                    "Male",
                    "Paris"
                ),

                (
                    "Diana Evans",
                    "Female",
                    "Mumbai"
                ),

                (
                    "Eve Wilson",
                    "Female",
                    "Tokyo"
                )
            ]

            cursor.executemany(
                """
                INSERT INTO customers
                (
                    name,
                    gender,
                    location
                )
                VALUES (?, ?, ?)
                """,
                customers
            )

            conn.commit()

            logger.info(
                "Customers database created successfully"
            )


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup():

    init_db()

    logger.info(
        "Gemma chatbot application started"
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
