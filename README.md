# LLM-Sql
A chatbot for writing sql queries 
*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built

I built an **LLM-powered Natural Language to SQL API** using **Google Gemma through Hugging Face**.

The idea is simple: instead of requiring someone to know SQL, they can ask a database question using normal language.

For example:

```text
Show me all female customers from Mumbai
```

The application converts that request into:

```sql
SELECT *
FROM customers
WHERE gender = 'Female'
AND location = 'Mumbai'
```

I built this with a friend or teammate in mind who may need to retrieve information from a database but does not necessarily know SQL syntax.

The goal is to make database interaction easier by allowing users to communicate with structured data using the same language they use every day.

The overall flow is:

```text
Natural Language Question
        ↓
      FastAPI
        ↓
Gemma via Hugging Face
        ↓
   Generated SQL
        ↓
   SQL Validation
        ↓
      SQLite
        ↓
      Results
```

One important part of the project is that the generated SQL is **not executed blindly**.

Before executing a query, the backend validates it and only allows safe read operations.

Currently, operations such as:

```text
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
```

are rejected.

Only `SELECT` queries are allowed.

## Demo

The API accepts requests through:

```text
POST /query
```

Example:

```json
{
  "query": "show me all female customers from Mumbai"
}
```

Gemma generates SQL such as:

```sql
SELECT *
FROM customers
WHERE gender = 'Female'
AND location = 'Mumbai'
```

The SQL is validated and executed against the SQLite database.

A response looks similar to:

```json
{
  "sql_query": "SELECT * FROM customers WHERE gender = 'Female' AND location = 'Mumbai'",
  "results": [
    {
      "customer_id": 2,
      "name": "Jane Smith",
      "gender": "Female",
      "location": "Mumbai"
    },
    {
      "customer_id": 6,
      "name": "Diana Evans",
      "gender": "Female",
      "location": "Mumbai"
    }
  ],
  "message": "Found 2 results",
  "execution_time": 0.82
}
```

**Demo Link:**  
Add your deployed API or video demo here.

## Code

The backend is built with Python and FastAPI.

The main LLM call uses Hugging Face's inference client with Gemma:

```python
from huggingface_hub import InferenceClient

def get_hf_client():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not configured")

    return InferenceClient(
        api_key=hf_token,
        provider="auto"
    )
```

The natural-language query is then sent to Gemma:

```python
chat_completion = client.chat.completions.create(
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
```

The model's response is extracted and validated before anything reaches the database.

For example, I maintain a list of forbidden SQL operations:

```python
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
```

The API also checks that the generated statement starts with:

```sql
SELECT
```

or:

```sql
WITH
```

This creates a safety layer between the LLM and the database.

**GitHub Repository:**  
https://github.com/adityabhaskarme/LLM-Sql

## How I Built It

The project uses the following stack:

- **Google Gemma** — generates SQL from natural-language requests
- **Hugging Face Inference** — provides access to the Gemma model
- **FastAPI** — exposes the backend API
- **SQLite** — stores and queries the sample customer database
- **Pydantic** — validates API requests and responses
- **Python** — powers the complete backend

The database currently contains a simple `customers` table:

```text
customers

customer_id
name
gender
location
```

When a user submits a question, the application first gives Gemma information about the database schema.

For example:

```text
Table: customers

Columns:
- customer_id
- name
- gender
- location
```

The prompt then instructs Gemma to return only a valid SQLite query.

After receiving the response, the application:

```text
1. Extracts the generated SQL
2. Checks for forbidden operations
3. Rejects multiple SQL statements
4. Ensures it is a SELECT query
5. Executes the query against SQLite
6. Returns the results through FastAPI
```

This was also an interesting part of the project because using an LLM for SQL generation is not only about getting the model to generate correct SQL.

You also have to think about what happens when that generated output interacts with an actual database.

## Why Does Open Innovation Matter?

Open innovation made this project much easier to experiment with.

Using an open-weight model like **Gemma** gives developers more flexibility in how AI is integrated into applications.

Instead of designing the entire project around one closed API, I can build the architecture around the model itself and choose how I want to run or serve it.

For example, the same project architecture could eventually use:

```text
Hugging Face hosted inference

        or

Self-hosted Gemma

        or

Local inference
```

without redesigning the complete application.

Open models also make it easier to understand, experiment with, customize, and potentially fine-tune the model for a specific use case such as SQL generation.

For a project like this, I could eventually train or adapt the model specifically around:

- A company's database schema
- Internal terminology
- Complex SQL queries
- Domain-specific tables
- Query examples
- Database-specific syntax

That flexibility is one of the things I find most interesting about building with open AI models.

## My Agent Session

I haven't added an agent session yet.

If I continue developing this project using DevRelay, I can add the session here later.

## Prize Categories

I am entering this project for categories related to:

- Open-source AI
- Hugging Face / open-weight models
- AI-powered developer tools
- Hacktoberfest Weekend Challenge

## What's Next?

The current version works with a simple single-table database, but there are several improvements I want to explore.

Next, I want to add:

- Multiple-table support
- Automatic schema discovery
- SQL `JOIN` generation
- More advanced SQL parsing
- Stronger query validation
- Better handling of ambiguous questions
- Query explanation
- Support for larger databases
- Database-specific SQL generation
- Self-hosted Gemma inference

Eventually, I would also like to move from:

```text
User → LLM → SQL
```

toward a more intelligent database agent:

```text
User
 ↓
Gemma
 ↓
Understand Database Schema
 ↓
Generate SQL
 ↓
Validate Query
 ↓
Execute Query
 ↓
Inspect Result
 ↓
Explain Result to User
```

This project started as a simple Natural Language to SQL experiment, but it has been a useful way to explore how open LLMs like Gemma can become practical interfaces between humans and structured data.

Thanks for reading! 🚀
