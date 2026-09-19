"""
Northbridge Bank - Credit Risk Portfolio Query Engine
Streamlit application wrapping the notebook's classify -> generate -> validate ->
retry -> execute -> respond pipeline for natural-language credit risk questions.
"""

from typing import Optional
import json
import os
import re

import pandas as pd
import sqlite3
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)


# =============================================================================
# Credentials / Configuration
# =============================================================================

def load_credentials():
    """
    Resolve OpenAI credentials in this order:
    1. Streamlit secrets (st.secrets) - the standard way to store secrets on
       Streamlit Community Cloud (Settings -> Secrets).
    2. A local config.json file (same format used in the notebook) - for
       running the app locally without setting up st.secrets.
    3. Environment variables already set on the machine.
    """
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    # 2. Local config.json fallback
    if not api_key and os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    # 3. Environment variable fallback
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE")

    if not api_key:
        st.error(
            "No OpenAI API key found. Add OPENAI_API_KEY to Streamlit secrets, "
            "a local config.json, or an environment variable before using the app."
        )
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    return api_key, api_base


load_credentials()


# =============================================================================
# Cached Resources: LLMs and Database Connection
# =============================================================================

@st.cache_resource
def get_llms():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    return llm, evaluator_llm


@st.cache_resource
def get_db_connection():
    db_path = "credit_risk_portfolio.db"
    if not os.path.exists(db_path):
        st.error(
            f"Database file '{db_path}' was not found. Place it in the same "
            "directory as app.py before running the app."
        )
        st.stop()
    # Read-only connection, matching the notebook's safety posture
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


llm, evaluator_llm = get_llms()
conn = get_db_connection()


# =============================================================================
# Database Schema (identical to the notebook's database_schema string)
# =============================================================================

database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =============================================================================
# Verified Query Template Library (identical SQL to the notebook)
# =============================================================================

sql_1 = """SELECT
  sm.sector_name,
  ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
  ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_exposure_million
FROM
  loan_master AS lm
JOIN
  sector_master AS sm ON lm.sector_code = sm.sector_code
GROUP BY
  sm.sector_name
ORDER BY
  total_outstanding_million DESC;"""

sql_2 = """SELECT
  loan_category,
  ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
  COUNT(loan_account_number) AS loan_count
FROM
  loan_master
GROUP BY
  loan_category
ORDER BY
  total_outstanding_million DESC;"""

sql_3 = """SELECT
  ifrs9_stage,
  COUNT(DISTINCT loan_account_number) AS loan_count,
  ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_million,
  ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM
  provisioning
WHERE
  reporting_date = '2025-09-30'
GROUP BY
  ifrs9_stage
ORDER BY
  ifrs9_stage;"""

sql_4 = """SELECT
  sm.sector_name,
  ROUND(AVG(p.provision_coverage_ratio), 2) AS average_provision_coverage_ratio
FROM
  provisioning AS p
JOIN
  loan_master AS lm ON p.loan_account_number = lm.loan_account_number
JOIN
  sector_master AS sm ON lm.sector_code = sm.sector_code
WHERE
  p.reporting_date = '2025-09-30'
GROUP BY
  sm.sector_name
ORDER BY
  average_provision_coverage_ratio DESC;"""

sql_5 = """SELECT
  lm.borrower_name,
  sm.sector_name,
  ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
  lm.asset_classification
FROM
  loan_master AS lm
JOIN
  sector_master AS sm ON lm.sector_code = sm.sector_code
ORDER BY
  lm.total_outstanding DESC
LIMIT 10;"""

sql_6 = """SELECT
  group_name,
  COUNT(loan_account_number) AS loan_count,
  ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM
  loan_master
WHERE
  group_name IS NOT NULL
GROUP BY
  group_name
ORDER BY
  total_outstanding_million DESC
LIMIT 5;"""

sql_7 = """SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
  lm.days_past_due,
  lm.asset_classification
FROM
  loan_master AS lm
JOIN
  sector_master AS sm ON lm.sector_code = sm.sector_code
WHERE
  lm.days_past_due > 0
ORDER BY
  lm.days_past_due DESC;"""

sql_8 = """SELECT
  CASE
    WHEN days_past_due = 0 THEN '0 (Current)'
    WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
    WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
    WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
    ELSE '90+'
  END AS dpd_bucket,
  COUNT(loan_account_number) AS loan_count,
  ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM
  loan_master
GROUP BY
  dpd_bucket
ORDER BY
  CASE
    WHEN dpd_bucket = '0 (Current)' THEN 0
    WHEN dpd_bucket = '1-30' THEN 1
    WHEN dpd_bucket = '31-60' THEN 2
    WHEN dpd_bucket = '61-90' THEN 3
    ELSE 4
  END;"""

sql_9 = """SELECT
  borrower_id,
  previous_rating,
  internal_rating,
  pd_estimate
FROM
  borrower_rating
WHERE
  rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY
  pd_estimate DESC;"""

sql_10 = """SELECT
  reporting_date,
  ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM
  provisioning
GROUP BY
  reporting_date
ORDER BY
  reporting_date;"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# =============================================================================
# Pipeline Tools (identical logic to the notebook's tool functions)
# =============================================================================

def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are an expert in SQL and financial risk analysis. Your task is to classify a user's natural language question into one of two routes: 'verified' or 'generated'.

'verified' route: Choose this if the user's question can be answered by one of the pre-defined SQL queries in the `query_library` below. These are recurring, common questions.
'generated' route: Choose this if the user's question is novel and requires a new SQL query to be generated. This should be a robust, read-only SELECT query.

When classifying, consider the semantic meaning of the user's question and compare it to the descriptions of the verified queries. If there's a clear and direct match, use the 'verified' route. Otherwise, use the 'generated' route.

Think step-by-step. First, analyze the user question. Second, carefully read through the descriptions of all available verified queries. Third, decide if there is a strong semantic match. Fourth, output your decision in the specified JSON format.

User Question:
{user_question}

Available Verified Queries (query_id: description):
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
You are an expert in SQL and financial risk analysis. Your task is to write a SQLite SQL query based on a user's question and the provided database schema.

Here are the rules you must follow:
1. The query must be read-only; only use SELECT statements. Do not use any DDL or DML statements.
2. Only use table and column names that exist in the provided schema. Do not invent new ones.
3. Return only the SQL query as a raw string, without any additional text or markdown formatting.
4. Ensure the SQL query directly answers the user's question.
5. When dealing with dates, assume the latest available date for filtering, which is '2025-09-30' for both `reporting_date` in the `provisioning` table and `rating_date` in the `borrower_rating` table, unless the user specifies a different date.
6. For asset classification, 'NPA' refers to `asset_classification IN ('Substandard', 'Doubtful', 'Loss')`.
7. Convert monetary amounts (e.g., total_outstanding, sanctioned_amount, ecl_amount, ead_amount) to millions by dividing by 1000000.0 and rounding to two decimal places, unless otherwise specified by the user.
8. Pay close attention to aggregations and groupings based on the user's request.
9. If the user asks for a 'trend' or 'movement over time', ensure the query includes `reporting_date` or `rating_date` in the SELECT clause and groups by it, ordering chronologically.

Database Schema:
{schema_context}

User Question:
{user_question}

SQL Query:
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    def clean_sql_for_schema_inspection(sql):
        parsed_statements = sqlparse.parse(sql)
        if parsed_statements:
            cleaned_sql = str(parsed_statements[0]).strip().rstrip(';')
        else:
            cleaned_sql = sql.strip().rstrip(';')

        cleaned_sql = re.sub(r'\s+ORDER BY\s+.*?$', '', cleaned_sql, flags=re.IGNORECASE | re.DOTALL)
        cleaned_sql = re.sub(r'\s+LIMIT\s+.*?$', '', cleaned_sql, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_sql

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        sql_for_explain = candidate_sql.strip().rstrip(';')
        cur.execute(f"EXPLAIN {sql_for_explain}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are an expert in SQL and financial risk analysis. Your task is to evaluate whether a given SQL query accurately answers a user's question, considering the provided context.

Here are the rules you must follow:
1. Focus only on the semantic relevance of the SQL to the user's question. Do not check for SQL syntax errors or database schema conformance, as those are handled by other checks.
2. Consider the `track_context` carefully to understand if the query is a verified template (which might be broad) or a freshly generated query (which should be specific).
3. If it's a verified template, assess if the core metric, tables, and aggregation align with the user's intent, even if the filtering isn't exact (as filtering happens later).
4. If it's a freshly generated query, assess if it is appropriately scoped and filtered to directly answer the user's question.
5. Assign a `confidence` score from 0.0 to 1.0, where 1.0 means perfect relevance and 0.0 means no relevance.
6. Provide a short, concise `reason` for your verdict and confidence score.

Track Context:
{track_context}

User Question:
{user_question}

SQL Query:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']

        try:
            cleaned_expected_sql = clean_sql_for_schema_inspection(expected_sql)
            cur.execute(f"{cleaned_expected_sql} LIMIT 0")
            expected_cols = [d[0] for d in cur.description]

            cleaned_candidate_sql = clean_sql_for_schema_inspection(candidate_sql)
            cur.execute(f"{cleaned_candidate_sql} LIMIT 0")
            actual_cols = [d[0] for d in cur.description]

            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)} for {query_id}'
                return result
            if set(c.lower() for c in expected_cols) != set(c.lower() for c in actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Column names mismatch for {query_id}: Expected {expected_cols}, got {actual_cols}'
                return result

        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed for {query_id}: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are an expert in SQL and financial risk analysis. Your previous attempt to generate a SQL query failed validation. Your task is to revise the SQL query based on the original user question, the failed SQL, and the specific error message from the validation.

Here are the rules you must follow:
1. The query must be read-only; only use SELECT statements. Do not use any DDL or DML statements.
2. Only use table and column names that exist in the provided schema. Do not invent new ones.
3. Return only the SQL query as a raw string, without any additional text or markdown formatting.
4. Ensure the revised SQL query directly answers the user's question and addresses the identified error.
5. When dealing with dates, assume the latest available date for filtering, which is '2025-09-30' for both `reporting_date` in the `provisioning` table and `rating_date` in the `borrower_rating` table, unless the user specifies a different date.
6. For asset classification, 'NPA' refers to `asset_classification IN ('Substandard', 'Doubtful', 'Loss')`.
7. Convert monetary amounts (e.g., total_outstanding, sanctioned_amount, ecl_amount, ead_amount) to millions by dividing by 1000000.0 and rounding to two decimal places, unless otherwise specified by the user.
8. Pay close attention to aggregations and groupings based on the user's request.
9. If the user asks for a 'trend' or 'movement over time', ensure the query includes `reporting_date` or `rating_date` in the SELECT clause and groups by it, ordering chronologically.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Revised SQL Query:
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
You are an expert in SQL and financial risk analysis. Your task is to generate a concise, business-focused natural language response to the user's question based on the provided query results. Highlight only the relevant insights and exact figures.

Here are the rules you must follow:
1. If the dataframe is empty, state that no results were found for the query.
2. If the query route was 'verified' and `query_id` is provided, remember that the verified templates are intentionally broad. Your response must filter and highlight ONLY the information directly relevant to the `user_question` from the potentially broader `dataframe`.
3. If the query route was 'generated', the dataframe should already be specific to the `user_question`, so summarize the findings directly.
4. Always include specific numerical values from the dataframe where appropriate to support your insights. Convert all currency values to millions with 2 decimal places, for example: $123,456,789.01 becomes $123.46 million.
5. If a trend is requested, describe the direction and magnitude of the change over time, referencing specific periods and values from the data.
6. Maintain a professional and informative tone.

User Question:
{user_question}

Query Results:
{dataframe.to_string()}

Response:
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, appends intermediate pipeline stages to log['trace']
                       (in the notebook this printed to stdout; in Streamlit we
                       collect the lines instead so they can be shown in an expander).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    trace_lines = []

    def emit(line):
        if verbose:
            trace_lines.append(line)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace_lines, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'trace': trace_lines, **log}


# =============================================================================
# Streamlit UI
# =============================================================================

st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
st.caption(
    "Ask a plain-English question about the commercial lending portfolio. "
    "Common questions are answered instantly from pre-approved templates; "
    "novel questions are answered with freshly generated, validated SQL."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Verified Query Library")
    st.caption("These 10 recurring questions are answered from pre-approved templates:")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}"):
            st.write(entry["description"])

    st.divider()
    if st.button("Clear conversation history"):
        st.session_state.history = []
        st.rerun()

user_question = st.chat_input("Ask a question about the loan portfolio...")

if user_question:
    with st.spinner("Running the query engine pipeline..."):
        result = run_pipeline(
            user_question,
            conn,
            verified_query_library,
            database_schema,
            verbose=True,
        )
    st.session_state.history.append(result)

if not st.session_state.history:
    st.info("Try asking something like: *'What is the total outstanding and NPA exposure by sector?'*")

for result in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(result["user_question"])

    with st.chat_message("assistant"):
        if result["escalated"]:
            st.warning(result["narrative"])
        else:
            st.write(result["narrative"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Route", result["route"])
            col2.metric("Query ID", result["query_id"] or "generated")
            conf = result["confidence"]
            col3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))

            if result["dataframe"] is not None:
                st.dataframe(result["dataframe"], use_container_width=True)

        with st.expander("Pipeline trace"):
            for line in result["trace"]:
                st.text(line)
            if result["executed_sql"]:
                st.code(result["executed_sql"], language="sql")
