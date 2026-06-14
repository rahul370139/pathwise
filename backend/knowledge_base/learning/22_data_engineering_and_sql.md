# Data Engineering & SQL — Comprehensive Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland (4.0 GPA)  
**Focus:** Data Engineer / Data Scientist — SQL, Data Pipelines, Big Data, Cloud Warehouses  
**Core Skills:** SQL (Advanced), PySpark, MS SQL Server, MySQL, Snowflake, Redshift, Hadoop, Spark, Supabase, Neo4j, pgvector, ETL/ELT Pipelines  
**Document Scope:** End-to-end coverage of SQL fundamentals through advanced topics, data warehousing, big data, ETL/ELT, graph databases, and data modeling — theory, code, architecture, and interview strategy

---

## Table of Contents

1. [SQL Fundamentals](#1-sql-fundamentals)
2. [Advanced SQL](#2-advanced-sql)
3. [Database Design](#3-database-design)
4. [Data Warehouses](#4-data-warehouses)
5. [Big Data](#5-big-data)
6. [ETL/ELT Pipelines](#6-etlelt-pipelines)
7. [Graph Databases](#7-graph-databases)
8. [Data Modeling for ML](#8-data-modeling-for-ml)
9. [Common Interview Questions (10+)](#9-common-interview-questions)
10. [Key Takeaways](#10-key-takeaways)

---

# 1. SQL Fundamentals

---

SQL (Structured Query Language) is the lingua franca of data. Every data engineering, data science, and analytics role requires fluency. This section covers the core building blocks.

## 1.1 SELECT, WHERE, GROUP BY, HAVING, ORDER BY

### Logical Query Processing Order

SQL statements are written in one order but *executed* in another. Understanding this order is critical for debugging and optimization.

```
┌─────────────────────────────────────────────────────────┐
│           SQL Logical Execution Order                    │
│                                                         │
│   1. FROM / JOIN       ← Identify source tables         │
│   2. WHERE             ← Filter rows                    │
│   3. GROUP BY          ← Aggregate into groups          │
│   4. HAVING            ← Filter groups                  │
│   5. SELECT            ← Choose columns / expressions   │
│   6. DISTINCT          ← Remove duplicates              │
│   7. ORDER BY          ← Sort results                   │
│   8. LIMIT / OFFSET    ← Paginate                       │
└─────────────────────────────────────────────────────────┘
```

> **Interview tip:** When asked "Why can't I use an alias in WHERE?" the answer is execution order — WHERE runs before SELECT where aliases are defined.

### Core Clauses with Examples

Assume a table `orders`:

| order_id | customer_id | product | amount | order_date |
|----------|-------------|---------|--------|------------|
| 1 | 101 | Widget | 50.00 | 2024-01-15 |
| 2 | 102 | Gadget | 120.00 | 2024-01-16 |
| 3 | 101 | Widget | 50.00 | 2024-02-01 |
| 4 | 103 | Gizmo | 200.00 | 2024-02-10 |
| 5 | 102 | Widget | 50.00 | 2024-03-01 |

```sql
-- Basic SELECT with WHERE
SELECT customer_id, product, amount
FROM orders
WHERE amount > 50
ORDER BY amount DESC;
```

```sql
-- GROUP BY with HAVING: customers who spent more than $100 total
SELECT
    customer_id,
    COUNT(*)        AS order_count,
    SUM(amount)     AS total_spent,
    AVG(amount)     AS avg_order
FROM orders
GROUP BY customer_id
HAVING SUM(amount) > 100
ORDER BY total_spent DESC;
```

| Clause | Purpose | Operates On |
|--------|---------|-------------|
| `SELECT` | Columns/expressions to return | Rows after all filtering |
| `WHERE` | Row-level filter *before* aggregation | Individual rows |
| `GROUP BY` | Collapse rows into groups | Rows passing WHERE |
| `HAVING` | Group-level filter *after* aggregation | Aggregated groups |
| `ORDER BY` | Sort final result set | Final output rows |
| `LIMIT` | Restrict number of rows returned | Sorted output |

### WHERE vs HAVING

```sql
-- WHERE filters rows BEFORE grouping
SELECT department, AVG(salary) AS avg_sal
FROM employees
WHERE hire_date > '2020-01-01'     -- filters individual rows first
GROUP BY department
HAVING AVG(salary) > 70000;        -- then filters groups
```

> **Key distinction:** WHERE cannot reference aggregate functions. HAVING can — because it runs after GROUP BY.

---

## 1.2 JOINs

JOINs combine rows from two or more tables based on related columns. Mastery of JOINs is the single most tested SQL skill in interviews.

### Visual Reference

```
INNER JOIN          LEFT JOIN           RIGHT JOIN          FULL OUTER JOIN
┌───┬───┐           ┌───┬───┐           ┌───┬───┐           ┌───┬───┐
│ A │ B │           │ A │ B │           │ A │ B │           │ A │ B │
│   │███│           │███│███│           │███│███│           │███│███│
│   │███│           │███│███│           │███│███│           │███│███│
│   │   │           │███│   │           │   │███│           │███│███│
└───┴───┘           └───┴───┘           └───┴───┘           └───┴───┘
 Only matching       All from A          All from B          All from both
```

### Sample Tables

**employees:**
| emp_id | name | dept_id |
|--------|------|---------|
| 1 | Alice | 10 |
| 2 | Bob | 20 |
| 3 | Carol | NULL |

**departments:**
| dept_id | dept_name |
|---------|-----------|
| 10 | Engineering |
| 20 | Marketing |
| 30 | Finance |

### INNER JOIN

Returns only rows with matches in **both** tables.

```sql
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;
```

| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | Marketing |

> Carol (NULL dept_id) and Finance (no matching employee) are excluded.

### LEFT JOIN (LEFT OUTER JOIN)

Returns **all rows from the left table** and matched rows from the right. NULLs fill unmatched right-side columns.

```sql
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;
```

| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | Marketing |
| Carol | NULL |

### RIGHT JOIN (RIGHT OUTER JOIN)

Returns **all rows from the right table** and matched rows from the left.

```sql
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;
```

| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | Marketing |
| NULL | Finance |

### FULL OUTER JOIN

Returns **all rows from both tables**. NULLs where there is no match.

```sql
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;
```

| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | Marketing |
| Carol | NULL |
| NULL | Finance |

> **Note:** MySQL does not support FULL OUTER JOIN natively. Emulate it with `LEFT JOIN UNION RIGHT JOIN`.

### CROSS JOIN

Returns the **Cartesian product** — every row of A paired with every row of B.

```sql
-- Every employee paired with every department (3 × 3 = 9 rows)
SELECT e.name, d.dept_name
FROM employees e
CROSS JOIN departments d;
```

**Use cases:** Generating date/dimension combinations, test data generation, pivot helper tables.

### SELF JOIN

A table joined to **itself** — useful for hierarchical or comparison queries.

```sql
-- Find employees and their managers (same table)
SELECT
    e.name       AS employee,
    m.name       AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;
```

```sql
-- Find all pairs of employees in the same department
SELECT
    a.name AS employee_1,
    b.name AS employee_2,
    a.dept_id
FROM employees a
JOIN employees b
    ON a.dept_id = b.dept_id
   AND a.emp_id < b.emp_id;   -- avoid duplicates and self-pairing
```

### JOIN Performance Tips

| Tip | Why |
|-----|-----|
| Index the JOIN columns | B-tree lookup vs. full table scan |
| Filter early with WHERE | Reduce rows before joining |
| Prefer explicit JOIN syntax | `FROM a, b WHERE a.id = b.id` is legacy and error-prone |
| Avoid SELECT * in JOINs | Reduces I/O and memory |
| Use EXPLAIN to verify | Check if the optimizer chooses the right join algorithm (nested loop, hash, merge) |

---

## 1.3 Subqueries, CTEs, and Window Functions

### Subqueries

A query nested inside another query. Can appear in SELECT, FROM, or WHERE.

```sql
-- Scalar subquery in WHERE: employees earning above average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated subquery: employees earning more than their dept average
SELECT e.name, e.salary, e.dept_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e.dept_id    -- references outer query
);

-- Subquery in FROM (derived table)
SELECT dept_id, avg_sal
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) dept_avgs
WHERE avg_sal > 80000;
```

### CTEs (Common Table Expressions)

CTEs use the `WITH` keyword and improve readability, especially for multi-step logic. They are **not** materialized by default in most databases (the optimizer can inline them).

```sql
-- Named, reusable intermediate result
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_sales
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
ranked_months AS (
    SELECT
        month,
        total_sales,
        RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
    FROM monthly_sales
)
SELECT month, total_sales, sales_rank
FROM ranked_months
WHERE sales_rank <= 3;
```

> **CTE vs Subquery:** CTEs are more readable and can be referenced multiple times. Subqueries are fine for simple, one-off use.

### Window Functions

Window functions perform calculations across a set of rows *related to the current row* without collapsing them (unlike GROUP BY).

**Syntax:**

```sql
FUNCTION_NAME(expression) OVER (
    [PARTITION BY column(s)]
    [ORDER BY column(s)]
    [ROWS/RANGE BETWEEN ... AND ...]
)
```

```
┌──────────────────────────────────────────────────────────────────┐
│                    Window Function Anatomy                        │
│                                                                  │
│   SUM(amount) OVER (PARTITION BY dept ORDER BY date              │
│                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)    │
│   ─────────── ───── ────────────── ────────────                  │
│    Function   OVER   Partitioning   Frame/Ordering               │
│                                                                  │
│   • Function: What to compute (SUM, ROW_NUMBER, RANK, etc.)     │
│   • PARTITION BY: Split rows into groups (like GROUP BY, no      │
│     collapse)                                                    │
│   • ORDER BY: Sort within each partition                         │
│   • Frame: Which rows relative to current row to include         │
└──────────────────────────────────────────────────────────────────┘
```

#### ROW_NUMBER, RANK, DENSE_RANK

```sql
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    RANK()       OVER (ORDER BY salary DESC) AS rank_val,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank_val
FROM employees;
```

| name | salary | row_num | rank_val | dense_rank_val |
|------|--------|---------|----------|----------------|
| Eve | 120000 | 1 | 1 | 1 |
| Dan | 110000 | 2 | 2 | 2 |
| Carol | 110000 | 3 | 2 | 2 |
| Bob | 90000 | 4 | 4 | 3 |
| Alice | 80000 | 5 | 5 | 4 |

> - **ROW_NUMBER:** Always unique, breaks ties arbitrarily
> - **RANK:** Ties get the same rank, next rank is *skipped* (1, 2, 2, **4**)
> - **DENSE_RANK:** Ties get the same rank, next rank is *not* skipped (1, 2, 2, **3**)

#### LEAD and LAG

Access data from a preceding or following row.

```sql
SELECT
    order_date,
    amount,
    LAG(amount, 1)  OVER (ORDER BY order_date) AS prev_amount,
    LEAD(amount, 1) OVER (ORDER BY order_date) AS next_amount,
    amount - LAG(amount, 1) OVER (ORDER BY order_date) AS change
FROM orders;
```

| order_date | amount | prev_amount | next_amount | change |
|------------|--------|-------------|-------------|--------|
| 2024-01-15 | 50 | NULL | 120 | NULL |
| 2024-01-16 | 120 | 50 | 50 | 70 |
| 2024-02-01 | 50 | 120 | 200 | -70 |
| 2024-02-10 | 200 | 50 | 50 | 150 |
| 2024-03-01 | 50 | 200 | NULL | -150 |

#### SUM OVER (Running Total)

```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total,
    SUM(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3_sum
FROM orders;
```

#### NTILE

Distributes rows into N approximately equal buckets.

```sql
-- Split employees into salary quartiles
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS salary_quartile
FROM employees;
```

---

## 1.4 Set Operations: UNION, INTERSECT, EXCEPT

```
┌─────────────┬──────────────────────────────────────────────────┐
│ Operation   │ Returns                                          │
├─────────────┼──────────────────────────────────────────────────┤
│ UNION       │ All distinct rows from both queries              │
│ UNION ALL   │ All rows (including duplicates) — faster         │
│ INTERSECT   │ Only rows present in both queries                │
│ EXCEPT      │ Rows in the first query but NOT in the second    │
│ (MINUS)     │ (Oracle uses MINUS instead of EXCEPT)            │
└─────────────┴──────────────────────────────────────────────────┘
```

```sql
-- Active customers from 2023 who are NOT active in 2024
SELECT customer_id FROM orders WHERE YEAR(order_date) = 2023
EXCEPT
SELECT customer_id FROM orders WHERE YEAR(order_date) = 2024;

-- Customers present in both years (retention)
SELECT customer_id FROM orders WHERE YEAR(order_date) = 2023
INTERSECT
SELECT customer_id FROM orders WHERE YEAR(order_date) = 2024;
```

> **Performance tip:** Always prefer `UNION ALL` over `UNION` unless you need deduplication — `UNION` forces a sort/distinct operation.

---

## 1.5 DML: INSERT, UPDATE, DELETE, MERGE/UPSERT

### INSERT

```sql
-- Single row
INSERT INTO employees (name, dept_id, salary)
VALUES ('Frank', 10, 95000);

-- Multiple rows
INSERT INTO employees (name, dept_id, salary)
VALUES
    ('Grace', 20, 88000),
    ('Hank',  30, 76000);

-- Insert from SELECT
INSERT INTO employee_archive
SELECT * FROM employees WHERE termination_date IS NOT NULL;
```

### UPDATE

```sql
-- Update with condition
UPDATE employees
SET salary = salary * 1.10
WHERE dept_id = 10 AND hire_date < '2022-01-01';

-- Update from another table (SQL Server / PostgreSQL syntax)
UPDATE e
SET e.dept_name = d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;
```

### DELETE

```sql
-- Delete specific rows
DELETE FROM orders
WHERE order_date < '2020-01-01';

-- Delete with subquery
DELETE FROM employees
WHERE dept_id IN (
    SELECT dept_id FROM departments WHERE is_active = FALSE
);
```

> **TRUNCATE vs DELETE:** TRUNCATE removes all rows without logging individual deletions — much faster but cannot be rolled back in some engines and does not fire triggers.

### MERGE / UPSERT

`MERGE` (SQL:2003 standard) performs INSERT, UPDATE, or DELETE in a single statement — essential for ETL.

```sql
-- SQL Server / Snowflake MERGE
MERGE INTO target_table t
USING source_table s
ON t.id = s.id
WHEN MATCHED AND s.updated_at > t.updated_at THEN
    UPDATE SET
        t.name = s.name,
        t.amount = s.amount,
        t.updated_at = s.updated_at
WHEN NOT MATCHED THEN
    INSERT (id, name, amount, updated_at)
    VALUES (s.id, s.name, s.amount, s.updated_at)
WHEN NOT MATCHED BY SOURCE THEN
    DELETE;
```

```sql
-- PostgreSQL UPSERT (INSERT ... ON CONFLICT)
INSERT INTO products (id, name, price)
VALUES (1, 'Widget', 29.99)
ON CONFLICT (id)
DO UPDATE SET
    name = EXCLUDED.name,
    price = EXCLUDED.price;

-- MySQL UPSERT
INSERT INTO products (id, name, price)
VALUES (1, 'Widget', 29.99)
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    price = VALUES(price);
```

---

# 2. Advanced SQL

---

## 2.1 Window Functions Deep Dive

### Frame Specifications

The window frame defines which rows relative to the current row participate in the calculation.

```
Frame Types:
  ROWS   — Physical row offsets (exact row counts)
  RANGE  — Logical value ranges (based on ORDER BY value)
  GROUPS — Group-based offsets (SQL:2011, PostgreSQL 11+)

Common Frames:
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW     ← cumulative
  ROWS BETWEEN 3 PRECEDING AND CURRENT ROW             ← rolling 4 rows
  ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING              ← centered 3 rows
  ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING      ← reverse cumulative
```

```sql
-- 7-day moving average of daily revenue
SELECT
    sale_date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d,
    -- Percentage of total
    revenue * 100.0 / SUM(revenue) OVER () AS pct_of_total
FROM daily_sales;
```

### Advanced Ranking: Top-N Per Group

```sql
-- Top 3 highest-paid employees per department
WITH ranked AS (
    SELECT
        name,
        department,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department
            ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT name, department, salary
FROM ranked
WHERE rn <= 3;
```

### FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
SELECT
    name,
    department,
    salary,
    FIRST_VALUE(name) OVER (
        PARTITION BY department ORDER BY salary DESC
    ) AS highest_paid,
    LAST_VALUE(name) OVER (
        PARTITION BY department ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees;
```

> **Gotcha:** `LAST_VALUE` uses a default frame of `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, so you must override it to see the true last value.

---

## 2.2 Recursive CTEs

Recursive CTEs traverse hierarchical or graph-like data (org charts, BOMs, category trees).

```sql
-- Employee hierarchy: find all reports under a manager
WITH RECURSIVE org_chart AS (
    -- Anchor: start with the CEO
    SELECT emp_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: join children to parents
    SELECT e.emp_id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT
    REPEAT('  ', level - 1) || name AS org_tree,
    level
FROM org_chart
ORDER BY level, name;
```

```sql
-- Generate a date series (PostgreSQL)
WITH RECURSIVE date_series AS (
    SELECT DATE '2024-01-01' AS dt
    UNION ALL
    SELECT dt + INTERVAL '1 day'
    FROM date_series
    WHERE dt < DATE '2024-12-31'
)
SELECT dt FROM date_series;
```

> **Safety:** Always include a termination condition and/or set `MAXRECURSION` (SQL Server) to prevent infinite loops.

---

## 2.3 Pivot / Unpivot

### Pivot (Rows → Columns)

```sql
-- Standard SQL (using conditional aggregation — works everywhere)
SELECT
    product,
    SUM(CASE WHEN quarter = 'Q1' THEN revenue ELSE 0 END) AS Q1,
    SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END) AS Q2,
    SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) AS Q3,
    SUM(CASE WHEN quarter = 'Q4' THEN revenue ELSE 0 END) AS Q4
FROM sales
GROUP BY product;

-- SQL Server PIVOT syntax
SELECT product, [Q1], [Q2], [Q3], [Q4]
FROM (
    SELECT product, quarter, revenue
    FROM sales
) src
PIVOT (
    SUM(revenue) FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
) pvt;
```

### Unpivot (Columns → Rows)

```sql
-- SQL Server UNPIVOT
SELECT product, quarter, revenue
FROM quarterly_sales
UNPIVOT (
    revenue FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
) unpvt;

-- Standard SQL alternative using UNION ALL
SELECT product, 'Q1' AS quarter, Q1 AS revenue FROM quarterly_sales
UNION ALL
SELECT product, 'Q2', Q2 FROM quarterly_sales
UNION ALL
SELECT product, 'Q3', Q3 FROM quarterly_sales
UNION ALL
SELECT product, 'Q4', Q4 FROM quarterly_sales;
```

---

## 2.4 JSON Handling in SQL

Modern databases natively support JSON, bridging the relational/document gap.

```sql
-- PostgreSQL: JSONB operators
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    payload JSONB NOT NULL
);

INSERT INTO events (payload) VALUES
('{"user": "rahul", "action": "login", "meta": {"ip": "10.0.0.1", "device": "mobile"}}');

-- Extract values
SELECT
    payload->>'user'                  AS username,      -- text
    payload->'meta'->>'ip'            AS ip_address,    -- nested text
    payload @> '{"action": "login"}'  AS is_login,      -- containment check
    jsonb_array_length(payload->'tags') AS tag_count     -- array length
FROM events;

-- Query with JSON conditions
SELECT * FROM events
WHERE payload->>'action' = 'login'
  AND payload->'meta'->>'device' = 'mobile';
```

```sql
-- Snowflake: VARIANT type
SELECT
    raw_data:user::STRING           AS username,
    raw_data:meta.ip::STRING        AS ip_address,
    raw_data:scores[0]::NUMBER      AS first_score
FROM events_raw;

-- Flatten nested arrays
SELECT
    e.id,
    f.value::STRING AS tag
FROM events_raw e,
LATERAL FLATTEN(input => e.raw_data:tags) f;
```

```sql
-- MySQL: JSON functions
SELECT
    JSON_EXTRACT(payload, '$.user')         AS username,
    JSON_UNQUOTE(JSON_EXTRACT(payload, '$.meta.ip')) AS ip,
    payload->'$.action'                     AS action
FROM events;
```

---

## 2.5 Indexing

Indexes are the primary lever for query performance. Understanding index types and when to use them is essential.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Index Types                                    │
├──────────────┬───────────────────────────────────────────────────────┤
│ B-tree       │ Default. Balanced tree structure. Best for equality   │
│              │ (=) and range (<, >, BETWEEN) queries. Supports       │
│              │ ORDER BY. O(log n) lookups.                           │
├──────────────┼───────────────────────────────────────────────────────┤
│ Hash         │ Equality-only lookups (=). O(1) average. Cannot      │
│              │ support range queries or sorting. Used in memory-     │
│              │ optimized tables.                                     │
├──────────────┼───────────────────────────────────────────────────────┤
│ GIN          │ Generalized Inverted Index. Best for full-text        │
│ (PostgreSQL) │ search, JSONB containment (@>), array overlap (&&).  │
│              │ Stores a mapping from value → rows.                   │
├──────────────┼───────────────────────────────────────────────────────┤
│ GiST         │ Generalized Search Tree. Spatial/geometric data       │
│ (PostgreSQL) │ (PostGIS), range types, nearest-neighbor. Lossy       │
│              │ but supports complex operators.                        │
├──────────────┼───────────────────────────────────────────────────────┤
│ BRIN         │ Block Range Index. Summarizes min/max per block.      │
│ (PostgreSQL) │ Ideal for large, naturally ordered tables (time-      │
│              │ series). Very compact.                                 │
├──────────────┼───────────────────────────────────────────────────────┤
│ Bitmap       │ (Oracle, Redshift) Best for low-cardinality columns.  │
│              │ Combines multiple indexes efficiently with AND/OR.     │
└──────────────┴───────────────────────────────────────────────────────┘
```

```sql
-- Create indexes
CREATE INDEX idx_orders_date ON orders(order_date);            -- B-tree (default)
CREATE INDEX idx_orders_composite ON orders(customer_id, order_date);  -- composite
CREATE INDEX idx_events_payload ON events USING GIN (payload); -- GIN for JSONB

-- Partial index (only index rows matching a condition)
CREATE INDEX idx_active_users ON users(email) WHERE is_active = TRUE;

-- Expression index
CREATE INDEX idx_lower_email ON users(LOWER(email));

-- pgvector: HNSW index for similarity search
CREATE INDEX idx_embeddings ON documents USING hnsw (embedding vector_cosine_ops);
```

### Composite Index Column Ordering

```
┌─────────────────────────────────────────────────────────┐
│   INDEX (a, b, c) — Leftmost Prefix Rule                │
│                                                         │
│   Can satisfy queries on:                               │
│     ✓ WHERE a = ?                                       │
│     ✓ WHERE a = ? AND b = ?                             │
│     ✓ WHERE a = ? AND b = ? AND c = ?                   │
│     ✓ WHERE a = ? AND b > ?                             │
│     ✗ WHERE b = ?          ← skips "a", can't use index │
│     ✗ WHERE c = ?          ← skips "a" and "b"          │
│     ✗ WHERE b = ? AND c = ?                             │
│                                                         │
│   Rule: Equality columns first, then range, then sort   │
└─────────────────────────────────────────────────────────┘
```

---

## 2.6 Query Optimization

### EXPLAIN / EXPLAIN ANALYZE

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 80000;
```

**Key things to look for in query plans:**

| Indicator | Meaning |
|-----------|---------|
| Seq Scan | Full table scan — consider adding an index |
| Index Scan | Using an index — good for selective queries |
| Index Only Scan | Covered query — all needed data is in the index |
| Hash Join | Building hash table — good for equality joins on large tables |
| Nested Loop | Row-by-row join — good for small tables or indexed lookups |
| Merge Join | Both sides sorted — good for large pre-sorted datasets |
| Sort | Explicit sort operation — check if an index can avoid it |
| actual time | Real execution time (only with ANALYZE) |
| rows | Estimated vs actual row counts — large gaps indicate stale statistics |

### Common Optimization Strategies

```
┌────────────────────────────────────────────────────────────────┐
│              Query Optimization Checklist                       │
├────────────────────────────────────────────────────────────────┤
│ 1. Add appropriate indexes (check WHERE, JOIN, ORDER BY)       │
│ 2. Avoid SELECT * — only fetch needed columns                  │
│ 3. Use covering indexes to avoid table lookups                 │
│ 4. Rewrite correlated subqueries as JOINs where possible       │
│ 5. Use EXISTS instead of IN for large subqueries               │
│ 6. Avoid functions on indexed columns in WHERE                 │
│    ✗ WHERE YEAR(date_col) = 2024                               │
│    ✓ WHERE date_col >= '2024-01-01' AND date_col < '2025-01-01'│
│ 7. Update statistics (ANALYZE / UPDATE STATISTICS)             │
│ 8. Partition large tables                                      │
│ 9. Materialize expensive CTEs if reused                        │
│10. Use LIMIT for exploratory queries                           │
└────────────────────────────────────────────────────────────────┘
```

---

## 2.7 Partitioning and Sharding

### Partitioning (Single Database)

Partitioning divides a large table into smaller, more manageable pieces — all within the same database.

```
┌────────────────────────────────────────────────────────────┐
│              Partitioning Strategies                         │
├───────────────┬────────────────────────────────────────────┤
│ Range         │ By value ranges (dates, IDs)                │
│               │ e.g., one partition per month               │
├───────────────┼────────────────────────────────────────────┤
│ List          │ By discrete values (region, status)         │
│               │ e.g., one partition per country             │
├───────────────┼────────────────────────────────────────────┤
│ Hash          │ Distribute evenly by hash of column         │
│               │ e.g., user_id % 8 → 8 partitions           │
└───────────────┴────────────────────────────────────────────┘
```

```sql
-- PostgreSQL: range partitioning by date
CREATE TABLE events (
    id         BIGSERIAL,
    event_date DATE NOT NULL,
    payload    JSONB
) PARTITION BY RANGE (event_date);

CREATE TABLE events_2024_q1 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE events_2024_q2 PARTITION OF events
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Queries with event_date in WHERE automatically prune partitions
SELECT * FROM events WHERE event_date = '2024-02-15';
-- Only scans events_2024_q1 — partition pruning
```

### Sharding (Multiple Databases/Servers)

Sharding distributes data across multiple database instances (horizontal scaling).

```
┌─────────────────────────────────────────────────────────────┐
│                    Sharding                                   │
│                                                              │
│   Application / Router Layer                                 │
│      │                                                       │
│      ├── Shard 1 (user_id % 4 == 0) → DB Server A          │
│      ├── Shard 2 (user_id % 4 == 1) → DB Server B          │
│      ├── Shard 3 (user_id % 4 == 2) → DB Server C          │
│      └── Shard 4 (user_id % 4 == 3) → DB Server D          │
│                                                              │
│   Challenges:                                                │
│   • Cross-shard queries are expensive                        │
│   • Joins across shards require scatter-gather               │
│   • Rebalancing when adding shards is complex                │
│   • Consistent hashing helps minimize data movement          │
└─────────────────────────────────────────────────────────────┘
```

| Aspect | Partitioning | Sharding |
|--------|-------------|----------|
| Scope | Single database | Multiple databases |
| Managed by | Database engine | Application or middleware |
| Scaling | Vertical (same server) | Horizontal (multiple servers) |
| Complexity | Low | High |
| Cross-shard queries | N/A (same DB) | Expensive |

---

# 3. Database Design

---

## 3.1 Normalization

Normalization reduces data redundancy and prevents anomalies (insert, update, delete anomalies).

### Normal Forms

```
┌─────────┬────────────────────────────────────────────────────────────┐
│ Form    │ Rule                                                       │
├─────────┼────────────────────────────────────────────────────────────┤
│ 1NF     │ • All columns contain atomic (indivisible) values          │
│         │ • Each row is unique (has a primary key)                    │
│         │ • No repeating groups or arrays in a single column          │
│         │                                                            │
│         │ ✗ phone: "555-1234, 555-5678"                              │
│         │ ✓ Separate phone table with one row per number              │
├─────────┼────────────────────────────────────────────────────────────┤
│ 2NF     │ • Must be in 1NF                                           │
│         │ • No partial dependency: non-key columns depend on the      │
│         │   WHOLE primary key, not just part of it                    │
│         │                                                            │
│         │ ✗ PK=(student_id, course_id), student_name depends only    │
│         │   on student_id                                            │
│         │ ✓ Move student_name to a separate students table            │
├─────────┼────────────────────────────────────────────────────────────┤
│ 3NF     │ • Must be in 2NF                                           │
│         │ • No transitive dependency: non-key columns depend ONLY     │
│         │   on the primary key, not on other non-key columns          │
│         │                                                            │
│         │ ✗ emp(emp_id, dept_id, dept_name) — dept_name depends      │
│         │   on dept_id, not emp_id                                   │
│         │ ✓ Move dept_name to departments table                       │
├─────────┼────────────────────────────────────────────────────────────┤
│ BCNF    │ • Must be in 3NF                                           │
│ (Boyce- │ • Every determinant is a candidate key                     │
│  Codd)  │ • Stricter than 3NF for tables with overlapping            │
│         │   candidate keys                                           │
└─────────┴────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Most production OLTP systems aim for 3NF. BCNF is important conceptually but the distinction from 3NF matters only when there are overlapping composite candidate keys.

---

## 3.2 Denormalization for Analytics

While normalization reduces redundancy, analytics queries benefit from denormalized (pre-joined) data to reduce expensive JOINs.

| Normalization | Denormalization |
|--------------|-----------------|
| Reduces redundancy | Introduces controlled redundancy |
| Fewer anomalies | Faster read queries |
| More JOINs needed | Fewer JOINs |
| Best for OLTP | Best for OLAP / analytics |
| Smaller storage | Larger storage |
| Write-optimized | Read-optimized |

---

## 3.3 Star Schema and Snowflake Schema

### Star Schema

```
                    ┌─────────────┐
                    │ dim_date    │
                    │─────────────│
                    │ date_key    │
                    │ date        │
                    │ month       │
                    │ quarter     │
                    │ year        │
                    └──────┬──────┘
                           │
┌─────────────┐    ┌──────┴──────┐    ┌─────────────┐
│ dim_product │    │ fact_sales  │    │ dim_store   │
│─────────────│    │─────────────│    │─────────────│
│ product_key │◄───│ date_key    │───►│ store_key   │
│ name        │    │ product_key │    │ store_name  │
│ category    │    │ store_key   │    │ city        │
│ brand       │    │ customer_key│    │ state       │
└─────────────┘    │ quantity    │    └─────────────┘
                   │ revenue     │
                   │ discount    │    ┌─────────────┐
                   └──────┬──────┘    │ dim_customer│
                          │           │─────────────│
                          └──────────►│ customer_key│
                                      │ name        │
                                      │ segment     │
                                      └─────────────┘
```

**Characteristics:**
- **One fact table** surrounded by **dimension tables**
- Dimensions are denormalized (one level of JOINs)
- Simple queries, fast aggregations
- Most common schema in data warehouses

### Snowflake Schema

```
┌──────────────┐    ┌───────────────┐
│ dim_category │    │ dim_brand     │
│──────────────│    │───────────────│
│ category_key │◄───│ brand_key     │
│ category_name│    │ brand_name    │
└──────────────┘    │ category_key  │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐    ┌──────────────┐
                    │ dim_product   │    │ fact_sales   │
                    │───────────────│    │──────────────│
                    │ product_key   │◄───│ product_key  │
                    │ product_name  │    │ date_key     │
                    │ brand_key     │    │ revenue      │
                    └───────────────┘    └──────────────┘
```

**Characteristics:**
- Dimensions are **normalized** (multiple levels of JOINs)
- Less storage redundancy
- More complex queries
- Slower query performance than star schema

### Star vs Snowflake Comparison

| Aspect | Star Schema | Snowflake Schema |
|--------|------------|------------------|
| Dimension tables | Denormalized | Normalized |
| Query complexity | Simpler (fewer JOINs) | More complex |
| Query performance | Faster | Slower |
| Storage | More redundancy | Less redundancy |
| ETL complexity | Simpler loads | More complex |
| Use case | Most warehouses, BI | Very large dimensions, strict normalization needs |

---

## 3.4 OLTP vs OLAP

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect              │ OLTP                 │ OLAP                 │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Purpose             │ Day-to-day operations│ Analytics & reporting│
│ Queries             │ Short, simple CRUD   │ Complex aggregations │
│ Users               │ Many (thousands)     │ Few (analysts)       │
│ Data freshness      │ Real-time            │ Periodic refresh     │
│ Schema              │ Normalized (3NF)     │ Denormalized (star)  │
│ Storage             │ Row-oriented         │ Column-oriented      │
│ Volume per query    │ Few rows             │ Millions of rows     │
│ Optimization        │ Write-optimized      │ Read-optimized       │
│ Examples            │ MySQL, PostgreSQL,   │ Snowflake, Redshift, │
│                     │ MS SQL Server        │ BigQuery, ClickHouse │
│ Concurrency         │ High                 │ Lower                │
│ Transaction support │ Full ACID            │ Often relaxed        │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 3.5 ACID Properties

```
┌───────────────┬───────────────────────────────────────────────────────┐
│ Property      │ Meaning                                               │
├───────────────┼───────────────────────────────────────────────────────┤
│ Atomicity     │ All operations in a transaction succeed or ALL fail.  │
│               │ No partial commits. "All or nothing."                 │
│               │ Implementation: Write-ahead log (WAL), undo log      │
├───────────────┼───────────────────────────────────────────────────────┤
│ Consistency   │ A transaction moves the database from one valid       │
│               │ state to another. Constraints (FK, CHECK, UNIQUE)     │
│               │ are always satisfied.                                 │
├───────────────┼───────────────────────────────────────────────────────┤
│ Isolation     │ Concurrent transactions don't interfere with each     │
│               │ other. As if they ran sequentially.                   │
│               │ Levels: READ UNCOMMITTED → READ COMMITTED →           │
│               │ REPEATABLE READ → SERIALIZABLE                       │
├───────────────┼───────────────────────────────────────────────────────┤
│ Durability    │ Once committed, data survives crashes / power loss.   │
│               │ Implementation: WAL flushed to disk before commit     │
│               │ acknowledged.                                        │
└───────────────┴───────────────────────────────────────────────────────┘
```

### Isolation Levels

| Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|-------|-----------|-------------------|--------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented |

> **Trade-off:** Higher isolation = more correctness, but lower concurrency and higher latency. Most production systems use READ COMMITTED (PostgreSQL default) or REPEATABLE READ (MySQL InnoDB default).

---

# 4. Data Warehouses

---

## 4.1 Snowflake

Snowflake is a cloud-native data warehouse that separates storage, compute, and services — Rahul has hands-on experience with Snowflake.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Services Layer                       │
│  (Authentication, Query Optimizer, Metadata, Security,       │
│   Transaction Manager, Infrastructure Manager)               │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────┐
│                    Compute Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Virtual WH   │  │ Virtual WH   │  │ Virtual WH   │       │
│  │ (ETL - XL)   │  │ (Analytics)  │  │ (Data Sci)   │       │
│  │              │  │ (Medium)     │  │ (Small)      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  • Independent scaling — each WH has its own cache           │
│  • Auto-suspend / auto-resume for cost control               │
│  • Can scale up (size) or out (multi-cluster)                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────┐
│                    Storage Layer                              │
│  (Columnar, compressed micro-partitions on S3/Azure/GCS)     │
│  • Automatic clustering and micro-partition pruning           │
│  • Zero-copy cloning                                        │
│  • Immutable storage enables Time Travel                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Virtual Warehouses** | Independent compute clusters. Scale up/down instantly. Pause when idle (pay only when running). |
| **Time Travel** | Query historical data (up to 90 days). `SELECT * FROM t AT(TIMESTAMP => '2024-01-01'::TIMESTAMP)` |
| **Zero-Copy Cloning** | Instantly clone databases/schemas/tables without copying data. Great for dev/test. |
| **Semi-structured data** | Native VARIANT type for JSON, Avro, Parquet, XML. |
| **Streams & Tasks** | CDC (Change Data Capture) via Streams. Scheduled SQL via Tasks. |
| **Snowpipe** | Continuous data ingestion from cloud storage. |

```sql
-- Snowflake: Time Travel
SELECT * FROM orders AT(OFFSET => -3600);        -- 1 hour ago
SELECT * FROM orders BEFORE(STATEMENT => '<query_id>');  -- before a specific query

-- Snowflake: Clone for development
CREATE TABLE orders_dev CLONE orders;             -- instant, zero storage cost

-- Snowflake: Querying semi-structured data
SELECT
    raw:user::STRING AS username,
    raw:events[0]:type::STRING AS first_event_type
FROM raw_events;
```

---

## 4.2 Amazon Redshift

Redshift is a columnar, MPP (Massively Parallel Processing) data warehouse.

### Architecture

```
┌────────────────────────────────────────────────────┐
│                   Leader Node                       │
│  • Receives queries, creates execution plan         │
│  • Distributes work to compute nodes                │
│  • Aggregates results                               │
└────────────────────────┬───────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───┴───┐           ┌───┴───┐           ┌───┴───┐
│Node 1 │           │Node 2 │           │Node 3 │
│┌─────┐│           │┌─────┐│           │┌─────┐│
││Slice││           ││Slice││           ││Slice││
│├─────┤│           │├─────┤│           │├─────┤│
││Slice││           ││Slice││           ││Slice││
│└─────┘│           │└─────┘│           │└─────┘│
└───────┘           └───────┘           └───────┘
  Each slice has its own portion of data and memory
```

### Distribution Styles

| Style | How it works | When to use |
|-------|-------------|-------------|
| **KEY** | Rows with the same key value go to the same slice | Large fact tables frequently joined on that key |
| **EVEN** | Round-robin distribution across slices | Default. When no clear join key exists |
| **ALL** | Full copy on every node | Small dimension tables joined with large fact tables |
| **AUTO** | Redshift decides (starts with ALL, switches to EVEN) | When unsure — let Redshift optimize |

### Sort Keys

```sql
-- Compound sort key: optimizes queries filtering on prefix columns
CREATE TABLE events (
    event_date DATE,
    user_id    INT,
    event_type VARCHAR(50),
    payload    VARCHAR(MAX)
)
DISTSTYLE KEY
DISTKEY (user_id)
COMPOUND SORTKEY (event_date, user_id);

-- Interleaved sort key: equal weight to all sort columns
-- (better for ad-hoc queries on different column combinations)
CREATE TABLE events (...)
INTERLEAVED SORTKEY (event_date, user_id, event_type);
```

| Feature | Compound Sort Key | Interleaved Sort Key |
|---------|------------------|---------------------|
| Best for | Queries filtering on prefix columns | Ad-hoc queries on various columns |
| Maintenance | Low (natural ordering) | Higher (requires VACUUM REINDEX) |
| Zone maps | Very effective for prefix | Effective for all key columns |
| Use case | Time-series data, known query patterns | BI dashboards with varied filters |

### Columnar Storage Advantage

```
Row-oriented (OLTP):                   Column-oriented (OLAP):
┌──────┬──────┬────────┬───────┐       ┌──────┬──────┬──────┬──────┐
│ id   │ name │ salary │ dept  │       │ id   │ id   │ id   │ id   │
│ 1    │ Alice│ 90000  │ Eng   │       │ 1    │ 2    │ 3    │ 4    │
│ 2    │ Bob  │ 85000  │ Mkt   │       ├──────┴──────┴──────┴──────┤
│ 3    │ Carol│ 92000  │ Eng   │       │ name │ name │ name │ name │
│ 4    │ Dan  │ 78000  │ Sales │       │ Alice│ Bob  │ Carol│ Dan  │
└──────┴──────┴────────┴───────┘       ├──────┴──────┴──────┴──────┤
                                       │salary│salary│salary│salary│
Read entire rows → good for           │90000 │85000 │92000 │78000 │
single-record lookups.                └──────┴──────┴──────┴──────┘

                                       Read only needed columns →
                                       great for AVG(salary) — reads
                                       only the salary column.
                                       Also compresses much better
                                       (same data type per block).
```

---

## 4.3 Google BigQuery

BigQuery is a serverless, highly scalable analytics warehouse.

### Key Characteristics

| Feature | Details |
|---------|---------|
| **Serverless** | No infrastructure to manage. Auto-scales. |
| **Pricing** | On-demand ($5/TB scanned) or flat-rate (slots) |
| **Storage** | Columnar (Capacitor format) on Colossus (Google's distributed FS) |
| **Compute** | Dremel engine — tree-based distributed SQL execution |
| **Partitioning** | By ingestion time, DATE/TIMESTAMP column, or integer range |
| **Clustering** | Up to 4 columns — data sorted within partitions for faster scans |

```sql
-- BigQuery: partitioned and clustered table
CREATE TABLE project.dataset.events
(
    event_date   DATE,
    user_id      INT64,
    event_type   STRING,
    properties   JSON
)
PARTITION BY event_date
CLUSTER BY user_id, event_type;

-- This query scans only the relevant partition and cluster range
SELECT event_type, COUNT(*) AS cnt
FROM project.dataset.events
WHERE event_date = '2024-06-15'
  AND user_id BETWEEN 1000 AND 2000
GROUP BY event_type;
```

### Warehouse Comparison

| Feature | Snowflake | Redshift | BigQuery |
|---------|-----------|----------|----------|
| Architecture | Shared storage, separate compute | MPP cluster | Serverless |
| Scaling | Instant warehouse resize | Resize cluster (minutes) | Automatic |
| Pricing model | Credits (compute + storage) | Node hours + storage | Per-query or slots |
| Semi-structured | VARIANT (native) | SUPER type | JSON / STRUCT / ARRAY |
| Time Travel | Up to 90 days | N/A (snapshots) | Up to 7 days |
| Concurrency | Multi-cluster warehouses | WLM queues | Automatic |
| Best for | Mixed workloads, flexibility | AWS-native, steady loads | Google ecosystem, ad-hoc |

---

# 5. Big Data

---

## 5.1 Hadoop Ecosystem

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Hadoop Ecosystem                         │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────┐  │
│  │   Hive    │  │   Pig     │  │  Sqoop    │  │ Flume  │  │
│  │  (SQL)    │  │ (Scripts) │  │ (RDBMS↔HD)│  │ (Logs) │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───┬────┘  │
│        │              │              │             │        │
│  ┌─────┴──────────────┴──────────────┴─────────────┴────┐  │
│  │                    MapReduce / Tez / Spark             │  │
│  │                 (Processing Engines)                   │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │                       YARN                            │  │
│  │          (Resource Management & Scheduling)           │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │                       HDFS                            │  │
│  │          (Hadoop Distributed File System)             │  │
│  │  • Files split into 128MB blocks                      │  │
│  │  • Default 3x replication for fault tolerance         │  │
│  │  • NameNode (metadata) + DataNodes (data)             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### HDFS (Hadoop Distributed File System)

| Concept | Details |
|---------|---------|
| Block size | 128 MB (default); large blocks reduce NameNode metadata overhead |
| Replication | 3x by default; blocks stored on different racks for fault tolerance |
| NameNode | Master — stores metadata (file→block mapping, permissions). Single point of failure (mitigated by HA NameNode) |
| DataNode | Workers — store actual data blocks, send heartbeats to NameNode |
| Write model | Write-once, read-many (append supported, no in-place updates) |

### MapReduce

```
Input → Split → MAP → Shuffle & Sort → REDUCE → Output

Example: Word Count
┌──────────────┐    ┌─────────────────┐    ┌────────────────┐
│ MAP Phase    │    │ Shuffle & Sort  │    │ REDUCE Phase   │
│              │    │                 │    │                │
│ "hello world"│──►│ (hello, [1,1])  │──►│ (hello, 2)     │
│ "hello foo"  │    │ (world, [1])    │    │ (world, 1)     │
│              │    │ (foo,   [1])    │    │ (foo,   1)     │
└──────────────┘    └─────────────────┘    └────────────────┘
```

> MapReduce writes intermediate results to disk — slow for iterative algorithms. This is why Spark (in-memory) replaced it for most use cases.

### YARN (Yet Another Resource Negotiator)

| Component | Role |
|-----------|------|
| ResourceManager | Cluster-wide resource allocation |
| NodeManager | Per-node agent managing containers |
| ApplicationMaster | Per-application coordinator requesting resources |
| Container | Allocated resource (CPU, memory) on a node |

---

## 5.2 Apache Spark

Spark is a unified analytics engine for large-scale data processing — 10–100x faster than MapReduce due to in-memory computing.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Driver Program                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SparkContext / SparkSession              │    │
│  │  • Creates execution plan (DAG)                      │    │
│  │  • Negotiates resources with cluster manager          │    │
│  │  • Distributes tasks to executors                     │    │
│  └─────────────────────────┬───────────────────────────┘    │
└────────────────────────────┼────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
    │ Executor 1│     │ Executor 2│     │ Executor 3│
    │ ┌───────┐ │     │ ┌───────┐ │     │ ┌───────┐ │
    │ │Task   │ │     │ │Task   │ │     │ │Task   │ │
    │ │Task   │ │     │ │Task   │ │     │ │Task   │ │
    │ │Cache  │ │     │ │Cache  │ │     │ │Cache  │ │
    │ └───────┘ │     │ └───────┘ │     │ └───────┘ │
    └───────────┘     └───────────┘     └───────────┘
```

### Core Abstractions

| Abstraction | Description | Introduced |
|-------------|-------------|------------|
| **RDD** (Resilient Distributed Dataset) | Low-level immutable distributed collection. Fine-grained control. | Spark 1.0 |
| **DataFrame** | Distributed table with named columns (like a SQL table). Catalyst optimizer. | Spark 1.3 |
| **Dataset** | Typed DataFrame (compile-time type safety). Scala/Java only. | Spark 1.6 |
| **Spark SQL** | SQL interface on top of DataFrames. Full SQL:2003 support. | Spark 1.0+ |

### Spark SQL

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SalesAnalysis") \
    .getOrCreate()

# Read data
df = spark.read.parquet("s3://bucket/sales/")

# Register as SQL table
df.createOrReplaceTempView("sales")

# Run SQL
result = spark.sql("""
    SELECT
        region,
        product_category,
        SUM(revenue)     AS total_revenue,
        COUNT(*)         AS num_transactions,
        AVG(revenue)     AS avg_revenue
    FROM sales
    WHERE year = 2024
    GROUP BY region, product_category
    ORDER BY total_revenue DESC
""")
result.show()
```

### MLlib

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Feature engineering
assembler = VectorAssembler(inputCols=["age", "income", "score"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=100)

pipeline = Pipeline(stages=[assembler, scaler, rf])
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

---

## 5.3 PySpark Deep Dive

Rahul has hands-on experience with PySpark for large-scale data processing.

### Transformations vs Actions

```
┌─────────────────────────────────────────────────────────────┐
│        Transformations (Lazy)     │     Actions (Eager)      │
├──────────────────────────────────┼──────────────────────────┤
│ map(), flatMap()                 │ collect()                │
│ filter(), where()                │ count()                  │
│ select(), withColumn()           │ show()                   │
│ groupBy(), agg()                 │ take(n), first()         │
│ join()                           │ write.parquet()          │
│ distinct(), dropDuplicates()     │ reduce()                 │
│ repartition(), coalesce()        │ foreach()                │
│ orderBy(), sort()                │ toPandas()               │
│                                  │                          │
│ → Build a DAG (logical plan)     │ → Trigger execution      │
│ → Nothing computed until action  │ → Materialize results    │
└──────────────────────────────────┴──────────────────────────┘
```

> **Lazy evaluation:** Spark builds a DAG of transformations but doesn't execute until an action is called. This allows the Catalyst optimizer to rearrange operations for efficiency.

### Practical PySpark Examples

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("ETL").getOrCreate()

# ── Read from multiple sources ──
orders_df = spark.read.parquet("s3://data-lake/orders/")
customers_df = spark.read.csv("s3://data-lake/customers.csv", header=True, inferSchema=True)

# ── Transformations (lazy — nothing runs yet) ──
cleaned = (
    orders_df
    .filter(F.col("amount") > 0)
    .filter(F.col("order_date") >= "2024-01-01")
    .withColumn("order_month", F.date_trunc("month", "order_date"))
    .dropDuplicates(["order_id"])
)

# ── Join ──
enriched = (
    cleaned
    .join(customers_df, "customer_id", "left")
    .select(
        "order_id", "customer_id", "customer_name",
        "amount", "order_month", "region"
    )
)

# ── Window function: running total per customer ──
window_spec = Window.partitionBy("customer_id").orderBy("order_date")

with_running = enriched.withColumn(
    "running_total", F.sum("amount").over(window_spec)
)

# ── Aggregation ──
monthly_summary = (
    enriched
    .groupBy("order_month", "region")
    .agg(
        F.sum("amount").alias("total_revenue"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.avg("amount").alias("avg_order_value")
    )
    .orderBy("order_month", "region")
)

# ── Action: write to warehouse ──
monthly_summary.write \
    .mode("overwrite") \
    .partitionBy("order_month") \
    .parquet("s3://data-warehouse/monthly_summary/")
```

### Partitioning and Performance

```python
# Check current partitions
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Repartition: full shuffle (expensive but evenly distributes)
df = df.repartition(200, "customer_id")

# Coalesce: reduce partitions without full shuffle (merge only)
df = df.coalesce(50)

# Broadcast join: replicate small table to all nodes
from pyspark.sql.functions import broadcast

result = large_df.join(broadcast(small_dim_df), "key")
```

### Common PySpark Pitfalls

| Pitfall | Solution |
|---------|----------|
| `collect()` on large DataFrame | Use `show()`, `take()`, or write to storage |
| Too few partitions | Repartition to match cluster cores (2-4x cores) |
| Too many small files | `coalesce()` before writing |
| Skewed joins (one key has millions of rows) | Salting: add random prefix to skewed key |
| UDFs are slow (Python serialization) | Use built-in Spark functions or Pandas UDFs |
| Not caching reused DataFrames | `df.cache()` or `df.persist(StorageLevel.MEMORY_AND_DISK)` |

### When to Use Spark vs SQL

| Scenario | Use SQL (Warehouse) | Use Spark |
|----------|-------------------|-----------| 
| Data fits in warehouse | ✓ | |
| Complex ETL with custom logic | | ✓ |
| Unstructured / semi-structured data | | ✓ |
| ML feature engineering at scale | | ✓ |
| Ad-hoc analytics | ✓ | |
| Stream processing | | ✓ (Structured Streaming) |
| Data > warehouse capacity | | ✓ |
| Team knows SQL but not Python | ✓ | |

---

# 6. ETL/ELT Pipelines

---

## 6.1 ETL vs ELT

```
ETL (Extract, Transform, Load):
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│  Source   │──►│  Staging /    │──►│  Data        │
│  Systems  │    │  ETL Server   │    │  Warehouse   │
└──────────┘    │ (Transform    │    └──────────────┘
                │  happens HERE)│
                └───────────────┘
                
ELT (Extract, Load, Transform):
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Source   │──►│  Data        │──►│  Data        │
│  Systems  │    │  Warehouse   │    │  Warehouse   │
└──────────┘    │ (raw landing)│    │  (Transform  │
                └──────────────┘    │  happens HERE │
                                    │  using SQL)   │
                                    └──────────────┘
```

| Aspect | ETL | ELT |
|--------|-----|-----|
| Transform location | External server / engine | Inside the warehouse |
| Best when | Warehouse compute is expensive, data needs heavy cleansing | Warehouse is powerful (Snowflake, BigQuery) |
| Latency | Higher (extra hop) | Lower (load first, transform in-place) |
| Flexibility | Less (schema must be defined upfront) | More (raw data preserved for future transformations) |
| Tools | Informatica, Talend, custom Python | dbt, Snowflake Tasks, BigQuery scheduled queries |
| Data volume | Moderate | Very large (warehouse handles scale) |
| Schema approach | Schema-on-write | Schema-on-read |

> **Modern trend:** ELT is dominant in cloud-native architectures. The warehouse has enough compute to handle transformations, and keeping raw data supports evolving requirements.

---

## 6.2 Pipeline Tools

### Apache Airflow

Airflow is the most widely used workflow orchestrator for data pipelines.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "rahul",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id="daily_sales_etl",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "sales"],
) as dag:

    extract = PythonOperator(
        task_id="extract_from_api",
        python_callable=extract_sales_data,
        op_kwargs={"date": "{{ ds }}"},       # Jinja-templated execution date
    )

    load_raw = SnowflakeOperator(
        task_id="load_to_raw",
        sql="COPY INTO raw.sales FROM @s3_stage/{{ ds }}/",
        snowflake_conn_id="snowflake_default",
    )

    transform = SnowflakeOperator(
        task_id="transform_sales",
        sql="sql/transform_sales.sql",
        snowflake_conn_id="snowflake_default",
    )

    quality_check = PythonOperator(
        task_id="data_quality_check",
        python_callable=run_quality_checks,
    )

    extract >> load_raw >> transform >> quality_check
```

```
DAG Visualization:
extract_from_api → load_to_raw → transform_sales → data_quality_check
```

### dbt (data build tool)

dbt is the standard tool for ELT transformations — SQL-based, version-controlled, tested.

```sql
-- models/marts/monthly_revenue.sql
{{
    config(
        materialized='incremental',
        unique_key='month_region',
        cluster_by=['region']
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
    {% if is_incremental() %}
    WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
    {% endif %}
),

customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
    DATE_TRUNC('month', o.order_date)  AS month,
    c.region,
    CONCAT(DATE_TRUNC('month', o.order_date), '_', c.region) AS month_region,
    SUM(o.amount)                      AS total_revenue,
    COUNT(DISTINCT o.customer_id)      AS unique_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY 1, 2, 3
```

```yaml
# models/marts/monthly_revenue.yml (dbt tests)
version: 2
models:
  - name: monthly_revenue
    description: Monthly revenue aggregated by region
    columns:
      - name: month_region
        tests:
          - unique
          - not_null
      - name: total_revenue
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
```

### Pipeline Tool Comparison

| Tool | Type | Strengths | Weaknesses |
|------|------|-----------|------------|
| **Airflow** | Orchestrator | Flexible, Python-based, huge ecosystem | Complex setup, not ideal for simple SQL workflows |
| **dbt** | Transformation | SQL-native, testing, docs, lineage | Only transforms (not extract or load) |
| **Fivetran** | EL (Extract + Load) | 300+ connectors, managed, reliable | Expensive, limited transformation |
| **Prefect** | Orchestrator | Modern Airflow alternative, easier setup | Smaller ecosystem |
| **Dagster** | Orchestrator | Asset-centric, built-in data lineage | Steeper learning curve |
| **Custom Python** | Any | Maximum flexibility | Maintenance burden, reinventing the wheel |

---

## 6.3 Data Quality Checks

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Quality Dimensions                    │
├──────────────┬──────────────────────────────────────────────┤
│ Completeness │ Are required fields populated?                │
│              │ CHECK: NULL counts, missing rate thresholds   │
├──────────────┼──────────────────────────────────────────────┤
│ Uniqueness   │ Are primary keys unique?                      │
│              │ CHECK: duplicate detection queries             │
├──────────────┼──────────────────────────────────────────────┤
│ Freshness    │ Is data arriving on time?                     │
│              │ CHECK: MAX(updated_at) within expected window │
├──────────────┼──────────────────────────────────────────────┤
│ Validity     │ Do values conform to expected formats/ranges? │
│              │ CHECK: email regex, dates in valid range       │
├──────────────┼──────────────────────────────────────────────┤
│ Consistency  │ Do values agree across sources/tables?         │
│              │ CHECK: referential integrity, cross-table sums │
├──────────────┼──────────────────────────────────────────────┤
│ Volume       │ Is the row count within expected bounds?       │
│              │ CHECK: row_count BETWEEN min AND max           │
└──────────────┴──────────────────────────────────────────────┘
```

```sql
-- Data quality check examples
-- 1. Completeness
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) AS null_emails,
    ROUND(100.0 * SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS null_pct
FROM customers;

-- 2. Uniqueness
SELECT order_id, COUNT(*) AS cnt
FROM orders
GROUP BY order_id
HAVING COUNT(*) > 1;

-- 3. Freshness
SELECT
    MAX(updated_at) AS latest_update,
    CURRENT_TIMESTAMP - MAX(updated_at) AS staleness
FROM orders
HAVING MAX(updated_at) < CURRENT_TIMESTAMP - INTERVAL '2 hours';
```

---

## 6.4 Incremental vs Full Load

| Strategy | Full Load | Incremental Load |
|----------|-----------|-----------------|
| What loads | Entire dataset every run | Only new/changed records |
| Complexity | Simple | Requires tracking changes |
| Duration | Long (grows with data) | Short (proportional to delta) |
| Idempotent | Naturally (replace all) | Must handle carefully |
| Use when | Small tables, first load, data recovery | Large tables, frequent updates |

```sql
-- Incremental load with high-water mark
-- Step 1: Get last loaded timestamp
SET @last_loaded = (SELECT MAX(loaded_at) FROM warehouse.orders);

-- Step 2: Extract only new/updated records
INSERT INTO warehouse.orders
SELECT * FROM source.orders
WHERE updated_at > @last_loaded;
```

---

## 6.5 CDC (Change Data Capture)

CDC captures row-level changes (INSERT, UPDATE, DELETE) from source databases.

```
┌──────────────────────────────────────────────────────────────┐
│                    CDC Methods                                │
├─────────────────┬────────────────────────────────────────────┤
│ Log-based       │ Read database transaction log (WAL, binlog)│
│                 │ Tools: Debezium, AWS DMS, Fivetran          │
│                 │ ✓ No source impact, captures all changes    │
│                 │ ✗ Requires DB admin access                  │
├─────────────────┼────────────────────────────────────────────┤
│ Trigger-based   │ DB triggers write to a shadow table         │
│                 │ ✓ Simple to implement                       │
│                 │ ✗ Adds overhead to source DB                │
├─────────────────┼────────────────────────────────────────────┤
│ Timestamp-based │ Query rows WHERE updated_at > last_sync     │
│                 │ ✓ Simple, no special DB access               │
│                 │ ✗ Misses deletes, requires reliable timestamp│
├─────────────────┼────────────────────────────────────────────┤
│ Diff-based      │ Compare snapshots to detect changes         │
│                 │ ✓ Works with any source                     │
│                 │ ✗ Resource-intensive for large tables        │
└─────────────────┴────────────────────────────────────────────┘
```

```sql
-- Snowflake Streams: built-in CDC
CREATE STREAM orders_stream ON TABLE raw.orders;

-- Stream captures INSERT, UPDATE, DELETE since last consumed
SELECT * FROM orders_stream;

-- Consume stream in a MERGE
MERGE INTO analytics.orders t
USING orders_stream s
ON t.order_id = s.order_id
WHEN MATCHED AND s.METADATA$ACTION = 'DELETE' THEN DELETE
WHEN MATCHED AND s.METADATA$ACTION = 'INSERT' THEN  -- UPDATE = DELETE + INSERT
    UPDATE SET t.amount = s.amount, t.updated_at = s.updated_at
WHEN NOT MATCHED AND s.METADATA$ACTION = 'INSERT' THEN
    INSERT (order_id, amount, updated_at) VALUES (s.order_id, s.amount, s.updated_at);
```

---

# 7. Graph Databases

---

## 7.1 Neo4j

Neo4j is the leading graph database, storing data as **nodes** and **relationships** rather than rows and tables. Rahul has experience with Neo4j.

### Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│              Graph Data Model                                │
│                                                             │
│   (Alice)──[:FRIENDS_WITH]──►(Bob)                          │
│     │                          │                            │
│     │ :WORKS_AT                │ :WORKS_AT                  │
│     ▼                          ▼                            │
│   (Google)                   (Meta)                         │
│     │                                                       │
│     │ :LOCATED_IN                                           │
│     ▼                                                       │
│   (Mountain View)──[:IN_STATE]──►(California)               │
│                                                             │
│   Node: Entity (person, company, city)                      │
│   Relationship: Connection with direction and type          │
│   Property: Key-value pairs on nodes or relationships       │
│   Label: Category/type tag on nodes (e.g., :Person)         │
└─────────────────────────────────────────────────────────────┘
```

### Cypher Query Language

```cypher
// Create nodes and relationships
CREATE (alice:Person {name: 'Alice', age: 30})
CREATE (bob:Person {name: 'Bob', age: 28})
CREATE (google:Company {name: 'Google'})
CREATE (alice)-[:FRIENDS_WITH {since: 2020}]->(bob)
CREATE (alice)-[:WORKS_AT {role: 'Engineer'}]->(google)

// Find friends of Alice
MATCH (alice:Person {name: 'Alice'})-[:FRIENDS_WITH]->(friend)
RETURN friend.name, friend.age

// Friends-of-friends (2 hops)
MATCH (alice:Person {name: 'Alice'})-[:FRIENDS_WITH*2]->(fof)
WHERE fof <> alice
RETURN DISTINCT fof.name

// Shortest path between two people
MATCH path = shortestPath(
    (a:Person {name: 'Alice'})-[:FRIENDS_WITH*]-(b:Person {name: 'Dan'})
)
RETURN path, length(path)

// Aggregation: company with most employees
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, COUNT(p) AS employee_count
ORDER BY employee_count DESC
LIMIT 10

// Pattern matching: mutual friends
MATCH (a:Person)-[:FRIENDS_WITH]->(mutual)<-[:FRIENDS_WITH]-(b:Person)
WHERE a.name = 'Alice' AND b.name = 'Carol'
RETURN mutual.name AS mutual_friend

// Recommendation: people who share interests
MATCH (me:Person {name: 'Alice'})-[:INTERESTED_IN]->(topic)<-[:INTERESTED_IN]-(other)
WHERE NOT (me)-[:FRIENDS_WITH]-(other)
RETURN other.name, COLLECT(topic.name) AS shared_interests, COUNT(*) AS score
ORDER BY score DESC
```

---

## 7.2 Graph DB vs Relational DB

| Aspect | Relational (SQL) | Graph (Neo4j) |
|--------|------------------|---------------|
| Data model | Tables with rows and columns | Nodes and relationships |
| Relationships | Foreign keys + JOINs | First-class citizens, stored with data |
| Traversal depth | Expensive multi-table JOINs (O(n) per hop) | Cheap (index-free adjacency, O(1) per hop) |
| Schema | Rigid (predefined tables) | Flexible (labels, properties) |
| Best for | Structured, tabular data | Connected data, network analysis |
| Query language | SQL | Cypher (Neo4j), Gremlin, SPARQL |

### When to Use Each

| Use Case | Best Choice | Why |
|----------|------------|-----|
| E-commerce orders, inventory | Relational | Structured, well-defined schema |
| Social networks, recommendations | Graph | Deep relationship traversals |
| Fraud detection (transaction chains) | Graph | Pattern matching across many hops |
| Financial reporting | Relational | Aggregations, well-defined dimensions |
| Knowledge graphs | Graph | Flexible schema, semantic relationships |
| IoT sensor data | Relational / Time-series | Regular structure, time-ordered |
| Supply chain dependencies | Graph | Complex multi-hop dependencies |

---

# 8. Data Modeling for ML

---

## 8.1 Feature Stores

A feature store is a centralized repository for storing, serving, and managing ML features.

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Store Architecture                 │
│                                                             │
│   ┌─────────────┐     ┌──────────────┐     ┌────────────┐  │
│   │ Data Sources │────►│ Feature      │────►│ Offline     │  │
│   │ (DB, API,   │     │ Engineering  │     │ Store       │  │
│   │  Streams)   │     │ Pipelines    │     │ (Training)  │  │
│   └─────────────┘     └──────┬───────┘     └────────────┘  │
│                              │                              │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │ Feature      │                      │
│                       │ Registry     │                      │
│                       │ (Metadata,   │                      │
│                       │  Schemas,    │                      │
│                       │  Lineage)    │                      │
│                       └──────┬───────┘                      │
│                              │                              │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │ Online Store │                      │
│                       │ (Low-latency │                      │
│                       │  serving)    │                      │
│                       └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

| Tool | Type | Key Features |
|------|------|-------------|
| **Feast** | Open source | Offline + online stores, Snowflake/Redshift support |
| **Tecton** | Managed | Real-time features, Databricks integration |
| **Vertex AI Feature Store** | GCP managed | BigQuery integration, monitoring |
| **SageMaker Feature Store** | AWS managed | Redshift integration, offline + online |

### Why Feature Stores Matter

- **Consistency:** Same features in training and serving (avoid train/serve skew)
- **Reusability:** Teams share features instead of rebuilding
- **Point-in-time correctness:** Features joined as of the training example's timestamp (avoids data leakage)
- **Monitoring:** Track feature drift, schema changes

---

## 8.2 Data Versioning

| Tool | Approach | Best For |
|------|----------|----------|
| **DVC** (Data Version Control) | Git-like commands for data files, stored in S3/GCS | Dataset versioning alongside code |
| **Delta Lake** | ACID transactions on data lakes (Spark ecosystem) | Lakehouse architecture |
| **Apache Iceberg** | Open table format with snapshot isolation | Multi-engine (Spark, Flink, Trino) |
| **LakeFS** | Git-like branching for data lakes | Experimentation, CI/CD for data |

```bash
# DVC: track a large dataset
dvc init
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc .gitignore
git commit -m "Track training data v1"
dvc push                                # Push data to S3/GCS

# Later: reproduce with exact same data
git checkout v1.0
dvc checkout
```

```sql
-- Delta Lake: time travel
SELECT * FROM delta.`/data/features/` VERSION AS OF 42;
SELECT * FROM delta.`/data/features/` TIMESTAMP AS OF '2024-06-01';

-- Restore previous version
RESTORE TABLE features TO VERSION AS OF 42;
```

---

## 8.3 Train/Test Data Management

```
┌─────────────────────────────────────────────────────────────┐
│              ML Data Pipeline Best Practices                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. IMMUTABLE DATASETS                                       │
│    • Never modify raw data in-place                         │
│    • Version every transformation                           │
│    • Raw → Cleaned → Features → Train/Test (clear stages)   │
│                                                             │
│ 2. REPRODUCIBLE SPLITS                                       │
│    • Use deterministic hashing for splits                    │
│    • Hash(user_id) % 10 < 2 → test set                     │
│    • Avoid random splits that change across runs            │
│                                                             │
│ 3. TEMPORAL SPLITS FOR TIME SERIES                           │
│    • Train on historical data, test on future data          │
│    • Never leak future data into training                   │
│    • Rolling window validation                              │
│                                                             │
│ 4. STRATIFIED SPLITS FOR IMBALANCED DATA                    │
│    • Preserve class distribution in train/test              │
│    • Use stratified k-fold cross-validation                 │
│                                                             │
│ 5. DATA LEAKAGE PREVENTION                                  │
│    • Feature computation ONLY on training data              │
│    • Fit scalers/encoders on train, apply to test           │
│    • Point-in-time joins for feature stores                 │
│                                                             │
│ 6. METADATA TRACKING                                        │
│    • Log dataset versions, split ratios, row counts         │
│    • Link each model version to its training data version   │
│    • Tools: MLflow, W&B, DVC                                │
└─────────────────────────────────────────────────────────────┘
```

```python
import hashlib

def deterministic_split(user_id: str, test_pct: float = 0.2) -> str:
    """Deterministic train/test split based on user_id hash."""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "test" if (hash_val % 100) < (test_pct * 100) else "train"
```

```sql
-- SQL-based deterministic split
SELECT
    *,
    CASE
        WHEN MOD(ABS(HASH(user_id)), 100) < 20 THEN 'test'
        ELSE 'train'
    END AS split
FROM feature_table;
```

---

# 9. Common Interview Questions

---

## Q1: Write a SQL query to find the 2nd highest salary

```sql
-- Method 1: Subquery
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 2: DENSE_RANK (handles ties correctly)
WITH ranked AS (
    SELECT
        name,
        salary,
        DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
)
SELECT name, salary
FROM ranked
WHERE rnk = 2;

-- Method 3: OFFSET (if only the value is needed)
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

> **Why DENSE_RANK over ROW_NUMBER?** If two employees share the highest salary, DENSE_RANK correctly assigns rank 2 to the next lower salary. ROW_NUMBER would arbitrarily assign ranks 1 and 2 to the tied employees.

---

## Q2: Write a SQL query for a running total

```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM orders;
```

**Follow-up: running total per customer:**

```sql
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
    ) AS customer_running_total
FROM orders;
```

---

## Q3: Detect and remove duplicates

```sql
-- Detect duplicates
SELECT
    email,
    COUNT(*) AS duplicate_count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- Remove duplicates (keep the one with the lowest id)
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY email
            ORDER BY id ASC
        ) AS rn
    FROM customers
)
DELETE FROM customers
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- PostgreSQL: elegant CTID approach
DELETE FROM customers a
USING customers b
WHERE a.ctid > b.ctid
  AND a.email = b.email;
```

---

## Q4: Explain window functions

> **Answer framework:**
>
> "Window functions perform calculations across a set of rows related to the current row, without collapsing them into a single output row like GROUP BY does. They use the OVER() clause which can include PARTITION BY (to define groups), ORDER BY (to define row order within each group), and a frame specification (to define which rows relative to the current row are included in the calculation).
>
> Common types include:
> - **Ranking:** ROW_NUMBER, RANK, DENSE_RANK, NTILE
> - **Value access:** LAG, LEAD, FIRST_VALUE, LAST_VALUE
> - **Aggregation:** SUM, AVG, COUNT, MIN, MAX (with OVER)
>
> For example, to compute a running average of sales per region: `AVG(sales) OVER (PARTITION BY region ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` gives a 7-day moving average within each region."

---

## Q5: OLTP vs OLAP

> **Answer framework:**
>
> "OLTP (Online Transaction Processing) handles day-to-day operations — think of a bank processing thousands of small transactions per second. It uses normalized schemas, row-oriented storage, and is optimized for fast writes with full ACID compliance. Examples: PostgreSQL, MySQL, MS SQL Server.
>
> OLAP (Online Analytical Processing) handles analytics — think of a data analyst running a query that aggregates a year's worth of sales across all regions. It uses denormalized schemas (star/snowflake), columnar storage, and is optimized for fast reads on large datasets. Examples: Snowflake, Redshift, BigQuery.
>
> The key trade-off: OLTP favors write speed and data integrity; OLAP favors read speed and analytical flexibility."

---

## Q6: Star schema vs Snowflake schema

> **Answer framework:**
>
> "Both are dimensional modeling approaches for data warehouses. A star schema has a central fact table surrounded by denormalized dimension tables — simple, fast queries, at the cost of some redundancy. A snowflake schema normalizes the dimension tables into sub-dimensions, reducing redundancy but requiring more JOINs and adding query complexity.
>
> In practice, star schemas are preferred for most data warehouse workloads because modern warehouses (Snowflake, Redshift, BigQuery) handle the redundancy cheaply and the simpler JOINs are much easier for analysts to write and the query optimizer to execute. Snowflake schemas are used when dimension tables are very large and the redundancy becomes significant."

---

## Q7: When would you use Spark over SQL?

> **Answer framework:**
>
> "I use SQL when the data is already in a warehouse and the transformation logic can be expressed in SQL — it's simpler, more widely understood, and modern warehouses like Snowflake and BigQuery optimize SQL extremely well.
>
> I switch to Spark when:
> 1. **Data exceeds warehouse capacity** or lives in a data lake (S3/HDFS) in formats like Parquet/JSON
> 2. **Complex ETL logic** that's difficult in pure SQL — custom parsing, ML feature engineering, calling external APIs
> 3. **Stream processing** is needed (Structured Streaming)
> 4. **Iterative ML algorithms** at scale (MLlib)
> 5. **Multi-source data** needs to be combined before loading into a warehouse
>
> In my experience with PySpark, I've used it for large-scale data processing and feature engineering tasks where the data lived on S3 and the transformations involved custom Python logic."

---

## Q8: How do you optimize a slow SQL query?

> **Answer framework:**
>
> "My systematic approach:
> 1. **Run EXPLAIN ANALYZE** to see the actual execution plan and identify bottlenecks (full table scans, expensive sorts, hash joins on large tables)
> 2. **Check indexes** — are the WHERE, JOIN, and ORDER BY columns indexed? Is the query hitting the right index (composite index column order matters)?
> 3. **Examine the query itself:**
>    - Replace `SELECT *` with specific columns
>    - Rewrite correlated subqueries as JOINs
>    - Use `EXISTS` instead of `IN` for large subqueries
>    - Avoid functions on indexed columns (e.g., `WHERE YEAR(date_col) = 2024` kills index usage)
> 4. **Check data distribution** — statistics might be stale (`ANALYZE TABLE`)
> 5. **Consider partitioning** for large tables (by date is most common)
> 6. **Materialized views** for frequently computed complex aggregations
> 7. **In warehouse context** — check distribution keys (Redshift), clustering keys (Snowflake), or partition pruning (BigQuery)"

---

## Q9: ETL vs ELT — When do you use which?

> **Answer framework:**
>
> "ETL transforms data before loading into the warehouse. ELT loads raw data first, then transforms inside the warehouse.
>
> I use **ELT** (the modern default) when:
> - The warehouse has strong compute (Snowflake, BigQuery, Redshift)
> - We want to preserve raw data for evolving transformation needs
> - The transformation logic is expressible in SQL
> - dbt makes the transformation layer maintainable and testable
>
> I use **ETL** when:
> - The warehouse compute is expensive and limited
> - Data needs significant cleansing or reformatting before it's useful (e.g., parsing unstructured logs with Python)
> - Sensitive data must be masked/anonymized before entering the warehouse
> - Data from many heterogeneous sources needs standardization
>
> In practice, most modern pipelines are hybrid: Fivetran or Airbyte extracts and loads (EL), then dbt transforms inside the warehouse (T)."

---

## Q10: How would you design a data pipeline for real-time analytics?

> **Answer framework:**
>
> "For real-time analytics, I'd design a Lambda or Kappa architecture:
>
> **Stream layer:**
> - Source systems emit events to Kafka (or Kinesis/Pub-Sub)
> - Spark Structured Streaming or Flink processes events in micro-batches or true streaming
> - Results written to a low-latency store (Redis, DynamoDB) for dashboards
>
> **Batch layer (Lambda only):**
> - Same source data lands in S3/GCS in Parquet format
> - Daily Spark/dbt jobs reprocess for accuracy (corrects late-arriving data)
> - Results in Snowflake/BigQuery for historical analytics
>
> **Serving layer:**
> - Real-time dashboards query the online store
> - Historical analytics query the warehouse
>
> Key considerations: idempotency (exactly-once processing), schema evolution, monitoring, backpressure handling, and alerting on latency SLA breaches."

---

## Q11: Explain CDC and how you would implement it

> **Answer framework:**
>
> "CDC (Change Data Capture) tracks row-level changes in source databases — INSERTs, UPDATEs, and DELETEs — and propagates them downstream.
>
> **Log-based CDC** is the gold standard: tools like Debezium read the database's transaction log (PostgreSQL WAL, MySQL binlog) and emit change events to Kafka. This has zero impact on source database performance and captures all changes including deletes.
>
> In Snowflake, I'd use **Streams** — a built-in CDC mechanism that tracks changes on a table and exposes them as a queryable stream, which I consume with a MERGE statement in a scheduled Task.
>
> **Timestamp-based CDC** is simpler: query `WHERE updated_at > last_sync_timestamp`. But it misses deletes and requires a reliable timestamp column on every table."

---

## Q12: What is data skew and how do you handle it in Spark?

> **Answer framework:**
>
> "Data skew occurs when some partitions have significantly more data than others, causing a few tasks to run much longer than the rest (the slowest task determines overall job completion).
>
> Common solutions:
> 1. **Salting:** Add a random prefix to the skewed key, join on the salted key, then aggregate to remove the salt
> 2. **Broadcast join:** If one side is small enough, broadcast it to all nodes (avoids shuffle entirely)
> 3. **Adaptive Query Execution (AQE):** Spark 3.0+ can automatically detect and split skewed partitions at runtime (`spark.sql.adaptive.enabled = true`)
> 4. **Repartition:** Explicitly redistribute data across more partitions
> 5. **Two-phase aggregation:** First aggregate within each partition, then aggregate the partial results"

---

## Q13: How does pgvector work for vector similarity search?

> **Answer framework (ties to Rahul's pgvector experience):**
>
> "pgvector is a PostgreSQL extension that adds vector data types and similarity search operators. It stores embeddings directly in PostgreSQL columns using the `vector` type, enabling combined SQL + vector queries in one database.
>
> Key features:
> - Distance functions: L2 (`<->`), inner product (`<#>`), cosine (`<=>`), L1 (`<+>`)
> - Index types: IVFFlat (inverted file, fast but approximate) and HNSW (graph-based, better recall)
> - Integrates with standard SQL — you can JOIN embeddings with metadata, filter by business logic, and sort by similarity all in one query
>
> I've used pgvector with Supabase for embedding-powered search in production, combining it with standard relational queries for filtering and ranking."

```sql
-- pgvector example
CREATE EXTENSION vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(1536)   -- OpenAI ada-002 dimension
);

-- HNSW index for fast approximate nearest neighbor
CREATE INDEX idx_docs_embedding ON documents
USING hnsw (embedding vector_cosine_ops);

-- Find 10 most similar documents
SELECT title, content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
WHERE category = 'engineering'   -- combine with SQL filters
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

---

# 10. Key Takeaways

---

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KEY TAKEAWAYS                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SQL FUNDAMENTALS                                                   │
│  • Master the logical execution order (FROM → WHERE → GROUP BY →    │
│    HAVING → SELECT → ORDER BY) — it explains most SQL "gotchas"    │
│  • JOINs are the most tested SQL topic — know all types and when   │
│    each is appropriate                                              │
│  • Window functions are essential — ROW_NUMBER, RANK, LAG/LEAD,    │
│    running totals                                                   │
│  • CTEs improve readability; recursive CTEs solve hierarchical      │
│    problems                                                         │
│                                                                     │
│  ADVANCED SQL                                                       │
│  • Index selection is the #1 performance lever — understand B-tree  │
│    leftmost prefix rule, covering indexes, and partial indexes      │
│  • EXPLAIN ANALYZE is your best friend for optimization             │
│  • Partitioning (range/list/hash) reduces scan scope for large      │
│    tables                                                           │
│                                                                     │
│  DATABASE DESIGN                                                    │
│  • OLTP = normalized, row-oriented, write-optimized                 │
│  • OLAP = denormalized, columnar, read-optimized                    │
│  • Star schema is the standard for warehouses                       │
│  • ACID guarantees correctness; isolation levels trade correctness   │
│    for performance                                                  │
│                                                                     │
│  DATA WAREHOUSES                                                    │
│  • Snowflake: separate compute + storage, Time Travel, Streams      │
│  • Redshift: distribution styles (KEY/EVEN/ALL), sort keys          │
│  • BigQuery: serverless, partition + cluster, pay-per-query          │
│                                                                     │
│  BIG DATA                                                           │
│  • Spark replaced MapReduce (in-memory vs disk-based)               │
│  • PySpark: lazy evaluation, transformations vs actions,             │
│    broadcast joins, partitioning                                    │
│  • Use Spark for scale + complex logic; SQL for warehouse analytics │
│                                                                     │
│  ETL/ELT                                                            │
│  • Modern stack: Fivetran (EL) + dbt (T) + Airflow (orchestration) │
│  • ELT is the modern default — transform inside the warehouse       │
│  • CDC (log-based) is the gold standard for change capture           │
│  • Data quality checks are non-negotiable in production pipelines    │
│                                                                     │
│  GRAPH DATABASES                                                    │
│  • Neo4j for connected data: social graphs, fraud detection,        │
│    recommendations                                                  │
│  • Cypher: declarative pattern matching for graph traversals        │
│  • Use graph DB when relationship traversal depth > 2-3 hops        │
│                                                                     │
│  DATA MODELING FOR ML                                               │
│  • Feature stores ensure consistency between training and serving   │
│  • Version your data alongside your code (DVC, Delta Lake)          │
│  • Deterministic splits > random splits for reproducibility          │
│  • Guard against data leakage at every stage                        │
│                                                                     │
│  RAHUL'S DIFFERENTIATORS                                            │
│  • Hands-on with Snowflake, Redshift, PySpark, Neo4j, pgvector     │
│  • Built production ETL pipelines with Supabase integration         │
│  • Experience spanning OLTP (PostgreSQL, MySQL, MS SQL Server)      │
│    and OLAP (Snowflake, Redshift)                                   │
│  • Combines data engineering with ML — feature engineering,          │
│    embedding pipelines, and vector search in production              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Prepared for Rahul Sharma — Data Engineering & SQL Interview Preparation*
