# Options for Cleaning Large Data: Pros, Cons, and When to Migrate

This doc summarizes local and cloud/distributed options for cleaning large datasets, with pros/cons and a simple decision guide for when to move off pandas.

---

## Local – Streaming / Chunked

### Pandas `read_csv(chunksize=...)` + process + write

**Pros:** Simple, well-known API; no new stack; easy to debug; works everywhere.  
**Cons:** Single-threaded; slower than Polars/DuckDB for big CSVs; you hand-roll streaming and dedupe.  
**Verdict:** Default choice when data is "large but fits a single machine" and you care more about simplicity than speed.

---

### Polars (LazyFrame, `scan_csv`, `sink_parquet`)

**Pros:** Much faster than pandas on big CSV/Parquet; streaming via lazy evaluation; familiar DataFrame API; good Rust-based engine.  
**Cons:** Different API and semantics (lazy, expressions); team has to learn it; ecosystem smaller than pandas.  
**Verdict:** Good upgrade when pandas chunked runs are too slow and you're still on one machine.

---

### DuckDB

**Pros:** SQL over CSV/Parquet; no "load into memory" step; very fast filters/aggregates/joins; in-process, no server; great for dedupe and ad-hoc cleaning.  
**Cons:** You think in SQL (or pyarrow-style APIs), not DataFrames; less natural for row-by-row Python logic.  
**Verdict:** Fits "filter, dedupe, aggregate, join then export" pipelines; less natural for heavy custom Python per row.

---

### PyArrow (Dataset / scan APIs)

**Pros:** Columnar, I/O efficient; integrates with Parquet/Arrow everywhere; can do streaming and partitioning.  
**Cons:** More low-level; less "analysis-ready" than pandas/Polars for messy cleaning; you often pair with pandas for complex logic.  
**Verdict:** Use when you're already Arrow/Parquet-centric or need maximum I/O control.

---

### Vaex

**Pros:** Can work with larger-than-RAM data via memory mapping and lazy ops; pandas-like for some workflows.  
**Cons:** Smaller community; API and behavior differ from pandas in subtle ways; less standard than pandas/Polars.  
**Verdict:** Niche when you explicitly want "pandas-like but out-of-core" and don't want Spark/Dask.

---

## Local – "Fit in Memory" or Database

### Pandas in-memory (full load)

**Pros:** Easiest: load once, then filter/dedupe/transform; full flexibility; fast iteration.  
**Cons:** Only for data that fits in RAM (and leaves room for operations).  
**Verdict:** Use when the full dataset fits comfortably in memory; otherwise use chunked pandas or another option.

---

### SQLite

**Pros:** One file, no server; SQL for dedupe/clean; portable; trivial to set up.  
**Cons:** Single writer; not built for analytics at scale; can be slow for huge imports.  
**Verdict:** Good for "clean and dedupe with SQL" on small–medium data (e.g. up to tens of GB with care).

---

## Cloud / Distributed

### Spark (Databricks, EMR, etc.)

**Pros:** Standard for big-data ETL; scales to TB+; rich ecosystem; SQL and DataFrame APIs.  
**Cons:** Cluster cost and ops; slower startup; overkill for tens of GB; tuning and debugging are harder.  
**Verdict:** Use when data or compute clearly exceed one machine (e.g. 100+ GB, or need many cores/nodes).

---

### Dask

**Pros:** Pandas-like API; scales from one machine to a cluster; can keep "pandas-style" code.  
**Cons:** Debugging and performance tuning are harder; cluster setup; not as fast as Spark for some workloads.  
**Verdict:** Good when you want "pandas at scale" and are willing to manage scheduling and clusters.

---

### Ray (Ray Data)

**Pros:** Flexible Python; good for ML pipelines and custom code; scales to a cluster.  
**Cons:** Newer in the data space; less "standard" than Spark for pure ETL; you manage Ray cluster.  
**Verdict:** Fits when you already use Ray for training/serving and want one stack for data + ML.

---

### BigQuery / Snowflake / Redshift

**Pros:** No cluster to manage; SQL only; auto-scale; good for one-off and recurring cleans.  
**Cons:** Data and compute in the cloud; cost per query/scan; less control than Spark; vendor lock-in.  
**Verdict:** Use when data is (or can be) in the cloud and cleaning is mostly SQL.

---

### Athena / Trino (query files in S3)

**Pros:** Query CSVs/Parquet in place; no load step; pay per query.  
**Cons:** Can be slow and costly for full scans; less ideal for heavy multi-pass cleaning.  
**Verdict:** Good for ad-hoc exploration and lighter cleans; not always the best for "full rewrite" pipelines.

---

### Batch jobs (Cloud Run Jobs, AWS Batch, etc.)

**Pros:** Run your existing script (e.g. pandas chunked) on a bigger VM; minimal code change; no distributed framework.  
**Cons:** Still one process; limited by single-machine RAM/CPU; no horizontal scale.  
**Verdict:** Use when the bottleneck is "my machine is too small" rather than "I need 10 machines."

---

## When to Migrate from Pandas (Decision Guide)

Use these as rules of thumb, not hard limits.

### Stay with pandas (chunked) when:

- **Runtime is acceptable** – e.g. full pipeline in minutes to low tens of minutes, and you're not re-running it constantly.
- **Data size is "one machine"** – e.g. tens of GB to low hundreds of GB, and chunked runs in reasonable time.
- **Team and maintenance matter more than speed** – pandas is known, debuggable, and sufficient.
- **You're still changing the pipeline** – logic and correctness matter more than shaving 2x off runtime.

**Signal to stay:** "It runs in a reasonable time and we're not waiting on it every day."

---

### Move to Polars when:

- **Pandas chunked is too slow** – e.g. same pipeline takes hours or blocks iteration.
- **Data is still on one machine** – no need for a cluster yet.
- **Pipeline is mostly "read → filter/aggregate/join → write"** – things Polars is good at.
- **You're willing to learn a new API** – LazyFrame and expression syntax.

**Signal to migrate:** "We run this often and we're waiting on pandas."

---

### Move to DuckDB when:

- **Cleaning is expressible in SQL** – filters, dedupe, joins, aggregates.
- **You want to avoid loading full data into Python** – query CSV/Parquet directly.
- **You do ad-hoc exploration** – quick SQL over the same files.
- **You're fine with SQL as the main interface** – not "every row through custom Python."

**Signal to migrate:** "Most of our cleaning could be a SQL query" or "we keep writing one-off pandas scripts to do what SQL could do."

---

### Move to Spark / Dask / Ray when:

- **Data or compute exceed one machine** – e.g. 100+ GB and pandas/Polars/DuckDB are slow or OOM.
- **You need parallelism across many cores or nodes** – e.g. 10+ workers.
- **You have (or will have) a cluster** – or use a managed Spark/Dask service.
- **Pipeline is stable enough** – worth the extra complexity of distributed execution and debugging.

**Signal to migrate:** "We can't finish on one machine" or "we need to scale out, not just up."

---

### Move to a cloud warehouse when:

- **Data already lives in the cloud** (e.g. S3, GCS) and you're okay with cloud-native processing.
- **Cleaning is mostly SQL** and you don't need heavy custom Python per row.
- **You prefer no cluster management** and are okay with per-query cost.
- **Downstream use is in the same cloud** (e.g. BigQuery → Looker, Snowflake → dbt).

**Signal to migrate:** "Our source of truth is in the cloud and we want to clean it there with SQL."

---

## Summary

| Situation | Option |
|-----------|--------|
| Default, "good enough" | Pandas chunked |
| Pandas too slow, one machine enough | Polars |
| Cleaning = SQL, avoid full load | DuckDB |
| One machine not enough | Spark / Dask / Ray |
| Data in cloud, SQL-only cleaning | BigQuery / Snowflake / Redshift |
| Same code, need more RAM/CPU | Batch job (bigger VM) |

**One-line:** Stay on pandas until it hurts (time or memory); then choose Polars (faster, same machine), DuckDB (SQL, same machine), or Spark/warehouse (scale out or cloud).
