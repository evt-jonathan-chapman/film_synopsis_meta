-- Ported to comscore_gower_job.py::gower_export_sql as a Python-side copy — keep in sync
-- (that copy escapes '%' as '%%' for Python string formatting; this file must not).
--
-- Reverted 2026-09-09 from DBT.EDW_ENT_PRD.CINR_TBM_GW_LIFE_TIME back to the raw
-- ENT_FORECAST_PRD.CURATED.GW_LIFE_TIME table (the pre-2026-08-12 source, before
-- commit 76571d9 "swapped" it to DBT "per request"). Confirmed via
-- MAX(run_date) on the DBT copy = 2026-03-02: that dbt model stopped refreshing
-- over 6 months ago, while the raw table keeps updating daily. Every Gower match
-- since the swap was scoring against that frozen March snapshot — caught two
-- concrete cases on 2026-09-09: "Practical Magic 2" still showing as "An
-- Untitled Practical Magic Film" (title not yet finalised as of March), and
-- "Sense and Sensibility"'s rel_date stuck at its old (since-changed) estimate.
-- Column names differ on the raw table: PRMRY_TITLE_NO (not PRIM_TITLE_NO) and
-- a real SNAPSHOT_DATE column already exists directly (the DBT copy only had
-- RUN_DATE, hence the old "AS snapshot_date" rename this file used to do).
WITH params AS (
    SELECT
        DATE '2025-01-01' AS start_date
)

SELECT
gw.* EXCLUDE(job_name, src_obj_name, prmry_title_no),
gw.prmry_title_no AS prim_title_no,
CASE
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY gw.snapshot_date DESC) = 1
        THEN 'latest'
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title
                            ORDER BY ABS(DATEDIFF(day, gw.snapshot_date, DATEADD(month, -1, gw.rel_date)))) = 1
        THEN '1m_pre_release'
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title
                            ORDER BY ABS(DATEDIFF(day, gw.snapshot_date, DATEADD(month, -3, gw.rel_date)))) = 1
        THEN '3m_pre_release'
END AS snapshot_type
FROM ENT_FORECAST_PRD.CURATED.GW_LIFE_TIME AS gw
CROSS JOIN params AS p
WHERE gw.ter_id = 'AU'
AND gw.rel_date >= p.start_date
AND gw.title NOT ILIKE '%untitled%'
AND gw.snapshot_date <= DATEADD(day, -1, gw.rel_date)
QUALIFY
ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY gw.snapshot_date DESC) = 1
OR ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY ABS(DATEDIFF(day, gw.snapshot_date, DATEADD(month, -1, gw.rel_date)))) = 1
OR ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY ABS(DATEDIFF(day, gw.snapshot_date, DATEADD(month, -3, gw.rel_date)))) = 1
order by rel_date desc