-- Ported to comscore_gower_job.py::gower_export_sql as a Python-side copy — keep in sync
-- (that copy escapes '%' as '%%' for Python string formatting; this file must not).
WITH params AS (
    SELECT
        DATE '2025-01-01' AS start_date
)

SELECT
gw.* EXCLUDE(job_name, src_obj_name, run_date),
gw.run_date AS snapshot_date,
CASE
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY gw.run_date DESC) = 1
        THEN 'latest'
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title
                            ORDER BY ABS(DATEDIFF(day, gw.run_date, DATEADD(month, -1, gw.rel_date)))) = 1
        THEN '1m_pre_release'
    WHEN ROW_NUMBER() OVER (PARTITION BY gw.title
                            ORDER BY ABS(DATEDIFF(day, gw.run_date, DATEADD(month, -3, gw.rel_date)))) = 1
        THEN '3m_pre_release'
END AS snapshot_type
FROM DBT.EDW_ENT_PRD.CINR_TBM_GW_LIFE_TIME AS gw
CROSS JOIN params AS p
WHERE gw.ter_id = 'AU'
AND gw.rel_date >= p.start_date
AND gw.title NOT ILIKE '%untitled%'
AND gw.run_date <= DATEADD(day, -1, gw.rel_date)
QUALIFY
ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY gw.run_date DESC) = 1
OR ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY ABS(DATEDIFF(day, gw.run_date, DATEADD(month, -1, gw.rel_date)))) = 1
OR ROW_NUMBER() OVER (PARTITION BY gw.title ORDER BY ABS(DATEDIFF(day, gw.run_date, DATEADD(month, -3, gw.rel_date)))) = 1
order by rel_date desc