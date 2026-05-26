WITH params AS (
    SELECT
        DATE '2018-01-01'              AS pre_covid_start,
        DATE '2020-02-01'              AS pre_covid_end,
        DATE '2021-12-01'              AS post_covid_start,
        DATE '2018-01-01'              AS rel_at_start,   -- Python substitutes train/test start
        DATE '2025-11-12'              AS rel_at_end,     -- Python substitutes train/test end
        DATEADD(month, -4, CURRENT_DATE) AS end_date      -- session-data completeness cutoff (unchanged)
),

cs_lookup as (
    select
    cs.*
    from EDW_ENT_PRD.SEMANTIC.VW_COMSCORE_BOX as cs
    cross join params as p
    where  (
          cs."Film Open Date" BETWEEN p.pre_covid_start AND p.pre_covid_end
          OR
          cs."Film Open Date" BETWEEN p.post_covid_start AND DATEADD(day, 28, p.rel_at_end)
      )
    and cs."Reporting Country" = 'Australia'
)

select * from cs_lookup

