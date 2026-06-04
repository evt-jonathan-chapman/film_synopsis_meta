WITH params AS (
    SELECT
        DATE '2018-01-01'              AS pre_covid_start,
        DATE '2020-02-01'              AS pre_covid_end,
        DATE '2021-12-01'              AS post_covid_start,
        DATE '2018-01-01'              AS rel_at_start,
        CURRENT_DATE                   AS rel_at_end,
        DATEADD(month, -4, CURRENT_DATE) AS end_date
),

film_release AS (
    SELECT
        ti.TITLE_GLOBAL_ID,
        ti.NAME AS FILM_NAME,
        ti.UPPER_NAME,
        ti.TITLE_AKA,
        ti.US_TITLE_NAME,
        ti.SHORT_NAME,
        ti.SYNOPSIS,
        ti.PRIM_CTGY_GLOBAL_ID,
        ti.IS_ALT_CONTENT,
        ti.LAST_CHNG,
        ti.ORIG_CNTRY,
        fl.CNTRY_ID,
        fl.STATE_GLOBAL_ID,
        fl.DISTR_GLOBAL_ID,
        fl.CCY_TYPE_NO,
        fl.CCY_ID,
        MIN(fl.EXHBTN_DATE) AS RELEASE_DATE
    FROM EDW_ENT_PRD.CURATED.IBOE_TITLES AS ti
    JOIN EDW_ENT_PRD.CURATED.IBOE_FLASH_GROSS_STATE_TITLE AS fl
        ON fl.TITLE_GLOBAL_ID = ti.TITLE_GLOBAL_ID
    WHERE fl.REL_WK_NO = 1
    GROUP BY
        ti.TITLE_GLOBAL_ID,
        ti.NAME,
        ti.UPPER_NAME,
        ti.TITLE_AKA,
        ti.US_TITLE_NAME,
        ti.SHORT_NAME,
        ti.SYNOPSIS,
        ti.PRIM_CTGY_GLOBAL_ID,
        ti.IS_ALT_CONTENT,
        ti.LAST_CHNG,
        ti.ORIG_CNTRY,
        fl.CNTRY_ID,
        fl.STATE_GLOBAL_ID,
        fl.DISTR_GLOBAL_ID,
        fl.CCY_TYPE_NO,
        fl.CCY_ID
)

SELECT fr.*
FROM film_release AS fr
CROSS JOIN params AS p
WHERE (
    fr.RELEASE_DATE BETWEEN p.pre_covid_start AND p.pre_covid_end
    OR
    fr.RELEASE_DATE BETWEEN p.post_covid_start AND DATEADD(day, 28, p.rel_at_end)
)
AND CNTRY_ID = 'AU'
ORDER BY fr.RELEASE_DATE DESC

