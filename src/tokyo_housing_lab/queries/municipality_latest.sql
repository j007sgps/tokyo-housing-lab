-- 購買模式：資料裡最新的 4 季（滾動 1 年）合計，市區町村 × 價格種類的 ㎡ 單價中位數
--   「最新 4 季」是每種價格各自的最新 4 季（取引和成約公開的時間可能不同）
WITH t AS (
    SELECT *, year * 4 + quarter - 1 AS qi
    FROM transactions
    WHERE starts_with(city, $region)
      AND property_type = '中古マンション等'
      AND unit_price_per_m2_status = 'ok'
),
latest AS (
    SELECT price_classification, max(qi) AS last_qi FROM t GROUP BY price_classification
)
SELECT
    city,
    any_value(municipality) AS name,
    price_classification,
    count(*) AS n,
    median(unit_price_per_m2) AS median_unit_price,
    1.4826 * mad(ln(unit_price_per_m2)) AS spread_log,
    any_value(latest.last_qi) - 3 AS first_qi,
    any_value(latest.last_qi) AS last_qi
FROM t JOIN latest USING (price_classification)
WHERE t.qi > latest.last_qi - 4
GROUP BY city, price_classification
ORDER BY city, price_classification;
