-- 每季的 ㎡ 單價中位數，以及「時點修正後」的中位數（換算成 time_adjustment.target 期間的水準）
--   修正後 = 單價 × 該區該季的係數（tokyo-housing-lab adjust 算的）；係數是 NULL 的成交不算進修正後的中位數
--   修正得好的話，修正後的中位數大致是平的（但仍會受每季賣出物件的組成影響）
SELECT
    t.year,
    t.quarter,
    t.price_classification,
    median(t.unit_price_per_m2) AS median_unit_price,
    median(t.unit_price_per_m2 * a.factor) AS median_adjusted,
    count(*) AS n,
    count(a.factor) AS n_adjusted,
    any_value(a.target) AS target,
    any_value(a.basis) AS basis
FROM transactions t
LEFT JOIN time_adjustment a USING (city, year, quarter)
WHERE t.city = $city
  AND t.property_type = '中古マンション等'
  AND t.unit_price_per_m2_status = 'ok'
GROUP BY t.year, t.quarter, t.price_classification
ORDER BY t.year, t.quarter, t.price_classification;
