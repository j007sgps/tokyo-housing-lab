-- M1 完成條件：中古マンション，每一季、每種價格的 ㎡ 單價中位數與成交筆數
--   price_classification：01 = 取引價格、02 = 成約價格
--   unit_price_per_m2 = TradePrice ÷ Area（API 的 Area 已經四捨五入到 5㎡，單價會有誤差）
--   只用單價能算出來的資料（status = 'ok'）；無法計算的筆數另外列出，不默默丟掉
SELECT
    year,
    quarter,
    price_classification,
    median(unit_price_per_m2) FILTER (WHERE unit_price_per_m2_status = 'ok') AS median_unit_price,
    count(*) FILTER (WHERE unit_price_per_m2_status = 'ok') AS n,
    count(*) FILTER (WHERE unit_price_per_m2_status <> 'ok') AS n_excluded
FROM transactions
WHERE city = $city
  AND property_type = '中古マンション等'
GROUP BY year, quarter, price_classification
ORDER BY year, quarter, price_classification;
