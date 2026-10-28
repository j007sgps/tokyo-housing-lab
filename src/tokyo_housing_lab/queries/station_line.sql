-- 路線上的車站，依推算的順序
SELECT operator, line, station_id
FROM station_line
ORDER BY operator, line, seq;
