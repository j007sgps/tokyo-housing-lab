# tokyo-housing-lab

東京中古公寓行情分析工具（個人學習用）。資料來源為國土交通省的公開 API（XIT001）。

目前進度：M1（立川市 → DuckDB）完成；資料擴大到東京都全域（2005Q3〜最新季），並加上市區町村地圖（趨勢／購買／投資模式）。

## 環境準備

```bash
cd tokyo-housing-lab
uv sync
cp .env.example .env   # 再把 REINFOLIB_API_KEY 換成自己的金鑰
```

金鑰也可以直接用環境變數 `REINFOLIB_API_KEY` 提供（環境變數優先於 `.env`）。

## 使用方式

```bash
uv run tokyo-housing-lab fetch   # 打 API，東京都全域 2005Q3〜 的原始 JSON 存到 data/raw/（已有的跳過，有資料的最新 4 季重抓）
uv run tokyo-housing-lab load    # 只讀 data/raw/，清理後寫進 data/tokyo_housing_lab.duckdb（不打 API）
uv run tokyo-housing-lab query   # 每季 ㎡ 單價中位數與筆數（中古マンション等）
uv run tokyo-housing-lab analyze buy                       # 各區最新 4 季的 ㎡ 單價中位數排行
uv run tokyo-housing-lab analyze trend --from 2015 --to 2025  # 各區起訖年的漲幅排行
uv run tokyo-housing-lab analyze invest                    # 各區最近 10 年的年成長率排行
uv run tokyo-housing-lab analyze trend --price combined    # 用綜合指標（取引＋成約）；01＝取引（預設）、02＝成約
uv run tokyo-housing-lab analyze trend --price hedonic     # 用品質調整指數（要估模型，十幾秒）
uv run tokyo-housing-lab hedonic --validate                # 品質調整模型的係數、配適度、預測誤差
uv run tokyo-housing-lab adjust                            # 時點修正係數 → DuckDB 的 time_adjustment 表（約 20 秒）
uv run tokyo-housing-lab query --city 13202 --adjusted     # 每季中位數，多一欄換算成現在水準的中位數
uv run tokyo-housing-lab map     # 產生 data/tokyo_map.html，用瀏覽器打開
```

### 地圖

`data/tokyo_map.html` 是一個單檔 HTML（資料和邊界都嵌在裡面，只有 d3 從 cdnjs 載入）：

- **趨勢模式**：選起年和迄年，看各區 ㎡ 單價中位數漲了多少（紅＝漲、藍＝跌）
- **購買模式**：最新 4 季的 ㎡ 單價中位數，顏色越深越貴
- **投資模式**：最近 10 年的年成長率（過去走勢的整理，不是預測）
- 點地圖上的區或排行裡的名字，右側（手機在下方）會顯示詳細資料：最新價格、每年的中位數走勢、建築年代別中位數
- 價格種類可切換 **品質調整**（預設）／綜合／取引價格／成約價格；投資模式不能選成約（不到 10 年）
- 灰色＝樣本不足（一年少於 30 筆）；斜線＝區間包含 0（看不出漲跌）或區間太寬（不確定）

### 綜合指標（取引＋成約）

取引價格從 2005 年起、歷史長；成約價格（透過仲介的成交）只有 2021 年起。兩者的漲跌很一致，
但同一區、同一年的價格水準有落差（各區不同，實測 −3%〜+53%），不能把成交直接混在一起算。綜合指標的做法：

1. 每區用 2021 年以後兩邊都有 30 筆以上的年份（至少 3 年），估計「取引比成約高多少」。
   用 random-effects（DerSimonian–Laird），落差每年的變動（τ²）也一起估計
2. 把成約換算成取引的水準，每年和取引依誤差大小加權平均。落差越不穩定，成約的權重越小；2020 年以前只有取引
3. 誤差以「各觀測值的線性組合」精確計算，換算本身的誤差也算進去
4. 三種模式都能用：趨勢（綜合序列的漲幅）、購買（以取引水準表示的最新價格）、投資（綜合序列 10 年的最小平方法）

- 重疊年不足的區，綜合就是取引（面板上會註明）
- 落差每年的差異大到抽樣誤差解釋不了的區（Q 檢定 5%），詳細面板會提醒：兩者走勢可能不同，請分開看
- 已知限制：同一筆成交可能同時出現在取引和成約裡（API 沒有交易 ID，無法去除重複），綜合的區間會稍微偏窄
- 計算方式寫在 `src/tokyo_housing_lab/analysis/combined.py`

### 品質調整指數（hedonic 迴歸）

中位數會受「那年賣了什麼物件」影響。例如都心各區賣出物件的屋齡中位數，從 2006 年的 6 年左右變成 2025 年的 20 年左右，
舊物件每㎡ 比較便宜，中位數就低估了漲幅。品質調整指數用迴歸把這些因素拆開，只看「同樣條件的物件」價格怎麼變：

```
ln(㎡ 單價) = 町名 + 市區町村×年 + 面積 + 屋齡 + 舊耐震 + 房型 + 構造 + 改裝 + 用途地域 + 容積率
              + 成約×市區町村 + 成約×年 + 誤差
```

- 「市區町村×年」就是指數。典型物件是 55㎡・屋齡 17 年・RC・2LDK・未改裝（東京的中位數條件），地段依該區所有成交的分布平均
- 面積、屋齡用分段線性；誤差用以町名為 cluster 的 cluster-robust 共變異矩陣；殘差超過 5σ 的極端值去掉再估一次
- 調停・競売等、関係者間取引不用；取引或成約不到 30 筆的區不放成約資料
- 購買模式的價格另外估一個「以最新一季為終點、每 4 季一期」的模型（最近 11 期），和中位數的最新 4 季對齊
- 驗證：隨機留 10% 預測，誤差中位數 13.7%（同町名同年的中位數是 21.2%）；係數全期間共用的版本，
  和「相鄰兩年各估一次再串起來」的鏈結指數只差約 2%（每 5 年一組係數的版本反而在時期交界會跳，沒有採用）
- 限制：API 沒有車站距離和樓層；特性的效果全東京、全期間共用；「屋齡 17 年」在不同年份是不同年代蓋的，
  新一代建物本身品質較好的部分也會算進漲幅；同一筆成交可能同時在取引和成約裡
- 計算方式寫在 `src/tokyo_housing_lab/analysis/hedonic.py`

### 時點修正

把舊的成交價換算成「最新 4 季的平均水準」（例：中央区 2020Q1 的成交 × 1.77）。用品質調整模型，期間設成
「以最新一季為終點、每 4 季一期」，再加上全東京共用的季內走勢（2025 年光是一年內就漲了約 6%，只修到年不夠）：

```
ln(係數) = [區×期間₀ ＋ 期間₀ 內 4 季的平均] − [區×期間_w ＋ 期間_w 內第 k 季]
```

- `adjust` 估一次模型，把每個「區 × 季」（2005Q3〜最新一季）的係數和 95% 區間寫進 `time_adjustment` 表（整張重寫）
- 該區那一期或目標期不到 30 筆 → 用全東京的係數（`basis = 'tokyo'`），區間另外加上各區走勢的離散
- 用法：`unit_price_per_m2 × factor`（join `city, year, quarter`）。例：

  ```sql
  SELECT t.*, t.unit_price_per_m2 * a.factor AS adjusted_unit_price
  FROM transactions t JOIN time_adjustment a USING (city, year, quarter)
  ```

- 驗證：拿最新 4 季的成交當答案，用同町名、同屋齡帶、同面積帶的前 5 年成交中位數來估 —
  不修正時平均低估 17%（誤差中位數 25%），修正後偏差 +2%（誤差中位數 16%）
- 時點修正只換算「時間」，不處理物件組成：同一季賣的物件不同，修正後的中位數一樣會跳動
- 新資料進來（`fetch` → `load`）後要重新執行 `adjust`，目標期會跟著移到新的最新 4 季

邊界資料 `src/tokyo_housing_lab/map/tokyo_municipalities.geojson` 由 `scripts/prepare_boundaries.py` 產生：
「国土数値情報（行政区域データ）」（国土交通省）を加工して作成（スマートニュース メディア研究所 japan-topography の簡略化データを使用）。

`analyze` 的數字都附 95% 區間；一年（或最新 4 季）少於 30 筆的區標成「樣本不足」，
區間包含 0 或太寬的標成「看不出漲跌／不確定」。計算方式寫在 `src/tokyo_housing_lab/analysis/trend.py`。
這些只是過去成交資料的整理，不是價格預測，也不是投資建議。

- `fetch --region 13202`：只抓某個市區町村（2 碼＝都道府縣、5 碼＝市區町村，預設 13 東京都全域）
- `fetch --force`：全部重抓；`fetch --refresh-recent 0`：不重抓最新幾季；`fetch --start 2023Q1`：改起始季
- 成約價格只有 2021Q1 以後、取引價格只有 2005Q3 以後的資料，更早的季不會請求
- 原始檔存成 `.json.gz`（解壓後和 API 回傳的 bytes 完全相同）；東京都全域約 25MB
- `load` 每次都把 `data/raw/` 整個重新載入，跑幾次結果都一樣；任何一個檔案有問題就整批不寫入
- 同一季已經有東京都全域的檔案時，該季的市區町村檔案（例如 M1 的立川市）會被跳過，不會重複計算
- `query --city 13104`：查其他市區町村（例：新宿区）

查詢用的 SQL 在 `src/tokyo_housing_lab/queries/quarterly_median.sql`，也可以直接用 DuckDB 開資料庫自己查：

```bash
uv run python -c "import duckdb; print(duckdb.connect('data/tokyo_housing_lab.duckdb', read_only=True).sql('SELECT count(*) FROM transactions'))"
```

資料表：
- `transactions`：中古マンション等的成交資料。數值欄位都有 `<欄位>_raw`（原字串）、`<欄位>`（數值）、`<欄位>_status`（ok / empty / sentinel / unparsable）
- `fetch_log`：每次打 API 的紀錄（含 404 的「該期無資料」）

欄位的實際觀察與清理規則的依據：`docs/xit001_fields.md`

## 測試

```bash
uv run pytest            # 單元測試 + 用 fixture 的整條管線測試（不會打 API）
uv run pytest -m manual  # 真的打 API 的整合測試（需要金鑰與網路）
```

## 資料目錄

- `data/raw/`：API 回傳的原始 JSON（不進 git）
- `data/*.duckdb`：清理後的資料庫（不進 git，可由 raw 重建）

---

このサービスは、国土交通省の不動産情報ライブラリのAPI機能を使用していますが、提供情報の最新性、正確性、完全性等が保証されたものではありません
