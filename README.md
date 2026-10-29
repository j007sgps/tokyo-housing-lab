# tokyo-housing-lab

東京中古公寓行情分析工具（個人學習用）。資料來源為國土交通省的公開 API（XIT001）。

目前進度：M1（立川市 → DuckDB）完成；資料擴大到東京都全域（2005Q3〜最新季），並加上市區町村地圖（趨勢／購買／投資模式）、
品質調整指數、時點修正、車站比較（M4）、淹水（洪水、高潮）與學區的圖層（M3），以及估價（M5，只給區間）。

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
uv run tokyo-housing-lab fetch-stations                    # 成交的最寄駅與車站資料（圖磚 API，第一次約 1 小時）→ data/raw/tiles/
uv run tokyo-housing-lab analyze buy --by station          # 各車站的典型物件價格排行（要先 fetch-stations、load）
uv run tokyo-housing-lab analyze trend --by station --line JR中央線   # 一條路線上各站，依站的順序
uv run tokyo-housing-lab fetch-hazards                     # 洪水、高潮、小學校區、中學校區（圖磚 API，第一次約 1 小時）→ data/raw/tiles/
uv run tokyo-housing-lab hazards                           # 各區面積中淹水的比例（要先 fetch-hazards、load）
uv run tokyo-housing-lab hazards --by station --city 13108 # 江東區各車站周邊 500m 的淹水比例、車站所在的學區
uv run tokyo-housing-lab estimate --city 新宿 --district 西新宿 --area 60 --built 2005   # 估價：50%／80% 區間＋類似的過去成交（約 20 秒）
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
- **顯示**可切換 行政區／**車站**／**路線**（有跑過 `fetch-stations` 才有）：
  - 車站：每站一個圓，大小是 2016 年以來的成交筆數；可拖曳移動、Ctrl＋滾輪或兩指縮放；點圓看這站的走勢、和所在區的比較
  - 路線：選一條路線，地圖上畫出這條線，下方的路線圖依站的順序畫出各站的值與 95% 區間（背景色帶是所在的區）
- **圖層**可疊上 洪水（想定最大規模）／高潮／小學校區／中學校區（有跑過 `fetch-hazards` 才有）：
  - 淹水依浸水深上色（「水害ハザードマップ作成の手引き」的 6 級標準色）；行政區模式疊淹水時，區不依價格上色
  - 學區畫界線，滑鼠移上去看校名；行政區模式下點學區會選取它所在的區
  - 區的面板多了「區的面積中」淹水的比例；車站的面板多了「車站周邊 500m」的淹水比例和車站所在的學區
- **估價**卡片：選區、町名，填面積、建築年、房型、構造、改裝，在瀏覽器裡直接算出取引、成約水準的 50%／80% 區間
  （和 `estimate` 指令的係數、算法相同；點地圖上的區會換成那一區）

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
ln(㎡ 單價) = 町名 + 市區町村×年 + 面積 + 面積×年 + 屋齡 + 舊耐震 + 房型 + 構造 + 改裝 + 用途地域 + 容積率
              + 成約×市區町村 + 成約×年 + 誤差
```

- 「市區町村×年」就是指數。典型物件是 55㎡・屋齡 17 年・RC・2LDK・未改裝（東京的中位數條件），地段依該區所有成交的分布平均
- 面積、屋齡用分段線性；誤差用以町名為 cluster 的 cluster-robust 共變異矩陣；殘差超過 5σ 的極端值去掉再估一次
- 「面積×年」＝ ln(面積/55㎡) × 年（全東京共用）：面積的效果會隨時間變。2016 年小坪數（<30㎡）每㎡ 比全期共用的係數
  貴 7%，2025 年反而便宜 8%（2021 年起大坪數相對變貴，只用取引資料也一樣）。沒有這一項時，指數其實是
  「該區賣出物件的平均坪數」的漲幅，小坪數多的都心區會低估；加上後才是 55㎡ 的漲幅
  （都心區 2016→2025 多 15〜24 個百分點，多摩地區幾乎不變；全部區的中位數 +6 個百分點）
- 調停・競売等、関係者間取引不用；取引或成約不到 30 筆的區不放成約資料
- 購買模式的價格另外估一個「以最新一季為終點、每 4 季一期」的模型（最近 11 期），和中位數的最新 4 季對齊
- 驗證：隨機留 10% 預測，誤差中位數 13.3%（同町名同年的中位數是 21.4%）；和「相鄰兩年各估一次再串起來」的
  鏈結指數，2006→2025 的漲幅只差約 1%（每 5 年一組全部係數的版本反而在時期交界會跳，沒有採用）
- 限制：API 沒有車站距離和樓層；面積以外的特性效果全東京、全期間共用；「屋齡 17 年」在不同年份是不同年代蓋的，
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
- `factor` 是 55㎡ 的係數；面積的效果隨時間變（見上一節），面積 A 的物件再乘 `(A / 55) ^ area_elasticity`
  （每季一個值，全東京共用。例：2016Q1 是 +0.24，25㎡ 再 ×0.83、80㎡ 再 ×1.10）。用法（join `city, year, quarter`）：

  ```sql
  SELECT t.*, t.unit_price_per_m2 * a.factor * power(t.area_m2 / 55, a.area_elasticity) AS adjusted_unit_price
  FROM transactions t JOIN time_adjustment a USING (city, year, quarter)
  ```

- 驗證：拿最新 4 季的成交當答案，用同町名、同屋齡帶、同面積帶的前 5 年成交中位數來估 —
  不修正時平均低估 18%（誤差中位數 27%），修正後偏差 +4%（誤差中位數 15%）。
  沒有面積修正時整體偏差雖然只有 +1%，但那是小坪數高估 14%、大坪數低估 4% 互相抵銷；加上後各面積帶都在 +3〜7%
- 區間只算 55㎡ 係數本身的誤差，沒算面積彈性的誤差
- 時點修正只換算「時間」，不處理物件組成：同一季賣的物件不同，修正後的中位數一樣會跳動
- 新資料進來（`fetch` → `load`）後要重新執行 `adjust`，目標期會跟著移到新的最新 4 季

### 車站（站點比較）

XIT001 沒有車站欄位。成交點版本的 API（XPT001，圖磚）把每筆成交放在**最寄駅**的位置上，
用它對回 XIT001 的成交，就知道每筆成交的最寄駅；車站的名稱、路線、乘降人數用 XKT015（国土数値情報 S12 駅別乗降客数）。

- `fetch-stations`：東京本島往外多一圈的 z = 11 圖磚（35 個）× 2016 年起每年一個請求，原始回應 gzip 存到 `data/raw/tiles/`
  （z = 10 不支援；一個圖磚一年約 20MB）。已有的跳過，今年和去年每次重抓
- `load` 時建 `station`（依 S12_001g 把各路線合成一站）、`station_line`（路線上的順序）、`transaction_station`（成交 → 車站）三張表
- 對應方式：先用 (季, 價格種類, 町丁目, 總價, 面積, 建築年, 房型, 構造) 比對，對不上的再用前 6 個；
  同一個鍵有好幾筆而且最寄駅不同時不對。2016 年以來對上 99.3%
- 每筆成交先用品質調整模型換算成典型物件（扣掉面積、屋齡、房型、構造、改裝、成約的效果；町名、用途地域、容積率等地點的差異保留），
  再依最寄駅分組，取引和成約一起用：
  - 購買：最近 12 季的成交，用時點修正換算成最新 4 季的水準後取中位數（只看 4 季的話大部分車站筆數不夠）
  - 趨勢、投資：每年的中位數，和區一樣的算法與門檻（30 筆）。2020 年以前只有取引，地圖的趨勢模式預設從 2021 年比
  - 所在區＝成交最多的區；面板上「區整體」的數字用同樣的方法算（和行政區地圖上的品質調整指數不完全相同）
- 驗證：標準化前，同一站內價格和屋齡的相關是 −0.75，標準化後 −0.08；面積的相關 −0.03（加上「面積 × 期間」後）
- 路線上的順序沒有現成資料：用位置推算（多個起點＋2-opt＋搬一站，取總長最短），大江戸線的 6 字形也排得對
- 限制：
  - API 只給最寄駅，**沒有到車站的距離**，同一站的物件不分遠近
  - 新開的車站可能還沒收錄；乘降人數各公司定義不同（JR 只算上車人數）
  - 国土数値情報的「山手線」只有品川〜新宿〜田端一側，東京・上野一側算在東海道線、東北線
  - 區間只算抽樣誤差，沒算標準化和時點修正係數本身的誤差
- 車站資料：「国土数値情報（駅別乗降客数データ）」（国土交通省）を加工して作成
- 計算方式寫在 `src/tokyo_housing_lab/analysis/stations.py`、`src/tokyo_housing_lab/processing/stations.py`

### 估價（M5）

輸入一間中古マンション的條件，估「現在（最新 4 季）同樣條件的物件，成交價會落在哪裡」。**只給區間，不給單一數字**。

```bash
uv run tokyo-housing-lab estimate --city 新宿 --district 西新宿 --area 60 --built 2005 [--plan 3LDK+S] [--structure SRC] [--renovated]
```

- 模型：車站分析共用的視窗模型（最近 11 年、每 4 季一期、加季內走勢；`src/tokyo_housing_lab/analysis/window.py`）
- ln ㎡ 單價 ＝ 區的現在水準（最新 4 季的平均）＋町名的地點效果（町名係數＋用途地域、容積率依該町名的成交平均）
  ＋面積、屋齡、面積×現在、舊耐震、房型、構造、改裝的效果（＋成約水準時，該區的成約效果）
  - 全部攤成一張係數表（`CoefTable`）；CLI 和地圖的 JS 都用它算，測試保證和模型的 `predict` 相同
- 區間：split conformal。最新 4 季的成交留 20% 不參加估計，用同樣的方法預測，「實際 ÷ 預測」的 25／75、10／90 分位數就是 50%、80% 區間（不對稱）
  - 分位數依「屋齡（10 年一段）× 23 區／多摩」分 10 組：全部共用一組時，屋齡 40 年以上的 80% 區間只蓋到 74%、10 年以內蓋到 86%、多摩 75%
  - 另外留 10% 檢驗（3,864 筆）：80% 區間蓋到 81.0%、50% 區間 51.8%；取引／成約、23 區／多摩、屋齡、面積的各群都在名目值 ±4 個百分點內
  - 寬度：新物件的 80% 區間約 −19%〜+30%，屋齡 40 年以上約 −31%〜+33%
- 類似的過去成交（CLI）：同町名、最近 3 年、面積 ±30%、建築年 ±10 年，依相似度取 8 筆。
  「換算成這間」＝這間的預測 × 那筆成交的殘差（時點、面積、屋齡、房型的差用模型的係數換算；樓層、方位等的差留在那筆自己的價格裡）
- 觀察：同一個町名、同樣條件，不同大樓可以差到 2 倍（例：西新宿 2005 年建 60㎡ 的成交分成約 200 萬／㎡ 和 105〜140 萬／㎡ 兩群），
  模型分不出是哪一棟。多摩的新物件模型估得偏低（分組的區間已經把位置移過去）：屋齡的效果全東京共用，多摩的新舊價差比 23 區大
- 限制：
  - 樓層、方位、離車站的距離、管理狀態、哪一棟大樓都不在模型裡，所以區間很寬
  - 區間是同樣條件的物件的成交價範圍，不是這間的價值；只是過去成交資料的整理，不是鑑價
  - 同一筆成交可能同時出現在取引和成約裡（API 沒有交易 ID）
- 計算方式在 `src/tokyo_housing_lab/analysis/estimate.py`

### 淹水、學區（圖層）

| 圖層 | API | 縮放 | 內容 |
|---|---|---|---|
| 洪水 | XKT026 洪水浸水想定区域（想定最大規模） | z14〜15 | 各河川的浸水想定區域，`A31a_205` 浸水深ランク |
| 高潮 | XKT027 高潮浸水想定区域 | z13〜 | 浸水深是文字分級（「0.3m以上0.5m未満」） |
| 小學校區 | XKT004 小学校区（令和 5 年度） | z11〜 | 一校一個通學區域 |
| 中學校區 | XKT005 中学校区（令和 5 年度） | z11〜 | 同上 |

- `fetch-hazards`：和東京都本島 53 個市區町村相交的圖磚（洪水 552、高潮 156、學區各 17 個），原始回應 gzip 存到 `data/raw/tiles/`（約 150MB）。
  已有的跳過；官方一年更新一次，要更新時用 `--force`。`--only flood surge` 可以只抓一部分
- 浸水深ランク依國土數值情報的官方代碼表（`water_depth_code.xlsx`）：1＝0〜0.5m未満、2＝0.5〜3.0m、3＝3.0〜5.0m、4＝5.0〜10.0m、5＝10.0〜20.0m、6＝20.0m以上
- `load` 時：
  - 洪水、高潮畫到 Web Mercator 縮放 20 的格網（東京約 31m 一格），每格取最大的浸水深等級（同一個地點常有好幾條河川的想定重疊），
    只留都內（地圖的行政區界）→ `hazard_grid`（每列的連續區段）
  - 學區的多邊形會超出圖磚約 40m，切齊圖磚的邊界後依學校代碼拼成一筆 → `school_district`（邊界線另外存，去掉圖磚邊界上的邊）
  - 幾何運算（掃描線填色、點在多邊形內、切邊、簡化）用 numpy 自己寫：`src/tokyo_housing_lab/processing/raster.py`
- `hazards`：浸水深 ≥0.5m（淹到地板以上）、≥3m（淹到 2 樓）的比例。區＝區的面積中；車站＝車站座標半徑 500m 內
  （都界附近的車站只算都內的部分，不到一半的不算）。計算方式在 `src/tokyo_housing_lab/analysis/hazards.py`
- 檢查：淹水比例最高的是荒川、足立、墨田、葛飾、江戶川區（洪水 ≥0.5m 77〜93%）；高潮的浸水面積 187km²，和東京都公布的想定（約 212km²）同一個量級
- 和價格並列看（只當參考）：同一區內，周邊洪水 ≥0.5m 的比例 100% 的車站，比 0% 的便宜約 12%（相關 −0.21）。
  但低地和高台的差別不只是淹水（街區的性質、新開發的埋立地…），這不是「淹水讓房價打幾折」
- 限制：
  - 成交只知道最寄駅，所以是**車站周邊**的指標，不是每筆物件的
  - 洪水是想定最大規模（千年一次程度），不含內水氾濫；行政區界是簡化版，邊界附近有約 100m 的誤差
  - 學區：品川區、荒川區國土數值情報沒有資料；可能和現在的指定不同；很多區的中學有學校選擇制。學區資料的利用條件依市區町村而不同，
    東京都本島（品川、荒川以外）都是「オープンデータ公開・特記條件なし」
- 資料：「国土数値情報（洪水浸水想定区域データ）」「国土数値情報（高潮浸水想定区域データ）」「国土数値情報（小学校区データ）」
  「国土数値情報（中学校区データ）」（国土交通省）を加工して作成

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
- `station`、`station_line`、`transaction_station`：車站（見上面）
- `hazard_grid`、`school_district`：淹水格網、學區（見上面）

欄位的實際觀察與清理規則的依據：`docs/xit001_fields.md`

## 部署（Vercel）

地圖公開在 https://tokyo-housing-lab.vercel.app （Vercel 專案 `tokyo-housing-lab`，Hobby 方案）。

- 內容來自 git 的 **`vercel-site` 分支**：沒有歷史的獨立分支，只放 3 個靜態檔，原始碼不在裡面
  - `index.html`：`tokyo-housing-lab map` 產生的單檔地圖，`<head>` 加上 `<meta name="robots" content="noindex, nofollow">`
  - `robots.txt`：`User-agent: *` ／ `Disallow: /`
  - `vercel.json`：所有回應加上 `X-Robots-Tag: noindex, nofollow` 標頭
  - 三層都是請搜尋引擎不要收錄（正規的搜尋引擎會遵守，但擋不了不守規矩的爬蟲）；知道網址的人都看得到
- Vercel 專案**沒有連 GitHub 自動部署**：連了之後，每個分支的 push 都會建立預覽部署，可能把 repo 的原始碼放到網路上。
  所以每次更新都要手動建立部署
- 資料庫和原始資料不在 git 裡（`data/` 不進版控），Vercel 沒辦法自己重建地圖：資料更新後要重新產生地圖、再部署一次

更新步驟：

```bash
# 1. 重新產生地圖（要先 fetch／load 好）
uv run tokyo-housing-lab map --out /tmp/site/index.html

# 2. 加上 noindex（放在 <meta name="viewport" 前面）
python3 -c "
import pathlib; p = pathlib.Path('/tmp/site/index.html'); s = p.read_text(encoding='utf-8')
p.write_text(s.replace('<meta name=\"viewport\"', '<meta name=\"robots\" content=\"noindex, nofollow\">\n<meta name=\"viewport\"', 1), encoding='utf-8')"

# 3. commit 到 vercel-site 分支（另開一個 worktree，不影響手上的分支）
git fetch origin vercel-site
git worktree add /tmp/vercel-site origin/vercel-site
cd /tmp/vercel-site && git switch -C vercel-site origin/vercel-site
cp /tmp/site/index.html . && git add index.html && git commit -m "部署用：更新地圖" && git push origin vercel-site
cd - && git worktree remove /tmp/vercel-site
```

4. 建立 production 部署：Vercel 的 dashboard → 專案 `tokyo-housing-lab` → Deployments → Create Deployment，
   選 GitHub 的 `j007sgps/CloudDeveloper`、分支 `vercel-site`；或用 Vercel API 的 create deployment
   （`gitSource: {type: "github", org: "j007sgps", repo: "CloudDeveloper", ref: "vercel-site"}`、`target: "production"`）。
   專案設定是 Framework 「Other」、不建置（Build／Install Command 空白、Output Directory `.`）
5. 用無痕視窗打開 https://tokyo-housing-lab.vercel.app 確認看得到

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
