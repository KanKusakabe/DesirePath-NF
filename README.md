# DesirePath-NF — 歩行動線から「望まれた道」を読む

**▶ GitHub Pages**: https://kankusakabe.github.io/DesirePath-NF/ ・ **統括**: [KAN-NF](https://kankusakabe.github.io/KAN-NF/)

屋外の自由歩行（ETH/UCY）の一歩 `p(d_forward, d_lateral | 文脈)` を条件付き Normalizing Flow で学習し、
**正確な尤度**で、混雑コリドー・迷いの場所・「まっすぐ目的地」からの系統的なズレ＝**けもの道(desire path)の芽**を数値化する。

## 芯となる考え

人の動線は「与えられた構造（provided：舗装・障害物・出入口）」に従いつつ、**"本当はこう歩きたい" という望まれた構造（desire）**を漏らす。
条件付き Flow の尤度は、その **desire（実際の流れ）と provided（目的地への測地線）の差**を測る物差しになる。
ズレが大きい場所こそ、環境が流れを曲げている＝**けもの道の候補**。

## データ

- **ETH/UCY** 歩行者軌跡 5 シーン（`seq_eth, seq_hotel, zara01, zara02, students03`、[OpenTraj](https://github.com/crowdbotp/OpenTraj) 経由の `obsmat`）。
- 平面座標 `(x,y)` → 自己中心の一歩増分 `(d_forward, d_lateral)`。約 3.9 万ステップ。
- **正直な範囲**：ETH/UCY に舗装マップの正解は無い。provided は「目的地への直進」で代用し、desire path は**測地線からの系統的偏差の代理**として示す（実舗装との照合には俯瞰＋レイアウトを持つ SDD 等が要る）。

## モデル

`(d_forward, d_lateral)` の zuko 条件付き Neural Spline Flow。文脈＝直近 8 ステップ増分の GRU 要約 ＋
`(正規化位置・目的地方位 sin/cos・残距離・シーン埋め込み)` の MLP。motionsim / MERL-NF と同じ実装様式。

## 結果（Leave-One-Scene-Out, held-out NLL, 小さいほど良い）

| モデル | NLL | 意味 |
|---|---|---|
| **NF（位置条件つき）** | **0.85** | 位置＝コリドー構造を使うと最も当たる |
| NF（位置ブラインド） | 0.97 | 位置を隠すと悪化＝**構造が効く証拠** |
| 直進ガウス（測地線） | 3.50 | 「まっすぐ目的地」では説明できない |
| 無条件ガウス | 3.08 | 素朴基線 |

- desire と直進の平均偏差 **0.70 rad ≈ 40°**、直進基線に対する尤度改善 平均 **−2.4 nats** ＝ desire ≠ provided の大きさ。
- **正直な例外**：小シーン `hotel` のみ位置ブラインドが勝った（0.77 vs 2.14）。配置が学習4シーンと大きく異なり過適合した1件で、シーン数の少ない LOSO の限界も示す。

## 実行

```bash
uv run python run_all.py   # 抽出 → LOSO 評価 → 読み出し → 図/metrics.json 再生成
```

出力：`reports/figures/{readouts,metrics}.png`, `results/metrics.json`, `docs/`（Pages）。

## この先（逆設計）

①サプライズ地図（迷い・混雑の骨格）②what-if（配置を変えて流れを再生成）③逆設計（偏差最小の歩道＝"けもの道の舗装案"を逆算）。
本実験は①と③の芽（偏差場）まで。実舗装マップを持つ俯瞰データに載せ替えれば②③を「園路ネットワークの引き直し」として完成できる。

KAN-NF 空間動線シリーズ。姉妹: [RouteDev-NF](https://kankusakabe.github.io/RouteDev-NF/)（ナビ逸脱）, [CityFlow-NF](https://kankusakabe.github.io/CityFlow-NF/)（都市回遊・立地）。
