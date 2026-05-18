# uav-nav-lab — Plan & Roadmap

> **位置付け**: `docs/findings.md` は *終わった研究の記録*、`README.md` は
> *入口とハイライト*、`CHANGELOG.md` は *バージョン毎の差分*、この
> `plan.md` は *これから何をやるか / なぜやるか / 引き継ぐ人が何を踏むか*
> をまとめる作戦ノート。
>
> 最終更新: 2026-05-18 (AirSim static-cube density sweep の途中経過を詳述)

---

## 0. Codex への引き継ぎ要点 (まずここを読む)

このドキュメントを引き継いだ codex への TL;DR。

### 0.1 リポジトリの今の状態

- **v0.2.0 タグ済み** (2026-05-17)。`CHANGELOG.md` に差分要約あり。
- v0.1.0 → v0.2.0 で **88 commit**。中身は §1 を参照。
- 論文ドラフト (`docs/paper_a/`) は §1〜§7 まで本文/appendix map あり。
- GitHub の About / README ヒーロー GIF / Roadmap 節は v0.2.0 公開時点で
  最新化済み。**README には触れずに研究を進める方が摩擦が少ない**。

### 0.2 直近で閉じた open question

**「AirSim 上で failure-level の planner 差を測れる discriminating cell
を作る」**は、`examples/exp_airsim_multi_discriminating_n30*.yaml`
で一旦閉じた。

dummy_3d n=100 の §3 ヘッドライン (`docs/findings.md` §"Multi-drone:
GPU MPPI's rollout cloud flips the coordination Δ") は、3 つの AirSim
セル (±2-4 m / ±1 m / 0 m altitude stagger, 各 n=30) に運んだところ:

- **±2-4 m / ±1 m**: 両 planner とも 4/4 joint で天井 (per-drone 100 %)
  → Δ の差は trajectory spread でしか観測できない (4-27× 比は出る、
  失敗レベルの Δ は出ない)。
- **0 m (uniform z=30)**: GPU MPPI 0/30 joint、MPC 14/30 joint
  (McNemar p ≈ 0.00012)。ただし GPU MPPI の per-drone が 28.3 % に
  落ちて indep⁴ が joint floor を割るため、ここでの Δ は §3 と同じ
  メカニズム (softmax 増幅) ではなく **mesh-mesh ボトルネックでの
  幾何的崩壊**。

2026-05-17 に Blocks 静的 cube を `simulator.static_obstacles` で
spawn する経路を追加し、n=30 paired を実走:

- MPC: per-drone 105/120 = 87.5 %, joint 22/30 = 73.3 %, Δ +14.7 pp
- GPU MPPI: per-drone 120/120 = 100 %, joint 30/30 = 100 %, Δ +0.0 pp
- McNemar: GPU-only success 8, MPC-only 0, p ≈ 0.008

→ AirSim に failure-level の差は作れた。ただし dummy_3d の
「joint tie で GPU MPPI の Δ が大きい」メカニズム再現ではなく、
この cell では **GPU MPPI が ceiling に逃げ切る**。次の open question は
「GPU MPPI も 60〜90 % per-drone 帯に落ちる static-cube density sweep を
作れるか」。

### 0.3 やる前に踏むトラップ集

1. **AirSim multi-drone reset の stale collision flag** —
   `client.reset()` 後に engine をすぐ pause しないと、`settle_after_reset`
   の間に 4 機が地面コリジョンを `simGetCollisionInfo().has_collided` に
   蓄積し、t=0 で 100 % joint collision に見える。
   `uav_nav_lab/sim/airsim_bridge.py` の `reset()` 内で
   `client.reset()` 直後に `client.simPause(True)` を呼んでいる
   (commit 382d207) — **絶対に消さないこと**。詳細は
   `docs/findings.md` §"AirSim bridge: pause-after-reset prevents
   stale t=0 collisions"。

2. **n=1 デモは bug を隠す** — `front_center` カメラを付けた demo は
   毎ステップ `simGetImages` RPC で readback を直列化し、上記の bug を
   masking する。n=30 study で camera を切ると surface する。**新規 YAML
   を書くときは camera 無しで動くことを先に確認**。

3. **AirSim multi-drone reset hang** — `client.reset()` を sequential
   に呼ぶと 1〜2 回で wedge する。AirSim サーバ側のディスパッチループの
   問題。`scripts/run_airsim_multi_chunked.sh` で **エピソード毎に
   Blocks server を再起動**して逃げている。upstream issue 化が
   open task (§3.2)。

4. **`pkill -f "Blocks"` は自殺する** — bash プロセスの command literal
   に "Blocks" が入るため自分も殺される。`/proc/<pid>/comm` を見て
   exact name で照合すること。chunked runner は対応済み。

5. **system python の matplotlib は壊れている** — `mpl_toolkits.mplot3d`
   が import 不能。**`uav-nav anim` 系は必ず `.venv/bin/python` で
   実行**。AirSim 実行 (msgpackrpc 必要) は system python。

6. **GPU MPPI の first-call cost ~14 s** — episode 毎に autograd graph を
   作り直すので、`plan_dt` の素朴な mean は steady-state の 10〜30×
   になる。`scripts/paired_analysis_*.py` 系は **first replan を捨てて
   steady-state mean / p95 を出す** convention になっている (`docs/paper_a/
   section_4_prerequisites.md` §4.3 参照)。

### 0.4 引き継ぎ用の最短再現手順

AirSim 4 機 n=30 paired を新しいセルで回す最短コマンド:

```bash
# 1. Blocks を一度立ち上げて settings.json (Drone1..Drone4) を確認
~/AirSim/Blocks/LinuxNoEditor/Blocks.sh &

# 2. chunked runner で MPC / GPU MPPI それぞれ n=30
scripts/run_airsim_multi_chunked.sh mpc      30 0 results/airsim_xxx_mpc      examples/exp_airsim_xxx.yaml
scripts/run_airsim_multi_chunked.sh gpu_mppi 30 0 results/airsim_xxx_gpu_mppi examples/exp_airsim_xxx_gpu_mppi.yaml

# 3. Wilson + McNemar 集計
python scripts/paired_analysis_airsim_multi.py results/airsim_xxx_mpc results/airsim_xxx_gpu_mppi
```

`docs/findings.md` の各 AirSim 節の「Reproduce」行に同じパターンの
コマンドが残っている。新しい paired study はそれを真似て YAML を
コピー + 改変するのが最短。

### 0.5 引き継ぎの心構え

- 数値クレームを書くときは **必ず Pareto cell を併記**
  (`(n=N, h=H)` の形)。Pareto セル外の数値は研究結果として書かない。
  paper §4.1 参照。
- GPU MPPI のコスト計算をいじるときは **goal-mask の再導入バグ
  (`docs/paper_a/section_4_prerequisites.md` §4.2 の 12-cell 反転)
  を踏まないこと**。`uav_nav_lab/planner/gpu_mppi.py` の goal-mask 処理
  (commit 2a9d196) を消したり書き換えたりするときは、その瞬間に
  全 Pareto sweep を回し直す覚悟が要る。
- Δ の符号の解釈に毎回引っかかるので明文化: **Δ > 0 は failure が
  seed 内でクラスタする (= ある seed では全機落ち、別 seed では全機通る)**。
  Δ < 0 は failure が seed 間に分散する。§3 ヘッドラインは「GPU MPPI の
  方が Δ が大きい = failure をクラスタさせる」。

---

## 1. v0.2.0 到達点 (2026-05-17 時点)

v0.1.0 (2025-11-XX) からの差分は `CHANGELOG.md` を一次資料とする。
ここでは plan.md として **「次に何ができるか」の判断材料** になる
部分だけ抜き出す。

### 1.1 フレームワーク本体 (差分なし)

- YAML 駆動 + 5 軸プラガブル + マルチドローン runner は v0.1.0 の
  形のまま継続。`runner/multi.py` の passive-first dispatch (v0.2.0)
  と AirSim two-phase step (v0.2.0) が地味だが load-bearing。

### 1.2 v0.2.0 で増えたバックエンド

| 軸          | v0.2.0 で追加                                                                   |
|-------------|-------------------------------------------------------------------------------|
| `sim`       | (差分なし — `airsim` / `ros2` は v0.1.0 末で入っていた。挙動修正多数)               |
| `scenario`  | `cells:` 明示配置 + `dynamic_obstacles` (linear+reflect 球) + `inflate: N`        |
| `planner`   | **`gpu_mppi`** (CUDA batched + Fibonacci-sphere + softmax) / `mppi` / `chomp` / `mpc_chomp` / `rrt` / `rrt_star` |
| `sensor`    | `lidar` / `pointcloud_occupancy` / `depth_image_occupancy` (v0.1.0 末)             |
| `predictor` | 既存 CV predictor のまま、planner 側 `use_prediction` で ON/OFF                  |

### 1.3 v0.2.0 で積み上がった研究結果

`docs/findings.md` に詳細。「これは押さえる」セット:

| トピック                                    | 結論                                                            |
|--------------------------------------------|----------------------------------------------------------------|
| MPC compute Pareto (2D + 3D)               | 2D `(16, 20)` / 3D `(8, 40)` の単独 Pareto                       |
| GPU MPPI Pareto (2D + 3D)                  | 2D `(128, 40)` 100 %/3.0 ms / 3D `(64-256, 20)` 100 %/3.5 ms。3D で CPU MPC を全軸で支配 |
| GPU MPPI goal-mask bug → Pareto 全反転      | `commit 2a9d196`。`docs/paper_a/section_4_prerequisites.md` §4.2 |
| **Multi-drone Δ-flip (dummy_3d n=100)**    | MPC +0.8 pp / GPU MPPI **+11.4 pp** at joint 78/77 % tie。 softmax がクラスタを増幅 |
| AirSim multi-drone n=30 × 3 cell           | 高度 stagger が **bimodal**。non-zero 100 % ceiling / uniform GPU MPPI 0/30 |
| Bridge fix: pause-after-reset              | `simPause(True)` 即発行で t=0 collision 蓄積を防ぐ (`commit 382d207`) |
| ROS 2 + AirSim-over-ROS-2 spatial parity   | 直結 vs ROS2 wrapper で ATE < 0.2 m (dummy)                       |
| 3D escape volume / 知覚遅延 / 風 miscal     | v0.1.0 の結果は v0.2.0 でも維持 (Pareto cell 更新後に再検証済み)    |

### 1.4 v0.2.0 で積み上がった可視化

- 単機 / 多機 3D anim に **per-drone rollout overlay** (cyan-tinted
  cloud + thick best-line) — `viz_rollouts: 8` で多機でも可読。
- `dynamic_obstacles` の replay scatter (赤球) を多機 3D anim に追加。
- README ヒーロー: AirSim 多機 obstacles GIF + rollout viz GIF の 2 枚。
- `record_airsim_*_compare.py` / `render_compare_gif.py` 系 — side-by-side
  GIF コンポジット。

### 1.5 v0.2.0 で書いた論文ドラフト (`docs/paper_a/`)

| ファイル                                  | 状態            |
|-----------------------------------------|----------------|
| `outline.md`                            | §1〜§7 骨子完成 |
| `section_1_motivation.md`               | 本文確定 (½p)   |
| `section_2_setup.md`                    | 本文確定 (1p)   |
| `section_3_headline.md`                 | 本文確定 (1.5p) |
| `section_4_prerequisites.md`            | 本文確定 (§4.1-4.3, 1.5p) |
| `section_4_4_sim_transferability.md`    | 本文確定 (§4.4, 1.5p) |
| `section_5_secondaries.md`              | 本文追加済       |
| `section_6_limitations.md`              | 本文追加済       |
| `section_7_repro_map.md`                | appendix map 追加済 |

合計 ~8 ページ。§7 まで本文/appendix の初稿は揃った。

---

## 2. 短期 (次の 1-3 PR で潰したい)

### 2.1 候補 A: **AirSim discriminating cell density sweep**

**動機**: §0.2 の static-cube cell で AirSim failure-level の planner
差は作れたが、GPU MPPI が 100 % ceiling に張り付いたため dummy_3d の
Δ-flip mechanism そのものは未再現。次は GPU MPPI も 60〜90 % per-drone
帯に落ちる density / placement sweep。

**やること**:
1. `examples/exp_airsim_multi_discriminating_n30*.yaml` をベースに、
   NS pillar の本数・位置と EW pillar の scale / `inflate` を sweep。
2. GPU MPPI per-drone が 60〜90 % に落ちる最小 cell を探す。
3. n=30 paired (MPC vs GPU MPPI) を `run_airsim_multi_chunked.sh` で回す。
4. dummy_3d と同じ「joint tie 付近で GPU MPPI の Δ が大きい」形が
   出るか確認。

**コスト**: 中〜高。1 cell あたり AirSim 起動 + 60 episode。
**判定**: 進行中。§0.2 の結果だけでも planner separation は書けるが、
Δ-flip transferability を caveat 無しにするにはこの sweep が必要。

#### 2.1.1 ここまでの読み (2026-05-18)

最初の static-cube discriminating cell は、AirSim 上で failure-level の
planner 差を作るという意味では成功した。ただし GPU MPPI が 30/30 joint
success の ceiling に張り付いたため、dummy_3d §3 の
「joint tie 付近で GPU MPPI の Δ が大きい」形にはまだ届いていない。

その後の sweep で分かったことはかなり明確:

1. **central occupancy は強すぎる。**
   `[29,33]` に central pillar を planner occupancy として入れると、
   GPU は per-drone 9/12 まで落ちるが joint は 0/3 になりやすい。
   これは target band ではなく floor。
2. **physical mesh x だけでは GPU を落とせない。**
   central mesh の physical x を 29.0, 29.25, 29.375, 29.45, 29.47,
   29.49, 29.50 と詰めても、planner occupancy が `[28,33]` 側なら
   GPU は 12/12, 3/3 の ceiling に戻る。
3. **inflate は二値化しやすい。**
   central occupancy `[29,33]` で `inflate=2` は ceiling、
   `inflate=3` は floor。`safety_margin` や `w_obs` を下げても
   floor は解けない。
4. **central 2 本化も hard すぎる。**
   `[29,33]` + `[31,33]`、`[32,33]`、`[31,34]` は全部 GPU 9/12,
   0/3 joint。2 本目を少し逃がしても同じ north collision floor に落ちる。
5. **baseline 5-pillar の physical EW 幅が初めて良い手触り。**
   central を足さず、baseline の 4 本 EW pillars を太くするだけで、
   GPU が ceiling から少し落ちた:
   - `base_ew05`: GPU 11/12 per-drone, 2/3 joint
   - `base_ew06`: GPU 10/12 per-drone, 1/3 joint
   - `base_ew07`: GPU 11/12 per-drone, 2/3 joint

つまり次に n=30 へ進めるべき候補は、central 系ではなく
**baseline 5-pillar + EW physical scale sweep**。ここなら GPU が
60-90 % per-drone 帯に入りそうで、joint も 0 ではない。MPC 側も
baseline n=30 で既に 87.5 % per-drone / 22/30 joint なので、EW width を
少し太くしても完全 floor には落ちにくい可能性がある。

#### 2.1.2 Pilot log

| candidate | YAML pair | 3-seed result (seeds 42-44) | read |
|-----------|-----------|-----------------------------|------|
| baseline | `exp_airsim_multi_discriminating_n30*.yaml` | n=30 済み: MPC 87.5 % per-drone / 22/30 joint、GPU 100 % / 30/30 | planner separation は出るが GPU ceiling |
| mid | `exp_airsim_multi_discriminating_mid_n30*.yaml` | MPC 5/12 per-drone, 0/3 joint; GPU 12/12, 3/3 joint | late `[33,45]` は MPC だけ悪化、GPU ceiling のまま |
| central | `exp_airsim_multi_discriminating_central_n30*.yaml` | MPC 7/12, 0/3; GPU 9/12, 0/3 | central `[29,33]` は GPU を target 帯に落とすが joint floor |
| central_soft | `exp_airsim_multi_discriminating_central_soft_n30*.yaml` | MPC 7/12, 0/3; GPU 9/12, 0/3 | mesh を細くしても central collision は残る |
| central_north | `exp_airsim_multi_discriminating_central_north_n30*.yaml` | GPU-only: 9/12, 0/3 | y=33→34 でも x=29 系は GPU floor 側 |
| central_west | `exp_airsim_multi_discriminating_central_west_n30*.yaml` | MPC 8/12, 0/3; GPU 12/12, 3/3 | x=28 に逃がすと GPU ceiling に戻る |
| central_west_thick | `exp_airsim_multi_discriminating_central_west_thick_n30*.yaml` | MPC 9/12, 1/3; GPU 12/12, 3/3 | MPC は少し戻るが GPU は ceiling |
| central_half | `exp_airsim_multi_discriminating_central_half_n30*.yaml` | GPU-only: 12/12, 3/3 | physical x=29.0 は GPU ceiling |
| central_29p25 | `exp_airsim_multi_discriminating_central_29p25_n30*.yaml` | GPU-only: 12/12, 3/3 | physical x=29.25 も GPU ceiling |
| central_29p375 | `exp_airsim_multi_discriminating_central_29p375_n30*.yaml` | GPU-only: 12/12, 3/3 | physical x=29.375 も GPU ceiling |
| x-sweep 29.45 | `/tmp` generated by `run_airsim_discriminating_x_sweep.sh` | GPU-only: 12/12, 3/3 | physical x=29.45 も GPU ceiling。seed 42 final_t=21.15s で境界感はある |
| x-sweep 29.47 | same | GPU-only: 12/12, 3/3 | physical x=29.47 も GPU ceiling |
| x-sweep 29.49 | same | GPU-only: 12/12, 3/3 | physical x=29.49 も GPU ceiling |
| x-sweep 29.50 | same | GPU-only: 12/12, 3/3 | physical x=29.50 でも GPU ceiling。central floor は mesh x ではなく planner occupancy `[29,33]` が主因 |
| occ29_inflate1 | `/tmp` generated by `run_airsim_discriminating_param_sweep.sh` | GPU-only: 12/12, 3/3 | central occupancy でも inflate=1 は GPU ceiling |
| occ29_inflate2 | same | GPU-only: 12/12, 3/3; paired pilot MPC 8/12, 0/3 vs GPU 12/12, 3/3 | inflate=2 も GPU ceiling。MPC だけ hard |
| occ29_margin04 | same | GPU-only: 9/12, 0/3 | safety_margin を下げても inflate=3 の floor は解けない |
| occ29_margin05 | same | GPU-only: 9/12, 0/3 | 同上 |
| occ28_x29p45_ew035 | same | GPU-only: 12/12, 3/3 | EW 0.35 は ceiling、ただし seed44 final_t=18.45s |
| occ28_x29p45_ew04 | same | GPU-only: 12/12, 3/3 | EW 0.4 も ceiling |
| occ28_x29p45_ew05 | same | GPU-only: 12/12, 3/3 | EW 0.5 も ceiling、seed42/43 は 16-20s まで遅延 |
| occ28_x29p45_ew06 | same | GPU-only: 12/12, 3/3 | EW 0.6 も ceiling |
| occ29_inflate2_ew05/06 | same | GPU-only: 12/12, 3/3 | central occupancy + inflate2 に EW 幅を足しても GPU ceiling |
| occ29y34_inflate2 | same | GPU-only: 12/12, 3/3 | central y+1 + inflate2 も ceiling |
| occ30y33_inflate2 | same | GPU-only: 12/12, 3/3 | central x+1 + inflate2 も ceiling |
| occ29_inflate2_second31 | same | GPU-only: 9/12, 0/3 | central 2 本化は floor |
| occ29_inflate2_second32 | same | GPU-only: 9/12, 0/3 | 2 本目を x=32 に逃がしても floor |
| occ29_inflate2_second31y34 | same | GPU-only: 9/12, 0/3 | 2 本目を y+1 に逃がしても floor |
| occ29_wobs100 | same | GPU-only: 9/12, 0/3 | inflate3 floor は w_obs 半減でも解けない |
| occ29_wobs50 | same | GPU-only: 9/12, 0/3 | w_obs 1/4 でも floor |
| base_ew05 | same, generated from baseline 5-pillar | GPU-only n=10: per-drone 39/40 (97.5 %), joint 9/10。failure は seed 43 のみ (drone idx 3) | ほぼ ceiling。1 seed の偶発 collision のみで target 帯下端ですらない |
| base_ew06 | same, generated from baseline 5-pillar | GPU-only n=10: per-drone 35/40 (87.5 %), joint 5/10。failure seeds 43,44,45,46,50 (全て drone idx 3) | target 帯上端、joint も tie 寄り。MPC baseline (87.5 %) と並ぶため Δ-flip 機会最大 |
| base_ew07 | same, generated from baseline 5-pillar | GPU-only n=10: per-drone 34/40 (85.0 %), joint 4/10。failure seeds 42,46,47,49,50,51 (全て drone idx 3) | ew06 とほぼ同じ hard さ。failure 数 ew06 と僅差 |
| dense | `exp_airsim_multi_discriminating_dense_n30*.yaml` | MPC 8/12, 0/3; GPU 9/12, 0/3 | 7 本柱は hard すぎ。n=30 に行く前に中間 knob が必要 |

#### 2.1.3 追加した runner

```bash
# cheap smoke: GPU only, default N=5, ordered from newest boundary probes
scripts/run_airsim_discriminating_sweep.sh

# physical x-position generator for the central AirSim mesh
X_VALUES="29.42 29.45 29.47" scripts/run_airsim_discriminating_x_sweep.sh

# planner/mesh parameter probes generated from committed bases
VARIANTS="occ29_inflate1 occ29_inflate2" scripts/run_airsim_discriminating_param_sweep.sh
VARIANTS="base_ew05 base_ew06 base_ew07" scripts/run_airsim_discriminating_param_sweep.sh

# paired pilot / full n=30
MODE=paired N=3 CANDIDATES=central_soft scripts/run_airsim_discriminating_sweep.sh
MODE=paired N=30 CANDIDATES=<winning-cell> scripts/run_airsim_discriminating_sweep.sh
```

#### 2.1.4 次にやること

**`base_ew06` n=30 paired まで完走 (2026-05-18)。Δ-flip の符号反転を発見。**

base_ew06 n=30 paired 結果 (seeds 42-71):

| planner  | per-drone (Wilson)            | joint (Wilson)              | indep⁴ | Δ        |
|----------|-------------------------------|-----------------------------|--------|----------|
| MPC      | 104/120 = 86.7 % [79.4, 91.6] | 19/30 = 63.3 % [45.5, 78.1] | 56.4 % | **+6.9 pp** |
| GPU MPPI | 114/120 = 95.0 % [89.5, 97.7] | 24/30 = 80.0 % [62.7, 90.5] | 81.5 % | -1.5 pp  |

McNemar paired-seed: both=14, MPC-only=5, GPU-only=10, neither=1 → p ≈ 0.302。
Sign は **GPU 優位** (joint 80 vs 63, McNemar 10 vs 5)。

**重要発見: Δ-flip の符号が dummy_3d と逆**。

| backend / cell           | MPC Δ    | GPU MPPI Δ |
|--------------------------|----------|-----------|
| dummy_3d §3 (n=100)      | +0.8 pp  | **+11.4 pp** ← GPU cluster |
| AirSim base_ew06 (n=30)  | **+6.9 pp** ← MPC cluster | -1.5 pp |

per-drone は両 backend で tied (94-95 %)、Δ で差を生む構造は transferable。
ただし **どちらの planner が cluster するかは backend/cell で反転**。

メカニズム読み (per-seed disagreement から):

- MPC failure 11 seeds: drone 3 単独 = 8 / 多機 cluster (drones 1-2-3) = 3
  (seeds 55, 67 で 3 機同時崩壊、seed 66 で drones 1-2 崩壊)。
- GPU MPPI failure 5 seeds: **全部 drone 3 単独** (seeds 43, 45, 46, 50, 52)。
- AirSim cell では MPC が north end の 5-pillar layout で多機を同方向に
  詰め込み、cluster failure を生む。GPU MPPI の rollout cloud は
  drone 3 (一番南/外側) の hard avoidance では geometric に落ちるが
  cluster は作らない。

---

#### 2.1.5 次の判断ポイント

A, B, C, D 全て完了 (2026-05-18)。完了した内容:

- **A**: `docs/paper_a/section_4_4_sim_transferability.md` に新 §4.4.4
  (Δ-flip 符号反転) を挿入、後続 renumber。outline.md と repro_map も
  同期。後に n=50 数値で全面リライト。
- **B**: `docs/findings.md` に "AirSim multi-drone base_ew06
  density-sweep n=50" セクション追加。
- **C**: n=50 paired 完走 (seeds 42-91)。McNemar p 0.302 → 0.167
  (まだ未有意)。**MPC Δ は +6.9 → +3.8 pp に縮小** — 新 20 seed で
  MPC cluster failure (seeds 55/66/67 タイプ) は再現せず、Δ の主源が
  rare events (3/50 = 6 % cluster rate) であることが判明。
- **D**: lane30 / lane22 GPU-only smoke で drone-3 bias の origin を probe。
  結論: **drone-3 単独 failure は lane shift で removable ではない**。
  両 variant とも 10/10 drone 3 fail。collision 位置は lane22 で
  (21.2, 23.3, 26.6) @ t=6.90s、base_ew06 (x=26) で同じ (21.2, 23.2, 26.6)
  @ t=7.35s — どちらも planner が inflate=3 で膨らんだ (25,27) pillar を
  detour した先で collide。lane30 は central (30, 29, 26.6) で別ハザード。
  drone-3 bias は (planner + inflate + EW pillar) の幾何的 root であり、
  MPC cluster failures (seeds 55/66/67) は独立の planner-specific failure
  mode で、これが Δ 符号反転の源。

**残された open work** (優先度の高い順):

~~1. (21.2, 23.2, 26.6) で何に collide しているか確認~~ **完了
   (2026-05-18)**。`airsim_bridge.py:525` パッチで `collision_object` を
   per-step JSON に記録。MPC seed 67 cluster 再走で drones 1, 2 が
   `collision_object` 空 = **drone-drone collision @ central crossing**、
   drone 3 は `uavnav_disc_ew_35` を hit と確定。

~~4. AirSim variability の原因解明 → 3 batch × n=15 で characterize~~
   **完了 (2026-05-18)**。fresh seeds 200-254 で 3 独立 batch 実行:
   - cluster rate: 1/15, 3/15, 0/15 (mean 8.9 %, n=50 の 6 % と consistent)
   - McNemar 方向は batch 1 tie / batch 2 GPU 寄り / batch 3 **MPC 寄り**
     と逆転、n=50 (GPU 寄り) は独立 sample で再現せず
   - 3 batch 累積 n=45: MPC joint = GPU joint = 39/45 = 86.7 %、
     McNemar p ≈ 0.77 (tie)
   - 結論: **「GPU > MPC」の数値主張は環境依存**。Δ-flip の cluster mode
     は両 backend で robust (MPC clusters, GPU 平坦) → §4.4.4 / findings.md
     に variability table 追加済

2. **n=80 まで延長して McNemar p<0.05 を狙う** — variability 解析で
   ナンセンスと判明 (batch ごとに p の方向が反転)。**取下げ**。
   代わりに n ≥ 200 paired or controlled environment が必要だが
   どちらも §4.4 scope 外。

3. **§3.1 (静的障害物密度 sweep)** に移る。dummy_3d n=100 の §3 headline は
   既存、AirSim では base_ew06 1 セルしか測ってない。N=2/3/4/6 や
   pillar density sweep は別軸の研究。優先度 中 (新規の研究軸)。

5. **論文 §6 limitations を更新** — variability 発見を limitations 節に
   反映 (AirSim multi-drone measurement は single n=N study では sample
   size を超えた variance を持つ; controlled environment が要る claim
   向き)。優先度 中。

---

**以下は完了済みの履歴。**

選定: **`base_ew06`** を n=10 paired (MPC + GPU) に進めた。

- per-drone 87.5 % は target 帯 60-90 % の中で MPC baseline (87.5 %) と
  並ぶ tie 付近の sweet spot。Δ-flip の seed-cluster 機構が観測しやすい。
- ew07 (85.0 %) はほぼ等価、ew05 (97.5 %) は ceiling 寄りで除外。
- **注意**: n=10 GPU-only の failure は全 seed で **drone idx 3 固定**。
  seed では散ってもいるが drone は不変。これは dummy_3d §3 の seed-cluster
  Δ-flip とはメカニズムが異なる可能性 (geometric drone-position effect)。
  原因候補は Drone4 spawn 位置 (settings.json) と 5-pillar layout の
  相互作用。n=10 paired で MPC 側に同じ drone-bias が出るかが診断材料。

次の command:

```bash
VARIANTS="base_ew06" \
  MODE=paired N=10 BASE_SEED=42 \
  scripts/run_airsim_discriminating_param_sweep.sh
```

ここで見るべきもの:

- MPC が完全 floor にならないか。
- GPU と MPC の per-drone が同じ 60-90 % 帯に乗るか。
- joint が tie 付近か、少なくとも McNemar の一方的勝ちだけで終わらないか。
- MPC でも drone idx 3 に failure が偏るか (偏れば geometric、
  偏らなければ planner-specific)。

n=10 paired が良ければ n=30 paired に進む:

```bash
VARIANTS="base_ew06" \
  MODE=paired N=30 BASE_SEED=42 \
  scripts/run_airsim_discriminating_param_sweep.sh
```

#### 2.1.5 今は後回しにする候補

- `central*` 系: GPU を落とせるが joint floor が強すぎる。
- `inflate` 系: `inflate=2` ceiling / `inflate=3` floor で中間がない。
- `w_obs` / `safety_margin`: floor/ceiling の境界を動かせなかった。
- `occ28_x29p45_ew05`: seed time は伸びるが 3/3 success。baseline EW
  scale の方が直接 failure を出したので優先度を下げる。
- NS `[27,15]` scale 0.1/0.2/0.3: まだ未実施。EW scale n=10 が
  ダメなら戻る。

手書き YAML はこれ以上増やさない。param generator に variant を足して
GPU-only short run で見る。

### 2.2 候補 B: **AirSim limit-case cliff** (`exp_airsim_latency_limit.yaml`)

**動機**: v0.1.0 末で「AirSim だと motor ramp が 3-4 step cliff を
平滑化する」結果。max_speed=15 + 高密度 (`random_layer`) なら cliff が
返ってくるはず、という仮説。YAML は用意済 (`examples/exp_airsim_latency_limit.yaml`)。

**やること**:
- `latency_steps ∈ {0,1,2,3,4,5,6}` × n=10 を AirSim で sweep。
- dummy_3d で見えた崖が AirSim 高速・高密度で復活するか確認。

**判定**: 優先度 中。A の方が論文ヘッドラインへの貢献が大きい。

### 2.3 候補 C: **ROS 2 ↔ AirSim wrapper の動作確認**

**動機**: v0.2.0 で `cmd_msg_type: airsim_vel_cmd` と
`scripts/compare_spatial_runs.py` まで整備済 (`docs/findings.md` §"AirSim
over ROS 2 parity harness")。残るのは **AirSim ROS2 wrapper の reset /
teleport** 経路の挙動確認 — wrapper 経由で multi-drone reset が
hang しないか、staggered スポーン姿勢が引き継がれるか。

**判定**: 優先度 低。論文 §4.4.6 (ROS 2 invariance) は dummy ROS2 sim で
既に成立しており、AirSim wrapper まで深掘りするのは scope 外気味。

### 2.4 候補 D: **論文 §7 を埋める** — 完了

**動機**: §1〜§6 は本文あり。残りは §7 reproducibility map
(各 result → YAML へのフルマップ)。

**やること**:
- `docs/paper_a/section_5_secondaries.md` — 追加済。
- `docs/paper_a/section_6_limitations.md` — 追加済。必要なら短縮・整形。
- `docs/paper_a/section_7_repro_map.md` — 追加済。
  `examples/exp_*.yaml` / runner / findings anchor の対応表を持つ。

**判定**: 完了。次は候補 A の AirSim discriminating cell density sweep。

---

## 3. 中期 (次の 5-10 PR)

### 3.1 AirSim Δ-flip シリーズの完結

候補 A が片付いたら、次の自然な層は:

- [ ] 静的障害物密度 sweep (3〜15 個) で Δ の per-drone-rate 依存を曲線で
- [ ] dynamic obstacle 速度 sweep (1〜5 m/s) で multi-drone interaction の
      severity 依存
- [ ] N=2, 3, 4, 6 で AirSim multi-drone N-scaling (現状 dummy_3d のみ)

### 3.2 AirSim multi-drone reset hang を upstream に報告

`docs/findings.md` §"AirSim multi-drone reset path: stale collision
flag and intermittent hang" に再現手順がまとまっている。Microsoft の
AirSim repo もしくは Colosseum fork に issue を切って、minimum
reproducer (`scripts/run_airsim_multi_chunked.sh` の Python 化 + 4 機
settings.json) を添付する。

**判定**: 優先度 低。我々の側は chunked runner で逃げている。upstream
報告は科学的礼節としてやる、レベル。

### 3.3 RL 比較ベースライン

v0.2.0 で `gym.Env` ラッパー + `stable_baselines3` SAC scaffolding は
入っている (commit 4e062ed)。次は:

- [ ] SAC を `voxel_world` 3D で学習させて成功率を測る
- [ ] CPU MPC / GPU MPPI / SAC を同じ scenario の paired comparison に乗せる
- [ ] 学習側の計算予算 (GPU hours) と planner の per-replan 予算で
      cost-aware Pareto を描く

**判定**: 優先度 中。論文 v2 (もしくは別論文) の素材。v0.2 paper は
学習ベースを引き合いに出さないで閉じる予定なので、急がない。

### 3.4 実機転移 (sim-to-real)

ROS2 bridge → MAVROS → PX4 SITL → 実機の最後の 1 段。屋内
OptiTrack で position feedback を入れる方が前段検証として固い。
論文 §6 limitations に明示する **「現状未検証」セクション** にも
対応する。

**判定**: 優先度 低 (本研究の scope 外)。

---

## 4. 長期 (構想だけ)

### 4.1 マルチエージェント学習との接続

§3.3 を発展させて、ピア予測器 (CV / LSTM / Transformer) を MPC/MPPI と
学習エージェント両方に共通インターフェースで差し込めるようにし、
「予測器の質が coordination Δ に与える影響」を planner 横断で測る。

### 4.2 公開化

論文 v1 (この plan の §1.5 / 候補 D で閉じる draft) を arXiv +
ワークショップ (RA-L / ICRA workshop / IROS workshop) に投げる。
タイトル候補 (CHANGELOG / `outline.md` 参照):

- "Compute-aware planner Pareto and coordination Δ on a unified
  2D/3D dynamic-obstacle benchmark"
- "GPU MPPI's softmax flips the multi-drone coordination Δ — a
  paired study on dummy_3d, AirSim, and AirSim-over-ROS-2"

### 4.3 ベンチマーク化

`uav-nav-lab` を benchmark suite として整える (HuggingFace datasets
的にエピソード log を公開、leaderboard 的な YAML 集合を提供)。
論文 v1 公開後、コミュニティの反応次第。

---

## 5. やらないと決めたもの (供養コーナー)

| 案                                                  | 却下理由                                                  |
|----------------------------------------------------|---------------------------------------------------------|
| MPC を CasADi/IPOPT に置き換え                       | フレームワークの「シンプル」を壊す。研究的価値も既存 sampling MPC で出ている。 |
| 全 planner に GPU 必須化                            | `gpu_mppi` を別 planner として並列に置く現方針で十分。       |
| Web UI / dashboard                                  | scope 外。CLI + matplotlib + GIF で足りている。           |
| `omegaconf` への移行                                 | 現 `ExperimentConfig` で困っていない。                    |
| collision 以外の per-step reward 設計                 | フレームワークが planner 比較ツールとしての性格を失う。     |
| Δ-flip を AirSim no-obstacle で測る                  | §0.2 の通り、3 セルとも天井 or floor 割れで mechanism を直接観測不可。**障害物を入れる方向 (候補 A) でしか測れない**ことを 2026-05 に確認済み。 |
| GPU MPPI の goal-mask を再リファクタ                  | `commit 2a9d196` の修正で 12-cell sweep が反転した経緯あり (`docs/paper_a/section_4_prerequisites.md` §4.2)。論文公開までは触らない。 |
| AirSim multi-drone reset の **C++ 側修正**            | upstream の dispatch loop の問題。chunked runner で workaround 済。我々の repo で fork する価値は無い。 |
| 単機 demo を更に派手にする                            | v0.2.0 で multi-drone obstacles GIF が hero になり、単機は脇役で良い。 |
