# uav-nav-lab — Plan & Roadmap

> **位置付け**: `docs/findings.md` は *終わった研究の記録*、`README.md` は
> *入口とハイライト*、`CHANGELOG.md` は *バージョン毎の差分*、この
> `plan.md` は *これから何をやるか / なぜやるか / 引き継ぐ人が何を踏むか*
> をまとめる作戦ノート。
>
> 最終更新: 2026-05-25 (README hero replaced with post-fix drone race GIF)

---

## 0. Codex への引き継ぎ要点 (まずここを読む)

このドキュメントを引き継いだ codex への TL;DR。

### 0.1 リポジトリの今の状態 (2026-05-25)

- **v0.2.0 タグ済み** (2026-05-17)。`CHANGELOG.md` に差分要約あり。
- v0.1.0 → v0.2.0 で **88 commit**。中身は §1 を参照。
- 論文ドラフト (`docs/paper_a/`) は §1〜§7 まで本文/appendix map あり。
- **2026-05-21 fix**: commit `1646e11` で multi-runner の
  dynamic-obstacle freeze bug を修正。total-wipeout episode の直後に
  dynamic obstacles が凍り、次 episode 以降の race / gates / dyn4 / chaos
  系の数値を汚染していた。修正後の再走では該当シナリオが各 planner
  **100 % collision** になり、旧 "MPC 51.7 % vs softmax 3.3 %" 系の
  headline は artifact 扱い。README はこの注意書きに更新済み。
- GitHub About / README hero は応急整理済み。`docs/findings.md` と
  `docs/paper_a/section_3_headline.md` の pre-fix dynamic-obstacle claim も
  `1646e11` invalidated 扱いに整理済み。残る dynamic-obstacle 作業は
  **新しい post-fix non-floor cell の再設計**であって、旧 claim の修復ではない。
- README 先頭 GIF は `docs/images/compare_race_temperature_avoid.gif` に
  差し替え済み。これは temporary な 4-way intersection hero ではなく、
  post-fix race-simple temperature counterfactual の実ログから描いた
  2D top-down drone race。2026-05-25 の見直しで全体 oval でも
  side-by-side でもなく、`scripts/render_race_avoidance_overlay_gif.py`
  の t≈29s encounter overlay に切り替えた。赤は vanilla `t=1.0` の
  `contact @ 29.25s`、緑は同じ cell の `t=0.1` が赤い safety halo
  外へ detour する軌跡。ここでやっと
  「image は drone race」「障害物を避けていることが読める」という
  入口条件を満たした。裏取り数値は
  `docs/data/race_hero_encounter_metrics.json` に固定済み
  (sweeper travel 8.40 m、low-temp window min clearance +0.45 m)。

#### 2026-05-22..24 の 3 日アーク (HEAD = `016e031`)

最新 27 commit は **大きく 4 つの塊**:

1. **F → J: peer prediction sweep + U-shape mechanism (2026-05-22 朝)**
   `e196ed2` … `7adfcb9`。E5 で見つけた σ=3 knee の cell 一般性を
   peer / v1 / wave で sweep。`fb45bcb` で **predictor.reset never called**
   バグ発見・修正 (全 noisy_* 系を再走、結果は qualitative には不変)。
   メカニズム: top-rollout disagreement と prior-alignment が U-shape の
   直接源と確定 (`7adfcb9 I`)。
2. **K → R': vanilla MPPI 「U-shape の谷」一般則 → N+P predictive rule (2026-05-22 昼〜2026-05-23 早朝)**
   `55d0472 K` … `19c2e20 R'`。vanilla MPPI が cell 依存で U/逆 U/単調を取る
   現象から、**(top2 disagreement, chosen-vs-goal angle) = N+P signal** が
   cell-optimal temperature を予測できる経験則を導出。`6ae3211 R` で
   `warmup_select_mppi` planner を実装 — 1 episode の warmup → N+P rule で
   t ∈ {0.1, 1.0, 10} を自動選択。`19c2e20 R'` で multi-drone session pooling
   bug 修正 (per-drone でなく全機合議で 1 つ選ぶ)。
3. **S/T/X/U: OOD validation + family selector + chokepoint scope condition (2026-05-23 昼)**
   `61e8709 S` (city_v1 OOD で N+P rule 当たる) → `566f31d T` (city 4 cell stress、3/4 当たり、chokepoint だけ確信を持って外す) → `366519a X` (MPC vs MPPI family
   selector 仮説を 9 cell で検証、**MPC は 0/9 で MPPI に dominated**、家族選別仮説死亡 — honest negative を「MPPI + N+P rule は 35pp 平均優位」と reframe) →
   `97b2ac2 U` (chokepoint scope condition の解明: 信号ベースでなく
   **geometric forced-commitment width** で決まる; 詳細は §2.7 / `docs/findings.md`)。

4 つ目の塊が **2026-05-23 夕〜2026-05-24 = scaling-law refactor campaign**。
研究軸とは独立に、ユーザー要望「モジュール分割・依存局所化・テスト単位細分化で
開発全体のスケーリング則を改善」に対応:

- `7abe635 R`: `uav_nav_lab/analysis/` レイヤを新設、5 scripts から
  `joint_outcomes`, `joint_success_rate`, `diagnose_warmup` を公開関数として
  抽出 (`_SHARED_SESSIONS` 等の private surface はライブラリ側に閉じる)。
  -230 / +56 LOC。tests/analysis 追加 (11 test)。
- `dba7c3c T`: `runner/multi/episode.py` 390 LOC を `phases.py` (110 LOC) +
  `outcomes.py` (100 LOC) + 残 230 LOC orchestrator に分割。
- `e36da4c S`: `planner/_grid.py` に 3 helper (`point_to_cell`,
  `point_is_occupied`, `mask_dynamic_obstacle_cells`) 集約 + `planner/mppi.py`
  317 LOC を `mppi/{__init__, planner, rollout, aggregator}.py` に分割。
  mpc.py の重複 4 cell-helper も同関数を使うよう更新。
- `1c84fcf S2`: `planner/mpc.py` 299 LOC を `mpc/{__init__, planner,
  aggregator}.py` に分割し、**mpc/mppi 間で重複していた 70 LOC `score_rollouts`
  を `planner/_rollout.py` に shared 抽出**。結果として MPC と MPPI のコード差は
  aggregator (argmin vs softmax) のみ。
- `81ca0cc S3/V`: `planner/chomp.py` 334 LOC を `chomp/{__init__, planner,
  objective}.py` に分割し、`mpc_chomp.py` の CHOMP private helper 依存を
  public objective API へ切り替え。同 commit で `tests/planner/conftest.py`
  を新設し、固定サイズ空 occupancy / registry fixture を共通化。
- `016e031`: N+P rule arc を paper draft へ統合。§5 に warmup temperature
  rule + family-selector negative、§6 に city_chokepoint の geometric scope
  condition、§7 repro map / outline に対応行を追加。

refactor 系 commit は pytest 全件 + x smoke が **bit-for-bit identical** で
通っていることを characterization tests で確認済。LOC delta は累計
**-460 LOC, +25 test**。短期 refactor campaign は S3/V まで完了、W は
GPU goal-mask 再発リスクがあるため v0.3 後判断。次の大きい打ち手は
post-fix dynamic-obstacle cell 再設計か、paper draft assembly。

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

7. **multi-runner dynamic obstacle freeze (commit `1646e11`)** —
   total-wipeout episode 後に dynamic obstacles が freeze し、次 episode
   以降が見かけ上 easy になる bug があった。2026-05-21 時点で fix 済み。
   **pre-fix の race / gates / dyn4 / chaos / Smart MPPI v4-v5 数値は
   citation 禁止**。使うなら必ず `1646e11` 以後で rerun する。

8. **file → package 昇格時の相対 import 段数バグ (refactor campaign で 2 回踏んだ)** —
   `planner/mpc.py` を `planner/mpc/planner.py` に昇格すると、相対 import が
   1 段深くなる。例えば旧 `from .base import Planner` (= `..base`) は
   新ファイル中で `from ..base import Planner` に書き換える必要がある。
   `planner/_grid.py` → `from .._grid import ...`。`predictor` のような
   2 段親への access は `from ...predictor import ...` に。
   **対策**: `grep -n 'from \.' new_subpackage/*.py` で 1 段 import を全件確認。
   その後 `find uav_nav_lab/planner -name __pycache__ -exec rm -rf {} +` で
   stale bytecode を消してから smoke 実行。

9. **refactor で smoke を bit-exact にしたい時は MPPI 経路を回す** —
   `scripts/x_planner_family_gather.py` は MPPI を warmup として再走する
   ため、planner internals (rollout cost) の数値変動が `docs/data/
   x_planner_family_data.json` に lossless に流出する。S/S2 はこの diff が
   0 byte であることを baseline 等価性の証拠としている。一方で MPC 経路の
   bit-exact check は現状 `tests/planner/test_mpc.py` のプロパティ test 経由
   でしか測れない。MPC 側で大きな変更を入れる時は **同レベルの smoke を
   別途用意する** (`scripts/x_planner_family_gather.py` の MPC 版が欲しい)。

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

#### 0.4.x refactor 系 PR の検証手順 (R/T/S/S2 で 4 連続成功した固定パターン)

```bash
# Step 1: __pycache__ を完全に消す (relative-import の段数バグを引きずらないため)
find uav_nav_lab -name '__pycache__' -type d -exec rm -rf {} +

# Step 2: pytest 全件 (要 venv + PATH + unset PYTHONPATH)
export PATH="$HOME/.local/bin:$PATH" && unset PYTHONPATH && \
  source .venv/bin/activate && pytest -q
# 期待: 209 passed, 1 skipped

# Step 3: x_planner_family smoke を baseline と diff
cp docs/data/x_planner_family_data.json /tmp/x_baseline.json
python scripts/x_planner_family_gather.py
diff /tmp/x_baseline.json docs/data/x_planner_family_data.json
# 期待: 差分なし (= MPPI rollout の数値同一)

# Step 4: commit (必ず gmail author)
git add <files>
git commit --author="Ryohei Sasaki <rsasaki0109@gmail.com>" -m "..."

# Step 5: push (ユーザー明示の "push!" を待ってから)
git push
```

R/T/S/S2/S3/V はこの pattern を踏襲して pass。次に refactor を入れるなら
同じ手順を流せば campaign としての一貫性が保てる。

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

#### 0.5.x このリポジトリで commit する時の制約 (CRITICAL — codex も Claude も同じ)

- **commit author は必ず gmail を per-commit で指定**。
  ```bash
  git commit --author="Ryohei Sasaki <rsasaki0109@gmail.com>" -m "..."
  ```
  local git config は work email (`ryohei.sasaki@map4.jp`) になっている。
  **git config は絶対に書き換えない** (global safety rule)。
  amend が必要なら `git commit --amend --author="Ryohei Sasaki
  <rsasaki0109@gmail.com>" --no-edit`。pushed commit を amend したくなる
  ケースは事前にユーザー確認。
- commit / PR 説明文に AI 生成表記 (`Co-Authored-By: Claude`,
  `🤖 Generated with Claude Code` 等) は **付けない**。CLAUDE.md
  (`~/.claude/CLAUDE.md`) に明文化されている。
- push は **ユーザー明示の "push!" を待つ**。auto-mode でも push は
  確認対象。1 commit ぶんの authorization は他 commit に持ち越さない。
- pytest 実行は **必ず venv + PATH + unset PYTHONPATH**:
  ```bash
  export PATH="$HOME/.local/bin:$PATH" && unset PYTHONPATH && \
    source .venv/bin/activate && pytest -q
  ```
  bare `pytest` は ROS の `PYTHONPATH` 汚染で fail する。
- refactor commit は **characterization test = pytest 209 passed + smoke
  bit-for-bit identical** が pass する場合のみ merge。`scripts/x_planner_family_gather.py` の
  出力 (`docs/data/x_planner_family_data.json`) を baseline と diff して
  数値同一性を確認するパターンが S/S2 で 2 連続成功した。新規 refactor も
  この pattern を踏襲すること。
- 公開済の planner import path は維持: `from uav_nav_lab.planner.mpc import
  SamplingMPCPlanner` と `from uav_nav_lab.planner.mppi import MPPIPlanner` は
  subpackage 化後も生きている (`mpc/__init__.py`, `mppi/__init__.py` で
  re-export)。同様に `from uav_nav_lab.planner.mpc_chomp import ...` も保持。

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

~~3. **§3.1 (静的障害物密度 sweep)** に移る。dummy_3d n=100 の §3 headline は
   既存、AirSim では base_ew06 1 セルしか測ってない。N=2/3/4/6 や
   pillar density sweep は別軸の研究。優先度 中 (新規の研究軸)。~~
   **完了 (2026-05-19)**。詳細は §2.5 を参照。dummy_3d で
   N ∈ {2, 3, 4, 6, 8, 10, 12} × density ∈ {30, 120, 240} の
   3×3 grid (中心 N=4/6/8 行) を全測。**Δ-flip の符号は (N, density)
   corner-specific** で、N=4 baseline が GPU clusters、N=4 dense で
   MPC clusters、N=6 で flip なし、N=8 baseline で GPU per-drone が
   8-fold-symmetric 中央交差で唯一 collapse (McNemar p ≈ 0.0001 が MPC 寄り)、
   N=10 が sweep max (Δ +24.3 pp)、N=12 で fall-back。

~~5. **論文 §6 limitations を更新** — variability 発見を limitations 節に
   反映 (AirSim multi-drone measurement は single n=N study では sample
   size を超えた variance を持つ; controlled environment が要る claim
   向き)。優先度 中。~~ **完了 (2026-05-19)**。`section_6_limitations.md`
   に variability 節 + (N, density) grid 全体図 + dynamic obstacle 軸を
   追記。§3 headline も "Scope of the headline claim" + "Dynamic-obstacle
   extension" の 2 ブロックを追加して N=4 baseline cell に scope。
   §4.4.4 / §7 / outline も整合。詳細は §2.5。

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

### 2.5 候補 E: **dummy_3d (N, density, dynamic obstacle) 軸の拡張** (2026-05-19)

**動機**: §3 headline の Δ-flip は N=4 / density=30 の 1 cell の現象。
論文の主張を一般化するには grid の他 cell + 動的軸での挙動が必要。

**今日 (2026-05-19) 完了したもの**:

1. **N-scaling sweep** `N ∈ {2, 3, 4, 6, 8, 10, 12}` paired (commits
   `4b82f66`, `97a9577`):
   - GPU MPPI の Δ 優位は **non-monotonic in N**
   - N=10 で sweep max (Δ +24.3 pp)、N=2/8/12 で逆転
   - N=8 で GPU per-drone が 8-fold symmetric 中央交差で唯一 collapse
     (per-drone 69 % vs MPC 92 %, McNemar p ≈ 0.0001 が MPC 寄り)

2. **(N, density) 3×3 grid** `N ∈ {4, 6, 8} × {30, 120, 240}` paired
   (commits `c3c0bd7`, `ce35c7f`, `d00e0f2`):
   - **N=4 行**: density で Δ 符号が flip (GPU baseline → MPC packed)
   - **N=6 行**: flip なし (GPU per-drone 優位がそのまま density で開く)
   - **N=8 行**: density が GPU collapse を *unwind* する (re-tie at dense)
   - retraction commit `ce35c7f` で N=4 で見えた flip は N=6/8 に generalize
     しないことを明示

3. **Dynamic obstacle 速度 sweep** at §3 N=4 baseline (commits `d9571ab`,
   `38bc4f0`) — **pre-fix result, 2026-05-21 時点では citation 禁止**:
   - v=2 m/s で GPU MPPI joint が **86.7 % → 3.3 %** 崩壊、MPC は 73 %
   - 失敗 drone は **常に北 drone** (t ≈ 5 s)、softmax bidirectional
     cancellation メカニズム
   - v=8 で両 planner ground
   - off-corridor probe (x=15): §3 baseline 完全復元 → mechanism は
     corridor-specific と確定
   - 2-obstacle compound (north + east): 両 planner ground、MPC mean_t=56 s
     (timeout dominance)
   - ただし commit `1646e11` の multi-runner freeze fix 後、この系列は
     runner bug の影響を受けていた可能性が高い。修正後再走では
     race / gates / dyn4 / chaos 系が全 planner 100 % collision に落ちた。
     §3 Table 2 と Smart MPPI 系の paper claim は、再設計・再走まで
     artifact として扱う。

4. **論文 §3 restructure** (commits `18b80d6`, `0e4592f`, `159410f`,
   `e760598`):
   - §3 headline を N=4 baseline に scope し、"Scope of the headline claim"
     パラグラフで (N, density) grid 全体図を提示
   - §3 に "Dynamic-obstacle extension" subsection (Table 2) を追加
   - §3 "Combined reading" を 3-mode taxonomy
     (clustering / cancellation / sign-reversal) に再構成
   - §4.4.4 / §6 / §7 / outline.md を整合

5. **AirSim base_ew06 variability** (commits `16cdbcd`, `c87c5e5`):
   - n=50 paired (seeds 42-91) で Δ-flip 符号反転を確定 (MPC clusters)
   - 3 fresh batch × n=15 で McNemar 方向が batch ごと逆転、combined p=0.77
   - 結論: cluster *mode* は robust、絶対 McNemar 方向は env-dependent

**残された open work** (優先度の高い順):

1. ~~**§3 dynamic-obstacle: off-corridor gradient probe**~~ — 完了
   (2026-05-19, commit `2a2b8bf`)。x ∈ {17, 18, 19} を走った結果、
   planner ロールが **非単調反転**: offset 0 (GPU 崩壊)、offset 1
   (tied)、offset 2 (**MPC 崩壊**)、offset 3+ (§3 baseline 復元)。
   Mechanism statement: softmax は argmin が wrong side に commit する
   場面で救う / 正しい側がない場面で崩壊する。

2. ~~**§1 / §2 / §5 を §3 の 3-mode 整理に整合させる**~~ — §1 / §2 完了
   (2026-05-19, commit `49eec56`)。§5 secondaries は §3 と独立した
   ablation (escape volume / peer prediction) なので未更新で OK。
   要件が出たら更新。

3. **AirSim dynamic obstacle 再現** — bridge 実装完了
   (2026-05-19)、smoke 動作確認済 (`exp_airsim_multi_dyn_n5_smoke.yaml`)。
   `airsim_bridge.py` に `_sync_dynamic_obstacles_initial` /
   `_update_dynamic_obstacle_poses` 追加、`scenario.advance(dt)` 呼び出し
   も追加。Bridge は cube spawn + 移動 + drone-vs-cube collision を
   正しく扱う (MPC の detour 軌跡で確認: x 30→28, z 30→32)。
   ただし **§3 Table 2 cliff (GPU joint 86.7 → 3.3) の再現は未達**。
   AirSim default cell (60×60×40, max_speed=4) は escape volume が
   dummy_3d (40×40×12, max_speed=8) より大きく、GPU MPPI が up-detour で
   bidirectional symmetry を破る。再現には以下のいずれか必要:
   ceiling obstacle 層、max_speed=8 化、40×40×12 化、または corridor
   側面の static cube wall。Future work。なお `1646e11` 後は
   dummy_3d Table 2 自体を主証拠に使えないため、この AirSim 再現も
   「旧 Table 2 を移植する」ではなく「dynamic obstacle cell を
   再設計する」タスクとして扱う。

4. **Figure 整備** — まずは static 系だけを対象にする:
   (N, density) heatmap、§4.4.4 cluster trace、AirSim base_ew06 variability。
   §3 Table 2 cliff / off-corridor gradient は `1646e11` 後の再設計・再走が
   終わるまで figure 化しない。

5. **SAC RL ベースライン**との比較 — §3 mechanism は planner-specific
   action selection rule の話だが、learned policy も同じ mode に陥るか
   未測。`train_rl_baseline.py` の scaffold あり。

6. ~~**Plan.md の §3 中期 / §4 長期** を更新~~ — 2026-05-21 に
   `1646e11` 後の retraction を反映。次は `docs/findings.md` /
   `docs/paper_a/section_3_headline.md` の本文側を直す。

---

### 2.6 候補 F: **1646e11 後の dynamic-obstacle claim 整理** (2026-05-21)

**動機**: `uav_nav_lab/runner/multi.py` の commit `1646e11` で、
total-wipeout episode 後に dynamic obstacles が凍る bug を修正した。
修正後の再走では race / gates / dyn4 / chaos が全 planner 100 %
collision になり、旧 dynamic-obstacle headline と Smart MPPI v4-v5 の
優位主張は pre-fix artifact の可能性が高い。

**今すぐやること**:

1. `docs/findings.md` の dynamic-obstacle race / gates / dyn4 / chaos /
   Smart MPPI v1-v5 節に **invalidated by `1646e11`** の注意を入れる。
2. `docs/paper_a/section_3_headline.md` の Table 2 / 4-mode framework /
   Smart MPPI 記述を、static coordination と AirSim static transferability
   だけで矛盾しない形に縮退する。
3. README は入口として応急整理済み。`plan.md` へのリンクは置かない方針。
4. 新しい dynamic-obstacle cell は、全 planner floor にならないように
   gate gap / obstacle speed / oval size / lookahead を再設計してから
   n=30 paired に戻す。

**2026-05-21 追加進捗**: `race_simple` 系から post-fix の non-floor
pilot を発見し、`examples/exp_race_simple_retuned_n5_{mpc,gpu_mppi}.yaml`
として固定した。設定は `radius=16`, `radius_y=12`, `period=20`,
`max_steps=800`, `w_goal=0.3`, `w_obs=200`、dynamic obstacles は
旧 simple の 2 slow intruders (radius 1.0, |v|=1.5) のまま。
full-duration n=5 (seeds 42-46) 結果:

| planner | per-drone | joint | note |
|---------|-----------|-------|------|
| MPC (n=8,h=40) | 15/20 = 75 % | 0/5 | drone 1 が全 seed で collision |
| GPU MPPI (n=64,h=40) | 20/20 = 100 % | 5/5 | ceiling |

`paired_analysis_aerobatic.py` では GPU tracking RMSE も MPC より
0.139 m 低く、20/20 drone-episodes で GPU better。全 planner floor は
脱出したが、GPU ceiling + deterministic MPC drone-1 failure なので、
次は (a) n=30 確認、または (b) obstacle radius/speed/phase を少し戻して
GPU も 60-90 % 帯に落とす sweep。

追加 boundary probe:

| period | max_steps | MPC n=3 | GPU MPPI n=3 | read |
|--------|-----------|---------|--------------|------|
| 18.0 | 720 | 0/3 joint | 0/3 joint | 両 planner floor |
| 19.0 | 760 | 0/3 joint (3/12 per) | 0/3 joint | 両 planner floor |
| 19.5 | 780 | 0/3 joint (3/12 per) | 0/3 joint (3/12 per) | floor 寄り |
| 19.8 | 792 | 3/3 joint | 3/3 joint | ceiling |
| 19.9 | 796 | 3/3 joint | 3/3 joint | ceiling |
| 20.0 | 800 | 0/5 joint (15/20 per) | 5/5 joint | MPC drone 1 固定 failure |

period=20 の MPC failure は全 seed で drone 1, t=29.6s,
`collision_object=null`。dynamic obstacle 直撃というより、回避後の
drone-drone / phase geometry failure。したがってこの cell は
「全 planner floor から抜けた regression pilot」として有用だが、
そのまま dynamic-obstacle mode claim にするには knife-edge すぎる。

#### 2.6.1 period / obstacle knob probe の追加ログ (2026-05-21)

目的: `period=20` cell は GPU ceiling + MPC deterministic failure で
non-floor にはなったが、planner mechanism としては knife-edge。そこで
`period=19.8` ceiling cell と `period=19.5` floor-ish cell の両側から、
dynamic obstacle の radius / speed を少し動かして中間帯が出るか見た。

共通設定:

- base: `exp_race_simple_mpc.yaml`
- `radius=16`, `radius_y=12`, `w_goal=0.3`, `w_obs=200`
- full-duration: `max_steps = period * 2 / 0.05`
- n=3 seeds 42-44
- planner: MPC `(n=8,h=40)` vs GPU MPPI `(n=64,h=40)`

結果:

| cell | obstacle knob | MPC n=3 | GPU MPPI n=3 | read |
|------|---------------|---------|--------------|------|
| p19.8 | baseline r=1.0, v=1.5 | 3/3 joint | 3/3 joint | ceiling |
| p19.8 | r=1.2 | 3/3 joint | 3/3 joint | radius 強化では ceiling 崩れず |
| p19.8 | v=2.0 | 3/3 joint | 3/3 joint | speed 強化でも ceiling 崩れず |
| p19.5 | baseline r=1.0, v=1.5 | 0/3 joint (3/12 per) | 0/3 joint (3/12 per) | floor 寄り |
| p19.5 | r=0.8 | 0/3 joint (3/12 per) | 3/3 joint (12/12 per) | GPU-only ceiling, MPC floor-ish |
| p19.5 | r=0.9 | 0/3 joint (3/12 per) | 3/3 joint (12/12 per) | r=0.8 と同じ |
| p19.5 | v=1.0 | 3/3 joint | 3/3 joint | 両 planner ceiling |
| p19.5 | r=0.95 | partial / killed | not completed | mpc 側が長時間化。途中保存分は drone 1-3 collision 型。citation 禁止 |

読み:

1. **period が主 knob で、obstacle radius/speed は二次 knob。**
   `period=19.8` では obstacle を r=1.2 / v=2.0 にしても両 planner ceiling。
   逆に `period=19.5` では baseline が両 planner floor-ish。つまり
   hardness は obstacle 単体の clearance というより、oval phase と
   drone-drone crossing geometry の resonance で決まっている。
2. **radius は discontinuous。**
   `period=19.5` で r=1.0 は両 planner floor-ish、r=0.9 / 0.8 は
   GPU ceiling + MPC floor-ish。中間の r=0.95 は実行が長時間化して
   incomplete。ここに境界はありそうだが、滑らかな partial-success band
   ではなく seed-invariant な幾何 phase failure に見える。
3. **speed を下げると簡単になりすぎる。**
   `period=19.5, v=1.0` は両 planner ceiling。speed sweep だけで
   30-60 % collision 帯を作る見込みは薄い。
4. **現時点の best regression pilot は `period=20` または `p19.5,r=0.9`。**
   どちらも GPU-only success を作るが、dynamic-obstacle hit ではなく
   phase / drone-drone geometry separation に寄っている疑いが強い。

次の推奨:

- **n=30 に上げるなら**、paper claim ではなく regression/pilot として
  `exp_race_simple_retuned_n5_{mpc,gpu_mppi}.yaml` を n=30 化する。
  目的は「post-fix でも non-floor scenario は作れる」の確認まで。
- **mode cell を狙うなら**、period/radius/speed ではなく obstacle
  **phase / start position** を sweep する。具体的には p19.5 or p19.8
  で obstacle start y を `[5,7,9]` / `[31,33,35]` にずらし、drone-drone
  crossing ではなく obstacle proximity が collision_object / final_t に
  出る cell を探す。
- **citation rule**: `period=20` / `p19.5,r=0.9` 系は、現段階では
  "dynamic obstacle mode restored" と書かない。書けるのは
  "post-fix all-planner floor was escaped in a retuned pilot, but the
  failure source is still phase-geometry dominated" まで。

**判定**: 最優先。README と findings / paper が矛盾している状態を先に
閉じる。新規実験や figure 整備はその後。

#### 2.6.2 dynamic obstacle phase/start-y sweep (2026-05-24)

`scripts/run_race_simple_phase_sweep.py` を追加。`examples/` に
one-off YAML を増やさず、`exp_race_simple_retuned_n5_{mpc,gpu_mppi}.yaml`
をベースに period と dynamic-obstacle start y だけを `/tmp` YAML へ生成し、
dummy_3d per-drone logs から collision source を分類する。

分類ルール:

- per-drone outcome が `collision` かつ step row に `collision=true` が
  ある場合は `env`。race-simple は static obstacle なしなので、境界外で
  なければ dynamic obstacle 起点と読む。
- per-drone outcome が `collision` だが step collision が無い場合は
  `peer`。multi-runner の peer-hit 後付け collision。
- dynamic obstacle との最小 signed clearance も併記する。log row は
  pre-step true_pos + post-step collision flag なので、clearance は
  strict な contact proof ではなく近接 proxy。

短い probe 結果:

| cell | n | MPC | GPU MPPI | read |
|------|---|-----|----------|------|
| p19.5, y=5/35 | 1 | 0/1 joint, 1/4 per, env=2 peer=1, min_dyn +0.06 m | 0/1 joint, 0/4 per, env=2 peer=2, min_dyn +0.03 m | all-planner floor 側 |
| p19.8, y=5/35 | 1 | 0/1 joint, 1/4 per, env=1 peer=2, min_dyn +0.08 m | 0/1 joint, 0/4 per, env=2 peer=2, min_dyn +0.01 m | p19.8 baseline から 1 m 外側に振ると floor |
| p19.8, y=7/33 | 1 | 0/1 joint, 0/4 per, env=2 peer=2, min_dyn -0.48 m | 0/1 joint, 0/4 per, env=2 peer=2, min_dyn -0.48 m | 1 m 内側も floor、clearance は contact proxy で負 |
| p19.8, y=6/34 | 1 | 1/1 joint, 4/4 per | 1/1 joint, 4/4 per | 同じ runner path で baseline ceiling を再確認 |
| **p19.8, y=5.5/34.5** | **10** | **10/10 joint, 40/40 per** | **0/10 joint, 10/40 per, env=10 peer=20, min_dyn +0.03 m** | deterministic post-fix dynamic-contact regression cell |

重要な読み:

1. start-y は period/radius/speed よりさらに鋭い phase knob。
   p19.8 は y=6/34 で ceiling、y=5/35 と y=7/33 で floor、y=5.5/34.5
   で MPC clear / GPU dynamic-contact failure に分かれた。
2. `p19.8, y=5.5/34.5` は seed 42-51 で deterministic:
   GPU は毎 seed で drone 3 が env collision at t=29.3 s、その後 drone 2
   が peer at t=34.2 s、drone 1 が peer at t=39.0 s。MPC は全機完走。
3. これは `period=20` pilot とは逆向き (MPC loss ではなく GPU loss) だが、
   failure 起点が dynamic-obstacle proximity に戻った点で、post-fix の
   dynamic-obstacle mode candidate としては period=20 より良い。

ただし citation status はまだ **regression cell / mechanism candidate**。
n=10 でも seed-invariant で、planner 差というより deterministic
geometry × softmax-action failure に見える。paper claim にする前に、
まず mechanism trace を見る。

```bash
# GPU MPPI には system python3 側の torch が必要 (.venv には torch 無し)
python scripts/run_race_simple_phase_sweep.py \
  --n 10 --period 19.8 --y-pair 5.5,34.5 --python /usr/bin/python3
```

Mechanism trace helper:

```bash
python scripts/analyze_race_simple_phase_trace.py
```

seed 42 / drone 3 の trace:

- MPC は t=28.9 で `cmd_y=+7.85`、t=29.1 で `cmd_y=+5.33` に切り、
  obstacle 側から +y / +z に逃げて clearance を +0.34 m → +1.10 m へ戻す。
- GPU MPPI は t=28.9 で `cmd_y=+1.80` までは逃げるが、t=29.1 で
  `cmd_y=-1.51` に切り返し、clearance が +0.52 m → +0.03 m へ縮む。
- GPU の visible rollouts は t=28.7 / 28.9 / 29.1 でそれぞれ
  1/24, 1/24, 2/24 が predicted dynamic hit。つまり dynamic obstacle を
  完全に見ていないわけではない。一方、選択可視 rollout の clearance は
  +1.07 m, +1.04 m, +0.47 m と positive に評価されており、実際の閉ループ
  command sequence が接触側へ戻っている。

暫定読み: post-fix dynamic obstacle は planner に入っているが、この cell の
GPU MPPI は short-horizon constant-action rollout では clean と評価した
内側 shortcut に softmax が戻り、次 replan で dynamic obstacle 側に
切り返す。MPC は argmin/CHOMP 的に +y/+z へ強く避けるため接触しない。

Static mechanism figure (2026-05-25):

```bash
python scripts/render_race_simple_phase_mechanism.py
```

`scripts/render_race_simple_phase_mechanism.py` を追加。既存の
`p19p8_y5p5_34p5` logs だけを読み、seed 42 / drone 3 / t=28.6〜29.35 の
XY 静止図を `results/_race_simple_phase_sweep/p19p8_y5p5_34p5/mechanism_trace.png`
へ出す。図は GPU/MPC actual path、reference、GPU の visible rollouts、
t=28.7/28.9/29.1/29.25/29.3 の dynamic obstacle contact disk、GPU の
env collision point を重ねる。

default output:

- GPU replan t=29.10 の visible rollouts は 24 本中 2 本が predicted hit。
  全 rollout の最小 clearance は -0.34 m。
- ただし選択 visible rollout の clearance は +0.47 m。局所評価上は clean。
- 実閉ループの GPU window min clearance は +0.03 m at t=29.25 で、
  同じ row から collision flag が立つ (episode final_t は 29.30)。
- MPC の同 window min clearance は +0.34 m at t=29.00。

これで「dynamic obstacle を見ていない」のではなく、「visible rollout 内では
clean に見える shortcut を選ぶが、replan 後の実閉ループが obstacle contact
disk 側へ戻る」という mechanism candidate を図で固定できた。

Batch mechanism metrics (2026-05-25):

```bash
python scripts/analyze_race_simple_mechanism_batch.py
```

`scripts/analyze_race_simple_mechanism_batch.py` を追加。全 `p19p8_y*`
cell の GPU drone logs を走査し、GPU env collision があればその row、
無ければ t=28.0〜30.5 の min-clearance row を event とする。各 event で
`event_t - 0.15 s` 近傍の GPU replan を取り、次を JSON/table 化する:

- selected visible rollout の predicted dynamic-obstacle clearance。
- 同 replan 後の実閉ループ actual clearance。
- `cmd_y` が nearest dynamic obstacle から逃げる向きから obstacle 側へ
  切り返したか。
- selected rollout の first-step `dy` と実際の `cmd_y` の符号 mismatch。
- 同じ episode/drone/time window における paired MPC actual clearance。

default run は
`results/_race_simple_phase_sweep/mechanism_batch_summary.json` へ row-level
JSON を出す。claim に使う主 table は GPU env-collision rows:

| cell | GPU env rows | paired MPC success | differential clean→near | GPU flip | cmd mismatch | selected clear mean/min | GPU actual mean/min | MPC actual mean/min |
|------|--------------|--------------------|--------------------------|----------|--------------|-------------------------|---------------------|---------------------|
| p19.8, y=5.25/34.75 | 6 | 0/6 | 0/6 | 6/6 | 6/6 | -0.27/-0.81 m | +0.10/+0.05 m | +0.23/+0.01 m |
| p19.8, y=5.375/34.625 | 3 | 3/3 | 3/3 | 3/3 | 3/3 | +0.37/+0.37 m | +0.07/+0.07 m | +0.52/+0.52 m |
| p19.8, y=5.50/34.50 | 10 | 10/10 | 10/10 | 10/10 | 10/10 | +0.47/+0.47 m | +0.03/+0.03 m | +0.59/+0.59 m |
| p19.8, y=5.625/34.375 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| p19.8, y=5.75/34.25 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

読み:

1. split band の 2 cell では計 13/13 GPU env rows が
   **paired MPC は success、GPU selected rollout は positive clearance、
   しかし実閉ループは near-contact**。単発図ではなく batch metric として
   mechanism を支持する。
2. hard side `5.25/34.75` は GPU selected rollout 自体が negative clearance
   平均 (-0.27 m) なので、これは clean-shortcut mechanism ではなく
   all-planner hard geometry。
3. easy side `5.625/34.375` 以降は GPU env collision row が 0。したがって
   mechanism claim は split band に限定するのが正確。

Action provenance check (2026-05-25):

GPU MPPI に opt-in の `planner.log_action_provenance` を追加し、
`scripts/run_race_simple_phase_sweep.py --gpu-log-action-provenance` から
有効化できるようにした。runner/recorder は `replans[].planner_meta.
action_provenance` に compact JSON を保存する。既存の `rollouts` /
`best_rollout_idx` consumers はそのまま。

repro:

```bash
python scripts/run_race_simple_phase_sweep.py \
  --n 1 --period 19.8 --y-pair 5.5,34.5 \
  --planner gpu_mppi \
  --output-root results/_race_simple_action_provenance \
  --gpu-log-action-provenance --python /usr/bin/python3

python scripts/run_race_simple_phase_sweep.py \
  --n 1 --period 19.8 --y-pair 5.375,34.625 \
  --planner gpu_mppi \
  --output-root results/_race_simple_action_provenance \
  --gpu-log-action-provenance --python /usr/bin/python3

python scripts/analyze_race_simple_action_provenance.py \
  --run-dir results/_race_simple_action_provenance/p19p8_y5p5_34p5/gpu_mppi
python scripts/analyze_race_simple_action_provenance.py \
  --run-dir results/_race_simple_action_provenance/p19p8_y5p375_34p625/gpu_mppi
```

Both split cells reproduce the same provenance at the pre-contact replan
(`replan_t=29.10`):

| cell | event_t | source | cmd vs chosen | chosen vs softmax | visible rollout clearance | cmd/chosen/softmax y | argmax/argmin y | weight mass y + / - |
|------|---------|--------|---------------|-------------------|----------------------------|----------------------|-----------------|---------------------|
| p19.8, y=5.375/34.625 | 29.20 | softmax | 0.000e+00 | 0.000e+00 | +0.37 m | -1.51 m/s toward obstacle | +3.61 m/s escape | 0.381 / 0.619 |
| p19.8, y=5.50/34.50 | 29.25 | softmax | 0.000e+00 | 0.000e+00 | +0.47 m | -1.51 m/s toward obstacle | +3.61 m/s escape | 0.381 / 0.619 |

Additional common values: `weight_max=0.053`, entropy `3.13`; top weighted
sample is also argmin (`idx=18`, `cost=-999999.9`, `y=+3.61`, escape).

This resolves the command provenance: the step `cmd` is exactly
`Plan.target_velocity`, `target_velocity` is exactly the vanilla softmax
weighted action, and the clean visible rollout is the highest-weight/argmin
sample used for visualization and waypoints, **not** the actual velocity
command. The failure mechanism is therefore more precise than the earlier
wording:

> GPU MPPI sees a clean escape rollout, but vanilla softmax action aggregation
> averages enough lower-weight toward-obstacle samples to emit a toward-obstacle
> target velocity. MPC/argmin would have taken the escape sample.

Temperature counterfactual check (2026-05-25):

```bash
python scripts/race_simple_temperature_counterfactual.py \
  --n 3 \
  --temperature 0.3 --temperature 0.1 --temperature 0.001 \
  --python /usr/bin/python3
```

`scripts/race_simple_temperature_counterfactual.py` を追加。既存の
`p19p8_y5p5_34p5` vanilla baseline (`t=1.0`, n=10) を参照し、同じ
split cell で GPU MPPI の temperature だけを下げた fresh rerun を行う。
出力は `docs/data/race_simple_temperature_counterfactual.json` と
`docs/images/race_simple_temperature_counterfactual.png`。

| arm | source | joint | per-drone | env | peer | probe chosen_y | probe argmin_y | window min |
|-----|--------|-------|-----------|-----|------|----------------|----------------|------------|
| t=1.0 | existing n=10 baseline | 0/10 | 10/40 | 10 | 20 | -1.51 m/s | +3.61 m/s | +0.03 m |
| t=0.3 | fresh n=3 | 3/3 | 12/12 | 0 | 0 | +0.08 m/s | -0.48 m/s | +0.10 m |
| t=0.1 | fresh n=3 | 3/3 | 12/12 | 0 | 0 | +2.82 m/s | +4.39 m/s | +0.46 m |
| t=0.001 | fresh n=3 | 3/3 | 12/12 | 0 | 0 | -4.25 m/s | -4.25 m/s | +0.08 m |

読み:

1. 同じ post-fix split cell で `t=1.0` だけが deterministic contact。
   `t=0.3/0.1/0.001` はすべて seed 42-44 で clean completion。
2. したがって race-simple の failure は "GPU MPPI cannot handle moving
   obstacles" ではなく、vanilla temperature の softmax aggregation valley。
3. 低温側は n=3 なので paper-grade rate claim ではない。ただし既存 n=10
   baseline と同じ seed prefix を温度だけ変えて救っており、mechanism
   counterfactual としては README に出せる。

0.25 m bracket sweep (n=3, seeds 42-44):

| cell | MPC | GPU MPPI | read |
|------|-----|----------|------|
| p19.8, y=5.25/34.75 | 0/3 joint, 3/12 per, env=3 peer=6, min_dyn +0.01 m | 0/3 joint, 0/12 per, env=6 peer=6, min_dyn +0.05 m | all-planner hard side |
| p19.8, y=5.50/34.50 | 10/10 joint, 40/40 per | 0/10 joint, 10/40 per, env=10 peer=20, min_dyn +0.03 m | deterministic split point |
| p19.8, y=5.75/34.25 | 3/3 joint, 12/12 per | 3/3 joint, 12/12 per | all-planner easy side |

`5.25/34.75` は deterministic だが failure drone が `5.50` と違う:
MPC は drone 1 env at t=29.3 s → drone 0/3 peer、GPU は drone 1/3 env
at t=29.2 s → drone 0/2 peer。`5.75/34.25` は両 planner ceiling。
0.25 m bracket では seed-level partial band は見えず、hard / split /
ceiling が離散的に切り替わる。

0.125 m fine bracket sweep (n=3, seeds 42-44):

| cell | MPC | GPU MPPI | read |
|------|-----|----------|------|
| p19.8, y=5.375/34.625 | 3/3 joint, 12/12 per | 0/3 joint, 3/12 per, env=3 peer=6, min_dyn +0.07 m | split side |
| p19.8, y=5.625/34.375 | 3/3 joint, 12/12 per | 3/3 joint, 12/12 per | ceiling side |

`5.375/34.625` は `5.50/34.50` と同じ failure mode:
GPU は毎 seed で drone 3 env at t=29.25 s、その後 drone 2 peer at
t=34.05 s、drone 1 peer at t=38.90 s。MPC は全機完走。
`5.625/34.375` は両 planner ceiling。したがって hard→split の
境界は (5.25, 5.375]、split→ceiling の境界は (5.50, 5.625] にある。
まだ partial seed band は見えていない。

次にやるなら:

1. 境界をさらに詰めるなら `y=5.3125/34.6875` と
   `y=5.5625/34.4375` を n=3 で見る。ただしここまで seed-invariant なので、
   partial band 探しより mechanism 図の優先度が上がった。
2. GPU seed 42 の rollout viz を作るなら、t=28.7〜29.3 付近に絞る。
   full GIF より、dynamic obstacle / reference / selected visible rollout /
   actual closed-loop path の静止図の方が mechanism 図として読みやすい。
3. n=30 は、partial band か mechanism figure の狙いが立ってからでよい。

---

### 2.7 候補 G: **N+P rule arc を paper §5 / findings 統合** — 完了 (commit `016e031`)

**現状**:
2026-05-22..23 の 2 日で **vanilla MPPI を 1 episode の warmup で観測し、
(top-2 disagreement = top2, chosen-vs-goal angle = cvg) の (N, P) signal
から cell-optimal な temperature を predict する経験則** を確立した。
コミット系列: `00cfff4 J` (2D phase diagram) → `55d0472 K` …
`384ab1f N` → `5c2196c O` (OOD 確認) → `4ae7a52 P` → `e9704b8 Q` (5-cell 図) →
`6ae3211 R` (warmup_select_mppi planner) → `19c2e20 R'` (multi-drone pooling fix) →
`61e8709 S` (city OOD) → `566f31d T` (city 4-cell stress test) →
`366519a X` (family-selector hypothesis kill) →
`97b2ac2 U` (chokepoint scope condition explained)。

**経験則 (実装版)**:
- `top2 > 50°` → `temperature = 1.0` (chaotic rollout landscape; vanilla維持)
- `top2 ≤ 50°` かつ `cvg < 12.5°` → `temperature = 10` (uniform、平均型)
- `top2 ≤ 50°` かつ `cvg ≥ 12.5°` → `temperature = 0.1` (argmin、コミット型)

paper §5 では calibration 上の解釈として `top2 > 50-60°` / `cvg < 12.5°`
の band として書いた。旧メモの `cvg ≥ 20°` / `cvg < 8°` は N/P/Q 時点の
粗い rule-of-thumb で、`warmup_select_mppi.py` の実装値ではない。

実装: `uav_nav_lab/planner/warmup_select_mppi.py` (継承 + 1 episode warmup +
session pooling via `_SHARED_SESSIONS`)。

**X arc の honest negative → 強い positive へ reframe**:
家族選別仮説 (MPC vs MPPI を信号で予測) は 9 cell 中 **MPC が 0 cell でしか
勝たない** (dynamic obstacle 環境では) で死亡。ただしこれを「MPPI + N+P rule
が MPC を平均 35pp dominated」と書き直せる。X commit message と
`docs/findings.md` の X section にそう reframe 済み。

**U arc の scope condition**:
chokepoint cell (cvg=3.6° で uniform を推すが、実測 best は argmin 0.1)
は **N+P rule の唯一の確信誤推**。原因は geometric:
- corner walls が navigable region を 12m corridor に潰す
- 中央に 4×4×4m cube cluster を置くと forced gap が 4m
- 4m < uncertainty radius になり、両側を平均する softmax は smear into
  obstacle、argmin = commit が勝つ
- 信号 (cvg) は hit cell と区別不能 (city_v1: 5.7, city_3x3: 5.8, chokepoint: 3.6)

詳細メカニズム表は `docs/findings.md` U section + 図
`docs/images/u_chokepoint_{timeseries,geometry}.png`。

**2026-05-24 完了内容 (`016e031`)**:

1. `docs/paper_a/section_5_secondaries.md` に §5.3 / §5.4 を追加。
   - §5.3: 1 episode vanilla MPPI warmup → top2/cvg → temperature 選択。
   - §5.4: family-selector 仮説の negative result を「MPPI + N+P rule が
     MPC を 9/9 cell で平均 35pp dominate」として reframe。
2. `docs/paper_a/section_6_limitations.md` に U scope condition を追加。
   `city_chokepoint` は信号の miss ではなく、4m forced gap / corridor width の
   geometric blind spot と明記。
3. `docs/paper_a/section_7_repro_map.md` に §5.3 / §5.4 / §6 U の artifact 行を追加。
4. `docs/paper_a/outline.md` の §5 map を更新。

**残り**:
- 図の差し込み位置は manuscript assembly 時に決める。候補は
  `docs/images/n_rule_summary.png`, `warmup_select_validate.png`,
  `x_planner_family_landscape.png`, `u_chokepoint_{timeseries,geometry}.png`。
- §3 末尾へ X_planner_family_landscape を置く案は未採用。現状は §5 に閉じる方が
  構成として自然。

**判定**: 完了。次は paper draft を単一 manuscript に組み上げる段階。

### 2.8 候補 H: **Scaling-law refactor campaign — S3/V 完了、W は後回し** (2026-05-23..24)

**動機**: ユーザー要望「モジュール分割・関数分割・依存局所化・テスト単位の
分離をきちんと行って、システムを分割し、影響範囲を閉じ込め、検証単位を
細かく設計して、開発全体のスケーリング則そのものを改善して」。

**完了済 (HEAD = `016e031`)**:

| commit | 内容 | LOC delta | tests |
|---|---|---|---|
| `7abe635` R | `uav_nav_lab/analysis/` 新設 (success_rates + warmup), 5 scripts から重複統計を抽出。private `_SHARED_SESSIONS` 等を `diagnose_warmup` 公開関数の裏に隠す。 | -230 / +56 (scripts) | +11 (`tests/analysis/`) |
| `dba7c3c` T | `runner/multi/episode.py` 390 LOC を `phases.py` (`_replan_one_drone`, `_log_step_for_drone`) + `outcomes.py` (`_handoff_master`, `_resolve_outcomes`, `_finalize_timeouts`) + 残 230 LOC orchestrator に分割。 | +231 / -177 | 既存 pass |
| `e36da4c` S | `planner/_grid.py` に `point_to_cell` / `point_is_occupied` / `mask_dynamic_obstacle_cells` 集約。`planner/mppi.py` 317 LOC → `mppi/{__init__, planner, rollout, aggregator}.py`。 | +285 / -191 | 既存 pass + smoke bit-exact |
| `1c84fcf` S2 | `planner/mpc.py` 299 LOC → `mpc/{__init__, planner, aggregator}.py`。**追加 payoff**: mpc/mppi 重複の 70 LOC `score_rollouts` を `planner/_rollout.py` に shared 抽出。 | +114 / -100 | 既存 pass + smoke bit-exact |
| `81ca0cc` S3/V | `planner/chomp.py` 334 LOC → `chomp/{__init__, planner, objective}.py`。`mpc_chomp.py` の private helper import を public objective API へ変更。`tests/planner/conftest.py` に registry / empty-grid fixture を集約。 | +441 / -450 | pytest 全件 + smoke bit-exact |

**結果**:
- `planner/` 下の MPC と MPPI は **コード差 = aggregator (argmin vs softmax) のみ**。
  rollout カーネルへの将来の修正は両方に伝播。
- `runner/multi/` は 6 module 構成 (`builder`, `peers`, `phases`, `outcomes`,
  `episode`, `experiment`) に分離。`episode.py` は orchestrator only。
- `analysis/` 経由で scripts は private surface に依存しなくなった。
- `chomp/` は package 化され、CHOMP-family の数学的 objective は
  `chomp.objective` の public API になった。planner subpackage を跨ぐ private
  helper import は現時点で解消。
- `tests/planner/` は固定サイズの空 occupancy と registry fixture を
  `conftest.py` に集約。テスト数は増やさず、今後の planner test 追加コストを下げた。

**残タスク**:

#### 2.8.1 候補 S3: chomp.py 334 LOC 分割 + mpc_chomp 依存切れ — 完了 (`81ca0cc`)

実装通りに完了:
- `uav_nav_lab/planner/chomp/__init__.py` — `ChompPlanner` re-export。
- `uav_nav_lab/planner/chomp/planner.py` — class + orchestration + `_resample_polyline`。
- `uav_nav_lab/planner/chomp/objective.py` — `distance_field`,
  `obstacle_cost_and_grad`, `smoothness_hessian` を public function として提供。
- `uav_nav_lab/planner/mpc_chomp.py` — `from .chomp.objective import ...` に変更。

検証:
- focused: `pytest -q tests/planner/test_chomp.py tests/planner/test_mpc_chomp.py`
- full: `pytest -q`
- smoke: `scripts/x_planner_family_gather.py` 後の
  `docs/data/x_planner_family_data.json` が baseline と identical。

#### 2.8.2 候補 V: tests/conftest 共通化 — 完了 (`81ca0cc`)

`tests/planner/conftest.py` を追加し、以下を共有 fixture 化:
- `planner_registry`
- `empty_grid`
- `empty_grid_20`
- `empty_grid_30`

固定 20×20 / 30×30 空 occupancy の重複を CHOMP / MPC-CHOMP / MPC / MPPI /
RRT / warmup-select MPPI tests から除去。障害物を置く test は各 test 内で
引き続き明示的に mutate しており、behavioral surface は変えていない。

#### 2.8.3 候補 W: gpu_mppi 構造との一貫性チェック

`uav_nav_lab/planner/gpu_mppi/` は既に `planner/rollout/aggregator/ctg_cache`
構成だが、S2 で mpc/mppi が共有した `_rollout.py` は CPU 版のみ。
GPU 版 (`cuda.jit` カーネル) はインターフェース互換でも実装が別。
**一貫させるか別物として残すかの判断**は未決定。

選択肢:
- (a) `_rollout.py` を CPU 専用と明示し、gpu_mppi はそのまま (= 現状)。
- (b) `planner/rollout/{cpu,gpu}.py` に二段化、共通 cost-shape を docstring
  として明文化。
- (c) gpu_mppi の rollout インターフェースを CPU と揃え、片方を呼び替える
  helper を用意。

GPU 周りに触ると goal-mask バグ (§0.5 参照) を再発するリスクがあるため
**v0.3 公開後に判断**で良い。短期 todo には入れない。

#### 2.8.4 今後の判断

短期 refactor campaign は R/T/S/S2/S3/V で一旦閉じる。W は
`gpu_mppi` goal-mask 周辺の再発リスクが高く、論文 v1 / v0.3 公開後に判断で良い。
次にコードを触るなら、refactor より **post-fix dynamic-obstacle cell 再設計**か、
manuscript assembly に必要な軽い script/doc 整備を優先する。

---

## 3. 中期 (次の 5-10 PR)

### 3.1 AirSim Δ-flip シリーズの完結

候補 A は 2026-05-19 に dummy_3d 側で実質完結 (§2.5 参照)。残るのは
AirSim 側の対応:

- [x] **静的障害物密度 sweep** — dummy_3d で完了。AirSim base_ew06 は
      1 cell のみで N=4 dense regime と一致。AirSim 側の density grid は
      cost に対し追加情報が薄いので保留。
- [ ] **dynamic obstacle 速度 sweep の再設計** — 旧 dummy_3d sweep は
      `1646e11` 前の runner bug 影響を受けていたため claim から外す。
      まず findings / paper の retraction を入れ、その後に全 planner floor
      にならない cell を再設計する。AirSim Blocks 側の moving cube path は
      bridge 実装済みだが、cross-sim 化は新 cell が固まってから。
- [x] **N=2, 3, 4, 6 で AirSim multi-drone N-scaling** — dummy_3d で
      N ∈ {2..12} まで完了。AirSim 側は base_ew06 N=4 1 cell のみ。
      N-scaling の AirSim 移植も cost vs payoff で見て保留。
- [ ] **AirSim dynamic obstacle 再現**: 旧 dummy_3d Table 2 cliff の
      移植ではなく、`1646e11` 後に再設計した dynamic cell を AirSim に
      移す。優先度 中、§4.4 transferability の補強候補。

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
