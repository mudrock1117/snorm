#!/usr/bin/env python3
"""
Hyperparameter grid search. Each combo runs MAX_ITERS, auto-eval every 1000 iters.
After each run, best sum_mAP is scraped and logged.

Usage:
    python run_hparam_grid.py           # launch ALL combos sequentially (nohup per combo)
    python run_hparam_grid.py --dry-run # print plan only
    python run_hparam_grid.py --run-now # blocking: run first combo and wait for result
    python run_hparam_grid.py --collect # no launch, just scrape existing work_dirs
"""
import os, sys, subprocess, argparse, re, time, shlex
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE_DIR / "train_single_c2f_bn_color_0112_render_ahdr.py"
BASE_CONFIG = BASE_DIR / "debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml"
BASE_BEST_PTH = BASE_DIR / "debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/best.pth"
GRID_ROOT = BASE_DIR / "debug/hparam_grid"

PYTHON = Path("/home/ubuntu/anaconda3/envs/domain/bin/python")
CUDA_DEVICES = "1"

MAX_ITERS = 5000
TRAIN_ITERS_PER_EPOCH = 19318

ENABLE_RESUME = True

GRID = [
    {"SOLVER.LR": 5e-5,  "INPUT.min_size": 512, "tag": "lr5e-5_res512"},
    {"SOLVER.LR": 5e-5,  "INPUT.min_size": 600, "tag": "lr5e-5_res600"},
    {"SOLVER.LR": 5e-5,  "INPUT.min_size": 640, "tag": "lr5e-5_res640"},
    {"SOLVER.LR": 1e-4,  "INPUT.min_size": 512, "tag": "lr1e-4_res512"},
    {"SOLVER.LR": 1e-4,  "INPUT.min_size": 600, "tag": "lr1e-4_res600"},
    {"SOLVER.LR": 1e-4,  "INPUT.min_size": 640, "tag": "lr1e-4_res640"},
    {"SOLVER.LR": 2e-4,  "INPUT.min_size": 512, "tag": "lr2e-4_res512"},
    {"SOLVER.LR": 2e-4,  "INPUT.min_size": 600, "tag": "lr2e-4_res600"},
    {"SOLVER.LR": 2e-4,  "INPUT.min_size": 640, "tag": "lr2e-4_res640"},
]


def build_yaml(combo):
    from yacs.config import CfgNode
    cfg = CfgNode(new_allowed=True)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(BASE_CONFIG))

    for k, v in combo.items():
        if k == "tag":
            continue
        if k.startswith("INPUT.") and "min_size" in k:
            ms = int(v)
            cfg.INPUT.TRANSFORMS_TRAIN = [
                {'name': 'random_flip'},
                {'name': 'resize', 'min_size': ms},
                {'name': 'normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'to_01': True},
                {'name': 'collect'},
            ]
            cfg.INPUT.TRANSFORMS_TEST = [
                {'name': 'resize', 'min_size': ms},
                {'name': 'normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'to_01': True},
                {'name': 'collect'},
            ]
        elif k.startswith("SOLVER."):
            sub = cfg.SOLVER
            for sk in k.split(".")[1:-1]:
                sub = getattr(sub, sk)
            setattr(sub, k.split(".")[-1], v)
        else:
            sub = cfg
            for sk in k.split(".")[:-1]:
                sub = getattr(sub, sk)
            setattr(sub, k.split(".")[-1], v)

    cfg.SOLVER.MAX_ITERS_EARLY_STOP = MAX_ITERS
    cfg.SOLVER.EPOCHS = 3

    return cfg


def launch(combo, dry_run=False, blocking=False):
    tag = combo["tag"]
    work_dir = GRID_ROOT / tag
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_yaml(combo)
    cfg.WORK_DIR = str(work_dir)
    cfg_path = work_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg.dump())

    log_path = work_dir / "train.log"
    eval_dir = work_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    resume_arg = f"--resume {BASE_BEST_PTH} " if ENABLE_RESUME and BASE_BEST_PTH.exists() else ""
    cmd = (
        f"cd {BASE_DIR} && "
        f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} "
        f"nohup {PYTHON} {TRAIN_SCRIPT} "
        f"--config-file {cfg_path} "
        f"{resume_arg}"
        f"> {log_path} 2>&1 &"
    )

    if dry_run:
        print(f"  [DRY] {tag}: LR={cfg.SOLVER.LR}, resize={combo.get('INPUT.min_size',600)}, WD={cfg.SOLVER.WEIGHT_DECAY}")
        print(f"         yaml={cfg_path}")
        if BASE_BEST_PTH.exists():
            print(f"         resume={BASE_BEST_PTH}")
        return None

    print(f"[RUN] {tag}: launching (LR={cfg.SOLVER.LR}, resize={combo.get('INPUT.min_size')}, WD={cfg.SOLVER.WEIGHT_DECAY}, MAX_ITERS={MAX_ITERS}, CUDA={CUDA_DEVICES})")
    print(f"  work_dir = {work_dir}")

    if blocking:
        args_list = [str(PYTHON), str(TRAIN_SCRIPT), "--config-file", str(cfg_path)]
        if BASE_BEST_PTH.exists():
            args_list += ["--resume", str(BASE_BEST_PTH)]
        subprocess.run(
            args_list, cwd=str(BASE_DIR), check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": CUDA_DEVICES},
        )
    else:
        subprocess.run(cmd, shell=True, check=False, cwd=str(BASE_DIR))

    return work_dir


def scrape_best(work_dir: Path):
    best_txt = work_dir / "best.txt"
    if best_txt.exists():
        text = best_txt.read_text().strip()
        m = re.search(r"sum_mAP:\s*([0-9.]+)\s*\(foggy:\s*([0-9.]+),\s*night:\s*([0-9.]+)\)", text)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def wait_for_result(work_dir: Path, poll=30):
    tag = work_dir.name
    print(f"[WAIT] {tag} polling every {poll}s ...")
    t0 = time.time()
    last_best = None
    last_step = 0
    while True:
        ps = subprocess.run(
            ["bash", "-c", f"pgrep -f '{tag}/config.yaml' | wc -l"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=str(BASE_DIR)
        )
        nproc = int((ps.stdout or "0").strip() or "0")

        log_path = work_dir / "train.log"
        if log_path.exists():
            tail = subprocess.run(["tail", "-n", "30", str(log_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout
            m_best = re.findall(r"Best sum mAP:\s*([0-9.]+)\s*\(foggy:\s*([0-9.]+),\s*night:\s*([0-9.]+)\)", tail)
            if m_best:
                latest = m_best[-1]
                cur = (float(latest[0]), float(latest[1]), float(latest[2]))
                if last_best is None or cur != last_best:
                    print(f"  [LIVE] sum={cur[0]:.4f} fog={cur[1]:.4f} night={cur[2]:.4f}")
                    last_best = cur

            m_step = re.findall(r"Epoch:\s*\[\d+\]\s+\[\s*(\d+)/", tail)
            if m_step and int(m_step[-1]) != last_step:
                print(f"  [PROG] iter {m_step[-1]}/{TRAIN_ITERS_PER_EPOCH}")
                last_step = int(m_step[-1])

            if "EARLY-STOP" in tail:
                print(f"  [DONE] early-stop triggered")
                break

        if nproc == 0 and (work_dir / "best.txt").exists():
            print(f"  [DONE] process exited + best.txt saved")
            break
        if nproc == 0:
            print(f"  [WARN] process exited but no best.txt yet")
            time.sleep(30)
            break

        time.sleep(poll)

    return scrape_best(work_dir)


def print_summary(results, ref_best=0.8118):
    print("\n" + "=" * 85)
    print("  SUMMARY  (ref = current best 0.8118)")
    print("=" * 85)
    print(f"  {'tag':<25} {'sum_mAP':>9} {'fog':>9} {'night':>9} {'delta':>8} {'status':>8}")
    print("-" * 85)
    rows = []
    for tag, s, f, n in results:
        if s is None:
            rows.append((tag, None, None, None))
            print(f"  {tag:<25} {'(pending)':>9} {'':>9} {'':>9} {'':>8} {'running':>8}")
        else:
            delta = s - ref_best
            sign = "+" if delta >= 0 else ""
            rows.append((tag, s, f, n))
            status = "BETTER" if s > ref_best + 1e-4 else "SAME" if s > ref_best - 1e-4 else "WORSE"
            print(f"  {tag:<25} {s:>9.4f} {f:>9.4f} {n:>9.4f} {sign}{delta:>7.4f} {status:>8}")
    print("=" * 85)

    csv_path = GRID_ROOT / "grid_summary.csv"
    with open(csv_path, "w") as f:
        f.write("tag,sum_mAP,fog,night\n")
        for tag, s, fog, nt in rows:
            if s is not None:
                f.write(f"{tag},{s:.4f},{fog:.4f},{nt:.4f}\n")
            else:
                f.write(f"{tag},,,\n")
    print(f"\nCSV saved: {csv_path}")

    completed = [(tag, s) for tag, s, _, _ in rows if s is not None]
    if completed:
        best_tag, best_score = max(completed, key=lambda x: x[1])
        print(f"\n  BEST so far: {best_tag}  sum_mAP={best_score:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-now", action="store_true", help="blocking: run combos one by one, wait for each result")
    parser.add_argument("--collect", action="store_true", help="no launch, just scrape existing")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(GRID))
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--sequential", action="store_true", help="launch one at a time (default: all at once)")
    args = parser.parse_args()

    GRID_ROOT.mkdir(parents=True, exist_ok=True)

    launched_combos = [GRID[i] for i in range(args.start, min(args.end, len(GRID)))]
    print(f"\nHyperparameter grid: {len(launched_combos)} combos (range [{args.start},{min(args.end,len(GRID))-1}]), MAX_ITERS={MAX_ITERS}, backup at ../backup_pre_hparam_20260910_163752\n")

    results = []
    for i, combo in enumerate(launched_combos, start=args.start):
        print(f"----- [{i}/{len(GRID)-1}] {combo['tag']} -----")
        wd = launch(combo, dry_run=args.dry_run, blocking=args.run_now and not args.sequential)
        if args.dry_run:
            continue
        if args.run_now:
            s, f, n = wait_for_result(wd, poll=args.poll)
            results.append((wd.name, s, f, n))
            print_summary(results)
            if i < args.end - 1:
                print("\n--- starting next ---\n")
        else:
            results.append((wd.name, None, None, None))

    if args.dry_run:
        print(f"\n[DRY-RUN] {len(launched_combos)} combos, nothing launched.")
        return

    if args.collect:
        results = []
        for combo in launched_combos:
            wd = GRID_ROOT / combo["tag"]
            s, f, n = scrape_best(wd)
            results.append((combo["tag"], s, f, n))
        print_summary(results)
        return

    if not args.run_now and not args.dry_run:
        print(f"\n[LAUNCHED] {len(launched_combos)} processes running in background.")
        print(f"  tail -f {GRID_ROOT}/*/train.log")
        print(f"  python run_hparam_grid.py --collect  (scrape results when ready)")

        results = []
        for combo in launched_combos:
            wd = GRID_ROOT / combo["tag"]
            s, f, n = scrape_best(wd)
            results.append((combo["tag"], s, f, n))
        print_summary(results)


if __name__ == "__main__":
    main()