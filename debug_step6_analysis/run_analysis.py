import os, pickle, glob, json

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

def load_frames(src_dir):
    files = sorted(glob.glob(os.path.join(src_dir, 'feats', 'step6_data_*.pkl')))
    return [pickle.load(open(f, 'rb')) for f in files]

dbs = {
    'src4': load_frames(os.path.join(BASE, 'step6_src4')),
    'src5': load_frames(os.path.join(BASE, 'step6_src5')),
}

print(f"Loaded src4={len(dbs['src4'])} frames, src5={len(dbs['src5'])} frames")

lines = []
def p(s=''):
    print(s); lines.append(s)

p("=" * 60)
p("SNorm Step6 Ablation Analysis Summary")
p("=" * 60)

p()
p("=== 1. 特征池 mem_stats (跨帧平均, src4/src5 应完全相同) ===")
for k in ['mem_0', 'mem_1', 'mem_2', 'mem_3', 'mem_4', 'mem_5']:
    vals = [d['mem_stats'][k] for d in dbs['src4']]
    m_mean = sum(v['mean'] for v in vals) / len(vals)
    m_std  = sum(v['std']  for v in vals) / len(vals)
    m_min  = sum(v['min']  for v in vals) / len(vals)
    m_max  = sum(v['max']  for v in vals) / len(vals)
    tag = '[RAW]' if k == 'mem_5' else ''
    p(f"  {k}: mean={m_mean:.4f}  std={m_std:.4f}  min={m_min:.4f}  max={m_max:.4f}  {tag}")

p()
p("=== 2. Step 输出演化 (跨帧平均) ===")
p(f"  {'step':>4s}  {'src4_mean':>10s} {'src4_std':>9s}  | {'src5_mean':>10s} {'src5_std':>9s}  | {'Δmean':>7s} {'Δstd':>6s}")
for s in range(1, 7):
    sk = f'step{s}_stats'
    m4 = sum(d[sk]['mean'] for d in dbs['src4']) / len(dbs['src4'])
    m5 = sum(d[sk]['mean'] for d in dbs['src5']) / len(dbs['src5'])
    s4 = sum(d[sk]['std']  for d in dbs['src4']) / len(dbs['src4'])
    s5 = sum(d[sk]['std']  for d in dbs['src5']) / len(dbs['src5'])
    p(f"  {s:>4d}  {m4:>10.4f} {s4:>9.4f}  | {m5:>10.4f} {s5:>9.4f}  | {m5-m4:>+7.4f} {s5-s4:>+6.4f}")

p()
p("=== 3. Step6 参考特征 (跨帧平均) ===")
for name, dlist in dbs.items():
    ref_mean = sum(d['ref6_stats']['mean'] for d in dlist) / len(dlist)
    ref_std  = sum(d['ref6_stats']['std']  for d in dlist) / len(dlist)
    src_idx  = dlist[0]['step6_src_idx']
    kern = dlist[0].get('kernel_sz_set', [])
    p(f"  {name}: ref6=mem[{src_idx}] mean={ref_mean:.4f} std={ref_std:.4f}  kernel_set={kern}")

p()
p("=== 4. step5→step6 变化 ===")
for name, dlist in dbs.items():
    s5_m = sum(d['step5_stats']['mean'] for d in dlist) / len(dlist)
    s6_m = sum(d['step6_stats']['mean'] for d in dlist) / len(dlist)
    s5_s = sum(d['step5_stats']['std']  for d in dlist) / len(dlist)
    s6_s = sum(d['step6_stats']['std']  for d in dlist) / len(dlist)
    p(f"  {name}: mean {s5_m:.4f}→{s6_m:.4f} (Δ{s6_m-s5_m:+.4f})  std {s5_s:.4f}→{s6_s:.4f} (Δ{s6_s-s5_s:+.4f})")

p()
p("=== 5. Detect mAP (bdd_night_sun_test) ===")
p("  src=4 (mem[4]=[19,19] norm):  mAP=0.3891  mREC=0.7079")
p("  src=5 (mem[5]=raw input):     mAP=0.4188  mREC=0.7334")
p("  Δ:                             +0.0297    +0.0255")

p()
p("=== 6. Per-class AP 差异 ===")
p("  class    src4_AP   src5_AP   ΔAP")
class_ap = {
    'bus':   (0.4376, 0.4572),
    'bike':  (0.4013, 0.4457),
    'car':   (0.5995, 0.6213),
    'motor': (0.1380, 0.1711),
    'person':(0.4372, 0.4576),
    'rider': (0.2760, 0.3184),
    'truck': (0.4342, 0.4604),
}
for cls, (a4, a5) in class_ap.items():
    p(f"  {cls:<7s}  {a4:>7.4f}  {a5:>7.4f}  {a5-a4:>+7.4f}")

p()
p("=" * 60)
p("KEY FINDING: Using raw input (mem[5]) as step6 reference instead of")
p("large-kernel norm feature (mem[4]) increases feature mean by ~0.04,")
p("boosts feature std by ~0.01, and improves night mAP by +0.0297.")
p("Small object classes (bike +0.044, motor +0.033, rider +0.042)")
p("benefit most, consistent with raw texture detail preservation.")
p("=" * 60)

with open(os.path.join(OUT_DIR, 'step6_analysis_summary.txt'), 'w') as f:
    f.write('\n'.join(lines))

summary = {
    'src4_step6_mean':  sum(d['step6_stats']['mean'] for d in dbs['src4'])/len(dbs['src4']),
    'src5_step6_mean':  sum(d['step6_stats']['mean'] for d in dbs['src5'])/len(dbs['src5']),
    'delta_mean':       sum(d['step6_stats']['mean'] for d in dbs['src5'])/len(dbs['src5'])
                      - sum(d['step6_stats']['mean'] for d in dbs['src4'])/len(dbs['src4']),
    'src4_step6_std':   sum(d['step6_stats']['std']  for d in dbs['src4'])/len(dbs['src4']),
    'src5_step6_std':   sum(d['step6_stats']['std']  for d in dbs['src5'])/len(dbs['src5']),
    'delta_std':        sum(d['step6_stats']['std']  for d in dbs['src5'])/len(dbs['src5'])
                      - sum(d['step6_stats']['std']  for d in dbs['src4'])/len(dbs['src4']),
    'mem5_vs_mem4_mean': dbs['src4'][0]['mem_stats']['mem_5']['mean'] - dbs['src4'][0]['mem_stats']['mem_4']['mean'],
    'night_mAP_src4':   0.3891,
    'night_mAP_src5':   0.4188,
    'night_mAP_delta':  0.0297,
    'class_AP_delta':   {c: b-a for c,(a,b) in class_ap.items()},
}
with open(os.path.join(OUT_DIR, 'step6_analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

p()
p(f"Saved summary to {OUT_DIR}")