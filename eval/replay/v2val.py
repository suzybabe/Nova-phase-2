"""Self-contained validation of Team NOVA's round 2 candidate v2 (model_angle2) vs alternatives.
Run: /opt/techarena/venvs/torch/bin/python v2val.py --data ~/r2data   [--cuts pre day2 day3 mid day4 day5] [--quick N] [--boot 1000]
Needs only numpy/pandas/scipy/torch and the files in this folder. Writes nothing outside a temp dir and --out.
Judge = conditional logit on the REAL R1 pooled slates in feedback.jsonl (copy of r2/vm/scorer.py, 17.79 vs official 17.49 on R1).
keep = observed pooled slate + our pick ; drop = our old R1 item (nova_user) removed first.
Judges: J_pre = fit on R1 clicks before the cut (fully forward, primary), J_all = fit on all R1 clicks, J_post = fit on test-period clicks."""
import argparse, hashlib, importlib.util, json, os, shutil, sys, tempfile, time, zipfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scorer_core import HabitIndex, ChoiceModel
W0, W1, DAY = 1736380800, 1736812800, 86400
G = 'itm_26ec3536f5917e5e62dce7312387e0d5'
V2_SHA = '0794adf95caf515ca27fe2b321d8da51bda00efe8cb44d96fd3f23c52d6ae9d1'
CANDS = ['v2', 'v4c', 'final', 'susan_v5', 'lea_v4']
T_START = time.time()
def log(*a): print(f'[{time.time()-T_START:6.0f}s]', *a, flush=True)
def load_mod(name):
    p = os.path.join(HERE, 'cands', name, 'solution.py')
    s = importlib.util.spec_from_file_location(f'sol_{name}_{time.time_ns()}', p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def run(name, d):
    import contextlib, io
    m = load_mod(name); t = time.time(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): P = pd.DataFrame([x for b in m.predict(d) for x in b], columns=['rowid', 'itemID'])
    return P, time.time() - t

# ---------------------------------------------------------------- robustness on the real R2 input
def robustness(data):
    log('== ROBUSTNESS (v2 on the round 2 input)')
    z = os.path.join(HERE, 'cands', 'v2', 'model.zip'); sha = hashlib.sha256(open(z, 'rb').read()).hexdigest()
    inzip = zipfile.ZipFile(z).read('solution.py'); same = inzip == open(os.path.join(HERE, 'cands', 'v2', 'solution.py'), 'rb').read()
    log(f'model.zip sha256 {sha[:16]}.. {"OK" if sha == V2_SHA else "MISMATCH"}; zip contents {zipfile.ZipFile(z).namelist()}; solution.py == bundled copy: {same}')
    cat = pd.read_csv(os.path.join(data, 'catalog.csv'), dtype={'itemID': str}).set_index('itemID'); x = pd.read_csv(os.path.join(data, 'x_pred.csv'), dtype={'rowid': str, 'user_id': str})
    def chk(P, xp, lab, rt):
        p = P.merge(xp, on='rowid', how='left'); t = p.timestamp.values; inc = p.itemID.isin(cat.index).values
        live = inc & (cat.valid_from.reindex(p.itemID).values <= t) & (t < cat.valid_to.reindex(p.itemID).values)
        ok = len(P) == len(xp) and P.rowid.nunique() == len(xp) and set(P.rowid) == set(xp.rowid) and live.all()
        log(f'  {lab:34s} rows {len(P)} unique {P.rowid.nunique()}/{len(xp)} not-in-cat {(~inc).sum()} not-live {(~live).sum()} distinct-items {P.itemID.nunique()} {rt:5.2f}s  {"PASS" if ok else "FAIL"}')
        return P.sort_values('rowid').reset_index(drop=True)
    tmp = tempfile.mkdtemp(prefix='v2val_rob_')
    try:
        P1, r = run('v2', data); P1 = chk(P1, x, 'as shipped (parquet if present) #1', r)
        P2, r = run('v2', data); P2 = chk(P2, x, 'as shipped #2', r); log(f'  deterministic: {P1.equals(P2)}')
        j = os.path.join(tmp, 'jsonl'); os.makedirs(j)
        for f in ['catalog.csv', 'x_pred.csv', 'feedback.jsonl']: shutil.copy(os.path.join(data, f), j)
        P3, r = run('v2', j); P3 = chk(P3, x, 'jsonl-only (no parquet/long)', r); log(f'  jsonl-only picks identical to parquet run: {P1.equals(P3)}')
        s = os.path.join(tmp, 'small'); os.makedirs(s)
        for f in ['catalog.csv', 'feedback.jsonl']: shutil.copy(os.path.join(data, f), s)
        xs = x.sample(7, random_state=1).copy(); xs.loc[xs.index[:3], 'user_id'] = ['usr_unknown1', 'usr_unknown2', 'usr_unknown3']; xs.to_csv(os.path.join(s, 'x_pred.csv'), index=False)
        P4, r = run('v2', s); chk(P4, xs, '7 rows incl. 3 unknown users', r)
        n = os.path.join(tmp, 'nofb'); os.makedirs(n); shutil.copy(os.path.join(data, 'catalog.csv'), n); x.head(50).to_csv(os.path.join(n, 'x_pred.csv'), index=False)
        try: P5, r = run('v2', n); chk(P5, x.head(50), 'no feedback file at all', r)
        except Exception as e: log('  no feedback file: CRASH', repr(e))
        for nm in CANDS[1:]:
            try:
                Pn, r = run(nm, data); Pn = chk(Pn, x, f'R2 {nm}', r); log(f'    {nm}: rows != v2 {int((Pn.itemID.values != P1.itemID.values).sum())}')
            except Exception as e: log(f'  R2 {nm}: CRASH', repr(e))
        log(f'  top item share {P1.itemID.value_counts(normalize=True).iloc[0]:.3f}; most common picks: ' + ', '.join(f'{k[:12]} {v}' for k, v in P1.itemID.value_counts().head(3).items()))
    finally: shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- folds
def write_cut(fb, te, cat1, c, d):
    os.makedirs(d, exist_ok=True); cut = fb[fb.timestamp < c]
    with open(os.path.join(d, 'feedback.jsonl'), 'w') as f:
        for r in cut.itertuples():
            f.write(json.dumps({'clicked_item': r.clicked_item if isinstance(r.clicked_item, str) else None, 'displayed_items': list(r.displayed_items),
                                'event_id': r.event_id, 'timestamp': int(r.timestamp), 'user_id': r.user_id}) + '\n')
    lg = cut[['event_id', 'user_id', 'timestamp', 'displayed_items', 'clicked_item']].explode('displayed_items')
    lg['clicked'] = (lg.displayed_items == lg.clicked_item).astype(int)
    lg.rename(columns={'displayed_items': 'itemID'})[['event_id', 'user_id', 'timestamp', 'itemID', 'clicked']].to_csv(os.path.join(d, 'feedback_long.csv'), index=False)
    te[['rowid', 'user_id', 'timestamp']].to_csv(os.path.join(d, 'x_pred.csv'), index=False); cat1.reset_index().to_csv(os.path.join(d, 'catalog.csv'), index=False)
    return cut
def rule_picks(xp, cut, cat1, how, fallback, decay_h=None):
    pre = cut[cut.clicked_item.notna()][['user_id', 'clicked_item', 'timestamp']]
    c = pre.join(cat1, on='clicked_item'); d = xp.merge(c, on='user_id', suffixes=('', '_c'))
    d = d[(d.timestamp_c < d.timestamp) & (d.valid_from <= d.timestamp) & (d.timestamp < d.valid_to)].copy()
    if how == 'recent': d['s'] = d.timestamp_c.astype(float)
    else:
        d['w'] = 1.0 if decay_h is None else 0.5 ** ((d.timestamp - d.timestamp_c) / 3600 / decay_h)
        g = d.groupby(['rowid', 'clicked_item']); d['s'] = g.w.transform('sum') + 1e-12 * g.timestamp_c.transform('max')
        if how == 'lea': d['s'] = g.w.transform('sum') * 1e7 + d.clicked_item.map(pre.clicked_item.value_counts()).fillna(0)
    best = d.sort_values('s').groupby('rowid').tail(1).set_index('rowid').clicked_item
    out = xp.rowid.map(best)
    if isinstance(fallback, pd.Series): out = out.fillna(xp.rowid.map(fallback))
    return out.values
def lea_fallback(xp, cut, cat1):
    pop = cut.clicked_item.value_counts(); ci = cat1.reindex(pop.index); vf, vt = ci.valid_from.values, ci.valid_to.values
    return pd.Series([pop.index[((vf <= t) & (t < vt)).argmax()] for t in xp.timestamp.values], index=xp.rowid.values)
def segments(xp, cut, cat1):
    pre = cut[cut.clicked_item.notna()]; c = pre[['user_id', 'clicked_item']].drop_duplicates().join(cat1, on='clicked_item')
    d = xp.merge(c, on='user_id'); d = d[(d.valid_from <= d.timestamp) & (d.timestamp < d.valid_to)]
    nl = d.groupby('rowid').size().reindex(xp.rowid).fillna(0).values; u = pd.Series(xp.user_id.values)
    return np.where(nl >= 2, 'multi-live', np.where(nl == 1, 'single-live', np.where(u.isin(set(pre.user_id)), 'priors-expired',
                    np.where(u.isin(set(cut.user_id)), 'shown-never-clk', 'unseen'))))

class Scorer:
    """Vectorised pooled-slate scoring for many candidates x slate scenarios x judges over one test set."""
    SCEN = ['keep', 'drop', 'no-global-top', '+recent-reclick team', '+most-clicked team', '+both reclick teams', 'no other reclickers']
    def __init__(self, te, picks, hab, old, extra):
        rows = []
        for i, (u, t, sl) in enumerate(zip(te.user_id.values, te.timestamp.values, te.displayed_items.values)):
            obs = set(sl); allit = set(obs)
            for p in picks.values(): allit.add(p[i])
            for e in extra.values():
                if isinstance(e[i], str): allit.add(e[i])
            for it in allit: rows.append((i, it, it in obs))
        L = pd.DataFrame(rows, columns=['r', 'item', 'obs']); self.L = L; self.n = len(te)
        self.F = hab.feats(te.user_id.values[L.r.values], L.item.values, te.timestamp.values[L.r.values])
        oldv = old[L.r.values]; recv = extra['rec'][L.r.values]; mostv = extra['most'][L.r.values]
        drop = L.obs.values & (L.item.values != oldv)
        prior = self.F[:, 0] > 0
        self.M = {'keep': L.obs.values, 'drop': drop, 'no-global-top': drop & (L.item.values != G),
                  '+recent-reclick team': drop | (L.item.values == recv), '+most-clicked team': drop | (L.item.values == mostv),
                  '+both reclick teams': drop | (L.item.values == recv) | (L.item.values == mostv), 'no other reclickers': drop & ~prior}
        key = pd.MultiIndex.from_arrays([L.r.values, L.item.values])
        self.pos = {k: key.get_indexer(pd.MultiIndex.from_arrays([np.arange(self.n), p])) for k, p in picks.items()}
    def score(self, m):
        idx = m.items.get_indexer(self.L.item.values); b = np.where(idx >= 0, m.b[np.maximum(idx, 0)], m.bdef)
        u = b + self.F @ m.w; r = self.L.r.values
        mx = np.full(self.n, -np.inf); np.maximum.at(mx, r, u); e = np.exp(u - mx[r])
        out = {}
        for k, pos in self.pos.items():
            ep = e[pos]
            for s, M in self.M.items():
                S = np.bincount(r, weights=e * M, minlength=self.n)
                out[(k, s)] = 100 * ep / (S + ep * (~M[pos]))
        return out

def boot_ci(D, B, seed=0):
    """D: n x k matrix of paired diffs -> (mean, lo, hi) arrays via Poisson bootstrap."""
    rng = np.random.default_rng(seed); n, k = D.shape; means = []
    for s in range(0, B, 100):
        W = rng.poisson(1.0, (min(100, B - s), n)).astype(np.float32)
        with np.errstate(invalid='ignore', divide='ignore'): means.append((W @ D.astype(np.float32)) / W.sum(1, keepdims=True))
    mm = np.vstack(means); return D.mean(0), np.nanpercentile(mm, 2.5, 0), np.nanpercentile(mm, 97.5, 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='~/r2data'); ap.add_argument('--cuts', nargs='*', default=['pre', 'day2', 'day3', 'mid', 'day4', 'day5'])
    ap.add_argument('--quick', type=int, default=0, help='subsample N test rows per cut (smoke test)'); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--no-robust', action='store_true'); ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    a = ap.parse_args(); data = os.path.expanduser(a.data); os.makedirs(a.out, exist_ok=True)
    try:
        import torch; torch.set_num_threads(min(8, os.cpu_count() or 1))
    except Exception: pass
    if not a.no_robust: robustness(data)
    log('== FOLDS: loading feedback')
    fb = pd.read_json(os.path.join(data, 'feedback.jsonl'), lines=True, dtype=False).sort_values(['timestamp', 'event_id']).reset_index(drop=True)
    cat1 = pd.read_csv(os.path.join(HERE, 'aux', 'r1_catalog.csv'), dtype={'itemID': str}).set_index('itemID')
    oldp = pd.read_csv(os.path.join(HERE, 'aux', 'r1_old_picks.csv.gz'))
    ev = fb[(fb.timestamp >= W0) & (fb.timestamp < W1)].copy().reset_index(drop=True); ev['rowid'] = ev.event_id
    ev = ev.merge(oldp.rename(columns={'itemID': 'old'}).drop_duplicates(['user_id', 'timestamp']), on=['user_id', 'timestamp'], how='left')
    log(f'feedback {len(fb)} events; R1 pooled events {len(ev)} (old R1 pick matched {ev.old.notna().sum()}, old pick == global top {(ev.old == G).sum()}, global top in slate {sum(G in l for l in ev.displayed_items)})')
    hab = HabitIndex(fb); clk = ev[ev.clicked_item.notna()]
    # items as of t for other-team scenarios (from all clicks before t)
    ug = {}
    for (u, i), arr in hab.g.items(): ug.setdefault(u, []).append((i, arr))
    vf, vt = cat1.valid_from.to_dict(), cat1.valid_to.to_dict(); rec, most = [], []
    for u, t in zip(ev.user_id.values, ev.timestamp.values):
        best_r = best_m = None
        for i, arr in ug.get(u, []):
            k = np.searchsorted(arr, t)
            if k and vf.get(i, 1e18) <= t < vt.get(i, -1):
                if best_r is None or arr[k - 1] > best_r[1]: best_r = (i, arr[k - 1])
                if best_m is None or (k, arr[k - 1]) > best_m[1]: best_m = (i, (k, arr[k - 1]))
        rec.append(best_r[0] if best_r else None); most.append(best_m[0] if best_m else None)
    ev['rec_t'], ev['most_t'] = np.array(rec, dtype=object), np.array(most, dtype=object)
    log('fitting J_all'); J_all = ChoiceModel().fit(clk, hab)
    mid = int(ev.timestamp.quantile(0.5))
    CUTS = {'pre': W0, 'day2': W0 + DAY, 'day3': W0 + 2 * DAY, 'mid': mid, 'day4': W0 + 3 * DAY, 'day5': W0 + 4 * DAY}
    tmp = tempfile.mkdtemp(prefix='v2val_cut_'); summary = []; seg_rows = []
    try:
        for cn in a.cuts:
            c = CUTS[cn]; te = ev[ev.timestamp >= c]
            if a.quick: te = te.sample(min(a.quick, len(te)), random_state=0).sort_values('timestamp')
            te = te.reset_index(drop=True); xp = te[['rowid', 'user_id', 'timestamp']]
            d = os.path.join(tmp, cn); cut = write_cut(fb, te, cat1, c, d)
            log(f'\n===== cut {cn} (ts {c}): train events {len(cut)}, test rows {len(te)}')
            P, RT = {}, {}
            for nm in CANDS:
                try:
                    p, RT[nm] = run(nm, d); P[nm] = p.set_index('rowid').itemID.reindex(xp.rowid).values
                    bad = pd.isna(P[nm]).sum()
                    if bad: log(f'  {nm}: {bad} rows missing'); P[nm] = np.where(pd.isna(P[nm]), P['v2'], P[nm])
                except Exception as e: log(f'  {nm} FAILED: {e!r}')
            v2s = pd.Series(P['v2'], index=xp.rowid.values)
            P['A recent'] = rule_picks(xp, cut, cat1, 'recent', v2s)
            P['B72 decay-count'] = rule_picks(xp, cut, cat1, 'count', v2s, 72)
            P['Lea freq-first'] = rule_picks(xp, cut, cat1, 'lea', lea_fallback(xp, cut, cat1))
            log('  predict runtimes: ' + ', '.join(f'{k} {v:.1f}s' for k, v in RT.items()))
            seg = segments(xp, cut, cat1); segs = [s for s in ['single-live', 'multi-live', 'priors-expired', 'shown-never-clk', 'unseen'] if (seg == s).any()]
            log('  segments: ' + ', '.join(f'{s} {(seg == s).sum()}' for s in segs))
            sc = Scorer(te, P, hab, te.old.values.astype(object), {'rec': te.rec_t.values, 'most': te.most_t.values})
            judges = {}
            if c > W0 + 3600: judges['J_pre'] = ChoiceModel().fit(clk[clk.timestamp < c], hab)
            judges['J_all'] = J_all; judges['J_post'] = ChoiceModel().fit(te[te.clicked_item.notna()], hab)
            win = {k: (p == te.clicked_item.values) for k, p in P.items()}; hasc = te.clicked_item.notna().values
            names = list(P)
            for jn, jm in judges.items():
                R = sc.score(jm); primary = jn == list(judges)[0]
                print(f'\n  judge {jn}{" (primary)" if primary else ""}: cand | keep | drop | keep-v2 [95% CI] | drop-v2 [95% CI] | rows!=v2 | click-win% all / clicked rows')
                D = np.column_stack([R[(k, s)] - R[('v2', s)] for s in ('keep', 'drop') for k in names]); mu, lo, hi = boot_ci(D, a.boot)
                for j, k in enumerate(names):
                    jk, jd = j, j + len(names)
                    print(f'   {k:16s} {R[(k, "keep")].mean():6.2f} {R[(k, "drop")].mean():6.2f} | {mu[jk]:+.2f} [{lo[jk]:+.2f},{hi[jk]:+.2f}] | {mu[jd]:+.2f} [{lo[jd]:+.2f},{hi[jd]:+.2f}] | {(P[k] != P["v2"]).sum():6d} | {100 * win[k].mean():5.2f} / {100 * win[k][hasc].mean():5.2f}', flush=True)
                    summary.append(dict(cut=cn, judge=jn, cand=k, keep=R[(k, 'keep')].mean(), drop=R[(k, 'drop')].mean(), dkeep=mu[jk], dkeep_lo=lo[jk], dkeep_hi=hi[jk],
                                        ddrop=mu[jd], ddrop_lo=lo[jd], ddrop_hi=hi[jd], differ=int((P[k] != P['v2']).sum()), clickwin=100 * win[k].mean()))
                if not primary: continue
                print(f'  segments ({jn}, keep): v2 score, then others as diff vs v2 [95% CI]')
                print('   ' + 'cand'.ljust(16) + ''.join(f'{s + " n=" + str((seg == s).sum()):>24s}' for s in segs))
                for j, k in enumerate(names):
                    line = '   ' + k.ljust(16)
                    for s in segs:
                        m = seg == s
                        if k == 'v2': line += f'{R[("v2", "keep")][m].mean():24.2f}'
                        else:
                            mu2, lo2, hi2 = boot_ci((R[(k, 'keep')] - R[('v2', 'keep')])[m][:, None], min(a.boot, 300)); line += f'{mu2[0]:+.2f} [{lo2[0]:+.2f},{hi2[0]:+.2f}]'.rjust(24)
                            seg_rows.append(dict(cut=cn, seg=s, cand=k, dkeep=mu2[0], lo=lo2[0], hi=hi2[0]))
                    print(line)
                print(f'  slate sensitivity ({jn}): v2 score per scenario, others diff vs v2 [95% CI]  (scenarios start from drop)')
                sc_names = [k for k in names if k != 'v2']
                for s in Scorer.SCEN:
                    D = np.column_stack([R[(k, s)] - R[('v2', s)] for k in sc_names]); mu3, lo3, hi3 = boot_ci(D, min(a.boot, 300))
                    print(f'   {s:22s} v2 {R[("v2", s)].mean():6.2f} | ' + ' | '.join(f'{k} {mu3[j]:+.2f}[{lo3[j]:+.2f},{hi3[j]:+.2f}]' for j, k in enumerate(sc_names)), flush=True)
            shutil.rmtree(d, ignore_errors=True)
    finally: shutil.rmtree(tmp, ignore_errors=True)
    S = pd.DataFrame(summary); S.to_csv(os.path.join(a.out, 'summary.csv'), index=False); pd.DataFrame(seg_rows).to_csv(os.path.join(a.out, 'segments.csv'), index=False)
    print('\n===== VERDICT TABLE: keep diff vs v2 on the primary judge per cut (J_pre, or J_all for the pre-R1 cut); * = CI above 0, - = CI below 0')
    prim = S.groupby('cut').judge.transform(lambda j: 'J_pre' if (j == 'J_pre').any() else 'J_all'); T = S[S.judge == prim]
    piv = T.assign(cell=[f'{d:+.2f}{"*" if l > 0 else ("-" if h < 0 else " ")}' for d, l, h in zip(T.dkeep, T.dkeep_lo, T.dkeep_hi)]).pivot(index='cand', columns='cut', values='cell')
    print(piv[[c for c in a.cuts if c in piv.columns]].to_string())
    v = T[T.cand == 'v2'].set_index('cut')[['keep', 'drop']].T; print('v2 absolute:\n' + v[[c for c in a.cuts if c in v.columns]].round(2).to_string())
    log('done')
if __name__ == '__main__': main()
