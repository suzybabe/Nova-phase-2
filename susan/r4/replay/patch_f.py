import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
a1 = '''    h = fb[fb.clicked == 1].groupby(["user_id", "itemID"]).agg(n=("clicked", "size"), last=("timestamp", "max")).reset_index()'''
a2 = '''    q = q[(q.itemID.map(vfm) <= q.timestamp) & (q.timestamp < q.itemID.map(vtm))]'''
a3 = '''(q.timestamp - q["last"]).to_numpy() / 3600) @ W)'''
for a in (a1, a2, a3): assert s.count(a) == 1, a
s = s.replace(a1, a1 + '''
    sn = fb[fb.clicked == 0].groupby(["user_id", "itemID"]).size().rename("seen").reset_index()''')
s = s.replace(a2, a2 + '''
    q = q.merge(sn, on=["user_id", "itemID"], how="left"); q["seen"] = q.seen.fillna(0)''')
s = s.replace(a3, '''(q.timestamp - q["last"]).to_numpy() / 3600) @ W - FATIGUE * np.log1p(q.seen.to_numpy()))''')
s = s.replace('POOLED_MAX_SLATE = 2.8', 'POOLED_MAX_SLATE = 2.8\nFATIGUE = 0.1')
s = s.replace('[nova-r4-v5] done', '[nova-r4-v5f] done')
open(dst, "w").write(s); print("written", dst)
