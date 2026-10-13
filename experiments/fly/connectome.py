"""What does the fly's wiring say about visual-motor organisation?

Data: FlyWire whole-brain connectome v783 (Dorkenwald et al. 2024, Schlegel et
al. 2024; https://zenodo.org/records/10676866) and the FlyWire annotations
(https://github.com/flyconnectome/flywire_annotations).

    python experiments/fly/connectome.py ~/datasets/flywire
"""

import collections
import os
import sys

import numpy as np
import pandas as pd

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/flywire")
ann = pd.read_csv(os.path.join(root, "Supplemental_file1_neuron_annotations.tsv"), sep="\t", low_memory=False)
con = pd.read_feather(os.path.join(root, "proofread_connections_783.feather"))
print("connection rows:", len(con), "| columns:", list(con.columns))
pre_col = [c for c in con.columns if c.startswith("pre")][0]
post_col = [c for c in con.columns if c.startswith("post")][0]
syn_col = [c for c in con.columns if "syn" in c and "count" in c][0]
# collapse neuropil rows into neuron-to-neuron edges
edges = con.groupby([pre_col, post_col], as_index=False)[syn_col].sum()
edges.columns = ["pre", "post", "syn"]
strong = edges[edges.syn >= 5]
print(f"neuron pairs: {len(edges):,}  (with >=5 synapses: {len(strong):,})  synapses: {edges.syn.sum():,}")

sc = dict(zip(ann.root_id, ann.super_class))
cc = dict(zip(ann.root_id, ann.cell_class))
ct = dict(zip(ann.root_id, ann.cell_type))
edges["pre_sc"] = edges.pre.map(sc)
edges["post_sc"] = edges.post.map(sc)


def report(title, mask):
    s = edges[mask].syn.sum()
    print(f"  {title:<55} {s:>12,} synapses")
    return s


print("\n1. Feedforward vs feedback between the eye (optic lobe) and the central brain")
optic = {"optic", "visual_projection", "visual_centrifugal"}
ff = report("visual projection neurons -> central brain neurons", (edges.pre_sc == "visual_projection") & ~edges.post_sc.isin(optic | {"sensory"}))
fb = report("visual centrifugal neurons -> optic lobe neurons", (edges.pre_sc == "visual_centrifugal") & edges.post_sc.isin(optic))
print(f"  feedback / feedforward = {fb / ff:.2f}")
dn_in = edges[edges.post_sc == "descending"]
tot = dn_in.syn.sum()
print("\n2. Where descending (motor command) neurons get their input")
for src, s in dn_in.groupby("pre_sc").syn.sum().sort_values(ascending=False).head(10).items():
    print(f"  from {src:<22} {s / tot:6.1%}")

# hop distance from photoreceptors to descending neurons (strong edges)
print("\n   hops from photoreceptors to descending neurons (edges >= 5 synapses):")
adj = collections.defaultdict(list)
for a, b in zip(strong.pre.values, strong.post.values):
    adj[a].append(b)
photo = set(ann.root_id[(ann.super_class == "sensory") & (ann.cell_class == "visual")])
dns = set(ann.root_id[ann.super_class == "descending"])
dist, frontier, hop = {p: 0 for p in photo}, list(photo), 0
while frontier and hop < 10:
    hop += 1
    nxt = []
    for u in frontier:
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = hop
                nxt.append(v)
    frontier = nxt
dh = collections.Counter(dist[d] for d in dns if d in dist)
print("  " + "  ".join(f"{h} hops: {n}" for h, n in sorted(dh.items())) + f"  (unreached: {len(dns) - sum(dh.values())})")

print("\n3. Efference copy: what drives the neurons that feed back into the eye?")
cf_in = edges[edges.post_sc == "visual_centrifugal"]
t = cf_in.syn.sum()
for src, s in cf_in.groupby("pre_sc").syn.sum().sort_values(ascending=False).head(8).items():
    print(f"  from {src:<22} {s / t:6.1%}")
dn_out = edges[(edges.pre_sc == "descending")]
print(f"  descending neurons -> visual centrifugal: {edges[(edges.pre_sc == 'descending') & (edges.post_sc == 'visual_centrifugal')].syn.sum():,} synapses; -> optic lobe: {edges[(edges.pre_sc == 'descending') & (edges.post_sc == 'optic')].syn.sum():,}")
# neurons that feed both descending neurons and centrifugal neurons (shared drive = potential corollary discharge)
feeds_dn = set(strong.pre[strong.post.isin(dns)])
cfs = set(ann.root_id[ann.super_class == "visual_centrifugal"])
feeds_cf = set(strong.pre[strong.post.isin(cfs)])
both = feeds_dn & feeds_cf
print(f"  neurons that strongly drive both descending and centrifugal neurons: {len(both)} "
      f"({len(both) / max(len(feeds_cf), 1):.0%} of those driving centrifugal neurons)")

print("\n4. Convergence: how many visual feature channels (VPN types) reach each descending neuron directly")
vpn_to_dn = strong[(strong.pre.map(sc) == "visual_projection") & strong.post.isin(dns)].copy()
vpn_to_dn["vtype"] = vpn_to_dn.pre.map(ct)
per_dn = vpn_to_dn.groupby("post").vtype.nunique()
print(f"  descending neurons with direct VPN input: {len(per_dn)} of {len(dns)} | VPN types per such DN: median {per_dn.median():.0f}, max {per_dn.max()}")
top = vpn_to_dn.groupby("vtype").syn.sum().sort_values(ascending=False).head(10)
print("  VPN types sending most synapses directly onto DNs:", ", ".join(f"{k} ({v:,})" for k, v in top.items()))

print("\n5. Location -> action: LC4 looming detectors onto the giant fibre (DNp01)")
gf = set(ann.root_id[ann.cell_type == "DNp01"])
lc4 = ann[ann.cell_type == "LC4"][["root_id", "pos_x", "pos_y", "pos_z", "side"]]
w = edges[edges.pre.isin(set(lc4.root_id)) & edges.post.isin(gf)].groupby(["pre", "post"]).syn.sum().reset_index()
m = w.merge(lc4, left_on="pre", right_on="root_id")
for side, g in m.groupby("side"):
    if len(g) > 5:
        for axis in ("pos_x", "pos_y", "pos_z"):
            r = np.corrcoef(g[axis], g.syn)[0, 1]
            print(f"  {side:<6} LC4 -> ipsilateral/contra GF, n={len(g):>3}: correlation of synapse count with {axis}: {r:+.2f}")
