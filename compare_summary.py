import json

with open("baseline_results.json") as f:
    base = json.load(f)
with open("round6_results.json") as f:
    r6 = json.load(f)

print(f"{'Model / Pipeline':<27} | {'Base Time':<10} | {'Opt Time':<10} | {'Speedup':<10} | {'Base RSS':<10} | {'Opt RSS':<10} | {'RAM Saved':<10}")
print("-" * 100)
for k in base:
    if k not in r6:
        continue
    b_t = base[k]["wall_time_s"]
    o_t = r6[k]["wall_time_s"]
    sp = b_t / o_t if o_t > 0 else 0
    b_m = base[k]["max_rss_mb"]
    o_m = r6[k]["max_rss_mb"]
    m_drop = (b_m - o_m) / b_m * 100
    print(f"{k:<27} | {b_t:<10.2f}s | {o_t:<10.2f}s | {sp:<9.2f}x | {b_m:<8.1f}MB | {o_m:<8.1f}MB | -{m_drop:<8.1f}%")
