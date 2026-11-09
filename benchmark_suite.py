import sys
import os
import re
import time
import json
import subprocess

test_configs = [
    ("alexnet_balance_batch", "alexnet", "balance", "batch"),
    ("resnet18_balance_batch", "resnet18", "balance", "batch"),
    ("squeezenet_balance_element", "squeezenet", "balance", "element"),
    ("mobilenetv2_balance_batch", "mobilenetv2", "balance", "batch"),
    ("alexnet_GA_batch", "alexnet", "GA", "batch")
]

def run_test(key, model, rep, pipe):
    cmd = [
        "/usr/bin/time", "-v",
        "./PIMCOMP-NN",
        f"-m={model}",
        f"-r={rep}",
        f"-p={pipe}",
        "-o=NO", "-v=NO", "-i=NO", "-s=NO"
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="/home/zhelang/projects/PIMCOMP-NN-opt/build")
    wall_time = time.time() - t0
    
    stdout = proc.stdout
    stderr = proc.stderr
    exit_code = proc.returncode
    
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    max_rss_mb = float(rss_match.group(1)) / 1024.0 if rss_match else 0.0
    
    time_match = re.search(r"OVERALL TIME\s*={5,}\s*\n([\d\.]+)s", stdout)
    reported_time_s = float(time_match.group(1)) if time_match else 0.0
    
    lat_match = re.search(r"practical \(ns\)\s*\n([\d\.]+)", stdout)
    practical_latency_ns = float(lat_match.group(1)) if lat_match else 0.0
    
    mvmul_match = re.search(r"MVMUL \(#\)\s*\n(\d+)", stdout)
    mvmul_count = int(mvmul_match.group(1)) if mvmul_match else 0
    
    comm_match = re.search(r"COMM \(#\)\s*\n(\d+)", stdout)
    comm_count = int(comm_match.group(1)) if comm_match else 0
    
    mem_match = re.search(r"MEM \(#\)\s*\n(\d+)", stdout)
    mem_count = int(mem_match.group(1)) if mem_match else 0
    
    return {
        "model": model,
        "replicating": rep,
        "pipeline": pipe,
        "wall_time_s": wall_time,
        "reported_time_s": reported_time_s,
        "max_rss_mb": max_rss_mb,
        "practical_latency_ns": practical_latency_ns,
        "mvmul_count": mvmul_count,
        "comm_count": comm_count,
        "mem_count": mem_count,
        "exit_code": exit_code
    }

def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    results = {}
    for key, model, rep, pipe in test_configs:
        print(f"Running {key}...")
        res = run_test(key, model, rep, pipe)
        results[key] = res
        print(f"  Done: wall_time={res['wall_time_s']:.2f}s, reported_time={res['reported_time_s']}s, practical_latency={res['practical_latency_ns']}ns, max_rss={res['max_rss_mb']:.1f}MB, exit={res['exit_code']}")
        
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
