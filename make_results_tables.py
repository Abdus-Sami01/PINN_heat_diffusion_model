import os
import json


def read_kv(path):
    d = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    d[parts[0]] = float(parts[1])
    return d


def main():
    out = []

    fwd = read_kv("results/forward_metrics.txt")
    if fwd:
        out.append("### Forward PINN vs FDM (no data, physics only)")
        out.append("")
        out.append("| metric | value |")
        out.append("|---|---|")
        out.append("| relative L2 error | " +
                    str(round(fwd["relative_l2"] * 100, 3)) + "% |")
        out.append("| final PDE residual | " +
                    "{:.2e}".format(fwd["final_pde_loss"]) + " |")
        out.append("")

    inv = read_kv("results/inverse_metrics.txt")
    if inv:
        out.append("### Inverse PINN headline (3 sensors, 1 C noise, h0 = 0.15)")
        out.append("")
        out.append("| | h |")
        out.append("|---|---|")
        out.append("| true | " + str(inv["h_true"]) + " |")
        out.append("| recovered | " + str(round(inv["h_recovered"], 5)) + " |")
        out.append("| error | " + str(round(inv["h_rel_error_pct"], 2)) + "% |")
        out.append("")

    if os.path.exists("results/sensitivity.json"):
        with open("results/sensitivity.json") as f:
            sens = json.load(f)

        ls = {}
        if os.path.exists("results/least_squares_h.txt"):
            with open("results/least_squares_h.txt") as f:
                for line in f.readlines()[1:]:
                    p = line.split()
                    ls[(int(p[0]), float(p[1]))] = float(p[3])

        out.append("### h recovery error (%) vs sensors and noise")
        out.append("")
        if ls:
            out.append("| sensors | noise (C) | inverse PINN | FDM least-squares |")
            out.append("|---|---|---|---|")
        else:
            out.append("| sensors | noise (C) | inverse PINN |")
            out.append("|---|---|---|")
        for r in sens:
            row = "| " + str(r["sensors"]) + " | " + str(r["noise"]) + \
                  " | " + str(round(r["h_err_pct"], 2)) + "%"
            key = (r["sensors"], r["noise"])
            if ls and key in ls:
                row = row + " | " + str(round(ls[key], 2)) + "%"
            row = row + " |"
            out.append(row)
        out.append("")

    fullgrid = read_kv("results/forward_fullgrid.txt")
    dnn = read_kv("results/baseline_data_only.txt")
    lstm = read_kv("results/baseline_lstm.txt")
    if fullgrid and dnn and lstm:
        out.append("### Full-field reconstruction from the same information")
        out.append("")
        out.append("| model | uses physics | full grid rel L2 | max abs error (C) |")
        out.append("|---|---|---|---|")
        out.append("| forward PINN | yes | " +
                    str(round(fullgrid["relative_l2"] * 100, 2)) + "% | " +
                    str(round(fullgrid["max_abs_error"], 2)) + " |")
        out.append("| data-only NN | no | " +
                    str(round(dnn["relative_l2"] * 100, 2)) + "% | " +
                    str(round(dnn["max_abs_error"], 2)) + " |")
        out.append("| LSTM | no | " +
                    str(round(lstm["relative_l2"] * 100, 2)) + "% | " +
                    str(round(lstm["max_abs_error"], 2)) + " |")
        out.append("")

    text = "\n".join(out)
    with open("results/RESULTS_TABLES.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
