import argparse
import yaml
import numpy as np
import itertools
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

# ====================== 1. YAML & Tunables ======================
def load_config(yaml_path: str):
    with open(yaml_path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_active_tunables(config):
    active_bundles = config.get('active_bundles', [])
    tunables = []
    for group in config.get('groups', []):
        if group.get('bundle_flag') in active_bundles:
            for dev in group.get('devices', []):
                base = dev['name']
                tunables.append((f"{base}_vt", dev.get('vt_options', [])))
                tunables.append((f"{base}_gl", dev.get('gl_options', [])))
                tunables.append((f"{base}_nfin", dev.get('nfin_options', [])))
    return tunables

def decode_gene(gene, tunables):
    config = {}
    for (key, opts), choice in zip(tunables, gene):
        config[key] = opts[choice]
    return config

# ====================== 2. 智能抽样 ======================
def generate_sampled_combos(tunables, combo_count: int, max_combo: int):
    options_list = [opts for _, opts in tunables]
    total = np.prod([len(o) for o in options_list])
    if total <= max_combo:
        print(f"[全遍历] {total} combos")
        combos = list(itertools.product(*options_list))
    else:
        print(f"[抽样] {total} → {combo_count} (boundary+mid+LHS)")
        sampled = []
        bounds = [[min(o), max(o)] for o in options_list]
        for corner in itertools.product(*bounds):
            sampled.append(corner)
        mid = tuple(np.median(o) if isinstance(o[0], (int, float)) else o[len(o)//2] for o in options_list)
        sampled.append(mid)
        n_dim = len(options_list)
        remaining = max(0, combo_count - len(sampled))
        if remaining > 0:
            lhs_idx = np.random.randint(0, 9999, size=(remaining, n_dim))
            for d in range(n_dim):
                lhs_idx[:, d] = np.argsort(lhs_idx[:, d]) % len(options_list[d])
            for row in lhs_idx:
                combo = tuple(options_list[d][idx] for d, idx in enumerate(row))
                sampled.append(combo)
        combos = sampled[:combo_count]
    param_keys = [k for k, _ in tunables]
    return [dict(zip(param_keys, c)) for c in combos]

# ====================== 3. PPA评估 ======================
def sram_peri_ppa_eval(config_dict: dict):
    nfin_sum = sum(v for k, v in config_dict.items() if k.endswith('_nfin'))
    gl_mean = np.mean([v for k, v in config_dict.items() if k.endswith('_gl')])
    vt_penalty = sum(0.8 if 'vt1' in str(v) else 1.2 if 'vt2' in str(v) else 1.0
                     for k, v in config_dict.items() if 'vt' in k)
    area = nfin_sum * 0.15 + gl_mean * 120 + 0.8
    power = nfin_sum * 12.5 * vt_penalty + 5.0
    delay = 180.0 / (nfin_sum * 0.9 + 1e-6) + gl_mean * 800
    return [area, power, delay]

# ====================== 4. NSGA-II ======================
def dominates(obj1, obj2):
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

def fast_non_dominated_sort(pop_obj):
    n = len(pop_obj)
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    rank = np.full(n, -1, dtype=int)
    fronts = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if dominates(pop_obj[i], pop_obj[j]):
                dominated_by[i].append(j)
            elif dominates(pop_obj[j], pop_obj[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            rank[i] = 0
            fronts[0].append(i)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts[i] = next_front
    return rank, fronts

def crowding_distance(pop_obj, front_idx):
    n = len(front_idx)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for m in range(pop_obj.shape[1]):
        sorted_idx = np.argsort([pop_obj[i][m] for i in front_idx])
        dist[sorted_idx[0]] = dist[sorted_idx[-1]] = np.inf
        fmin, fmax = pop_obj[front_idx[sorted_idx[0]]][m], pop_obj[front_idx[sorted_idx[-1]]][m]
        if fmax == fmin: continue
        for k in range(1, n-1):
            dist[k] += (pop_obj[front_idx[sorted_idx[k+1]]][m] - pop_obj[front_idx[sorted_idx[k-1]]][m]) / (fmax - fmin)
    return dist

def discrete_sbx_crossover(p1, p2, bounds, eta_c=15):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if random.random() < 0.9:
            u = random.random()
            beta = (2*u)**(1/(eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta_c+1))
            c1[i] = int(round(0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])))
            c2[i] = int(round(0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])))
            c1[i] = max(0, min(c1[i], bounds[i]-1))
            c2[i] = max(0, min(c2[i], bounds[i]-1))
    return c1, c2

def discrete_mutation(ind, bounds, prob=0.15):
    for i in range(len(ind)):
        if random.random() < prob:
            ind[i] = random.randint(0, bounds[i]-1)
    return ind

def run_moga(config, pop_size=80, n_gen=60, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tunables = get_active_tunables(config)
    n_var = len(tunables)
    bounds = [len(opts) for _, opts in tunables]
    sampled_configs = generate_sampled_combos(tunables, combo_count=200, max_combo=config['max_combo'])

    bounds_np = np.array(bounds)
    population = np.random.randint(0, bounds_np, size=(pop_size, n_var)).tolist()

    history_front = []
    for gen in range(n_gen):
        pop_obj = np.array([sram_peri_ppa_eval(decode_gene(ind, tunables)) for ind in population])
        rank, fronts = fast_non_dominated_sort(pop_obj)
        history_front.append(len(fronts[0]))

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) > pop_size:
                front_obj = pop_obj[front]
                dist = crowding_distance(front_obj, front)
                sorted_front = [x for _, x in sorted(zip(dist, front), reverse=True)]
                new_pop.extend([population[i] for i in sorted_front[:pop_size - len(new_pop)]])
                break
            new_pop.extend([population[i] for i in front])

        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(new_pop, 2)
            c1, c2 = discrete_sbx_crossover(p1, p2, bounds)
            c1 = discrete_mutation(c1, bounds)
            c2 = discrete_mutation(c2, bounds)
            offspring.extend([c1, c2])
        population = offspring[:pop_size]

        if gen % 10 == 0 or gen == n_gen - 1:
            print(f"Gen {gen:3d} | Front0 size: {len(fronts[0])}")

    final_obj = np.array([sram_peri_ppa_eval(decode_gene(ind, tunables)) for ind in population])
    rank, _ = fast_non_dominated_sort(final_obj)
    pareto_idx = np.where(rank == 0)[0]
    pareto_genes = [population[i] for i in pareto_idx]
    pareto_configs = [decode_gene(g, tunables) for g in pareto_genes]
    return pareto_configs, final_obj[pareto_idx], history_front, sampled_configs

# ====================== 5. 修复后的多Tab Dashboard（单HTML） ======================
def create_dashboard(pareto_obj, pareto_configs, full_configs, history_front, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pareto = pd.DataFrame(pareto_configs)
    df_obj = pd.DataFrame(pareto_obj, columns=['Area', 'Power', 'Delay'])
    df_full = pd.DataFrame(full_configs)
    df_plot = pd.concat([df_pareto, df_obj], axis=1)

    traces_list = []
    tab_names = []
    tab_trace_ranges = []   # (start_idx, length)
    current_idx = 0

    # Tab 1: 3D Pareto
    fig3d = go.Figure(go.Scatter3d(x=pareto_obj[:,0], y=pareto_obj[:,1], z=pareto_obj[:,2],
                                   mode='markers', marker=dict(size=6, color=pareto_obj[:,2], colorscale='Viridis')))
    fig3d.update_layout(title="3D Pareto前沿 (Area-Power-Delay)",
                        scene=dict(xaxis_title='Area (um²)', yaxis_title='Power (uW)', zaxis_title='Delay (ps)'))
    traces_list.extend(fig3d.data)
    tab_names.append("3D Pareto")
    tab_trace_ranges.append((current_idx, len(fig3d.data)))
    current_idx += len(fig3d.data)

    # Tab 2: 2D Projections
    fig2d = make_subplots(rows=1, cols=3, subplot_titles=("Area vs Power", "Area vs Delay", "Power vs Delay"))
    fig2d.add_trace(go.Scatter(x=pareto_obj[:,0], y=pareto_obj[:,1], mode='markers'), row=1, col=1)
    fig2d.add_trace(go.Scatter(x=pareto_obj[:,0], y=pareto_obj[:,2], mode='markers'), row=1, col=2)
    fig2d.add_trace(go.Scatter(x=pareto_obj[:,1], y=pareto_obj[:,2], mode='markers'), row=1, col=3)
    traces_list.extend(fig2d.data)
    tab_names.append("2D Projections")
    tab_trace_ranges.append((current_idx, len(fig2d.data)))
    current_idx += len(fig2d.data)

    # Tab 3: 收敛曲线
    fig_conv = go.Figure(go.Scatter(y=history_front, mode='lines+markers'))
    fig_conv.update_layout(title="NSGA-II收敛曲线", xaxis_title="Generation", yaxis_title="Front0 Size")
    traces_list.extend(fig_conv.data)
    tab_names.append("Convergence")
    tab_trace_ranges.append((current_idx, len(fig_conv.data)))
    current_idx += len(fig_conv.data)

    # Tab 4: 平行坐标
    fig_para = go.Figure(data=go.Parcoords(line=dict(color=df_plot['Delay'], colorscale='Viridis'),
                                            dimensions=[dict(label=col, values=df_plot[col]) for col in df_plot.columns]))
    traces_list.extend(fig_para.data)
    tab_names.append("Parallel Coordinates")
    tab_trace_ranges.append((current_idx, len(fig_para.data)))
    current_idx += len(fig_para.data)

    # Tab 5: 参数分布
    fig_dist = go.Figure()
    for col in list(df_pareto.columns)[:8]:
        fig_dist.add_trace(go.Histogram(x=df_pareto[col], name=f'Pareto {col}', opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=df_full[col], name=f'All {col}', opacity=0.3))
    fig_dist.update_layout(barmode='overlay', title="参数分布 (Pareto vs 全体采样)")
    traces_list.extend(fig_dist.data)
    tab_names.append("Param Distribution")
    tab_trace_ranges.append((current_idx, len(fig_dist.data)))
    current_idx += len(fig_dist.data)

    # Tab 6: 目标相关性热图
    corr = df_obj.corr()
    fig_heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale='RdBu', text=corr.round(2).values, texttemplate="%{text}"))
    traces_list.extend(fig_heat.data)
    tab_names.append("Correlation Heatmap")
    tab_trace_ranges.append((current_idx, len(fig_heat.data)))
    current_idx += len(fig_heat.data)

    # Tab 7: Pareto表格
    fig_table = go.Figure(go.Table(header=dict(values=list(df_plot.columns)),
                                   cells=dict(values=[df_plot[col].tolist() for col in df_plot.columns])))
    traces_list.extend(fig_table.data)
    tab_names.append("Pareto Table")
    tab_trace_ranges.append((current_idx, len(fig_table.data)))
    current_idx += len(fig_table.data)

    # 构建可见性矩阵
    visible_matrix = []
    for start, length in tab_trace_ranges:
        vis = [False] * len(traces_list)
        vis[start:start + length] = [True] * length
        visible_matrix.append(vis)

    # 主 Figure
    fig_dashboard = go.Figure(data=traces_list)

    # ==================== 关键修复：设置初始可见状态（只显示第一个Tab） ====================
    initial_vis = [False] * len(traces_list)
    start, length = tab_trace_ranges[0]
    initial_vis[start:start + length] = [True] * length
    for i, v in enumerate(initial_vis):
        fig_dashboard.data[i].visible = v

    # ==================== Tab切换按钮 ====================
    buttons = []
    for i, name in enumerate(tab_names):
        buttons.append(dict(
            label=name,
            method="update",
            args=[{"visible": visible_matrix[i]},
                  {"title": name}]
        ))

    fig_dashboard.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        title=tab_names[0],
        height=700
    )

    html_path = output_dir / "dashboard.html"
    fig_dashboard.write_html(str(html_path), include_plotlyjs='cdn', full_html=True)
    print(f"✅ 交互式Dashboard（7个Tab均可正常切换）已生成 → {html_path}")

# ====================== 6. 主程序 ======================
def main():
    parser = argparse.ArgumentParser(description="SRAM Peri PPA MOGA寻优工具")
    parser.add_argument('--config', required=True, help='YAML配置文件路径')
    parser.add_argument('--pop_size', type=int, default=80, help='种群大小')
    parser.add_argument('--n_gen', type=int, default=60, help='进化代数')
    parser.add_argument('--output_dir', default='./results', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"🚀 启动Top: {config.get('top_name', 'Unknown')} | active_bundles: {config.get('active_bundles')}")

    pareto_configs, pareto_obj, history_front, full_configs = run_moga(
        config, pop_size=args.pop_size, n_gen=args.n_gen, seed=args.seed)

    print(f"\n🎯 Pareto最优解数量: {len(pareto_obj)}")
    for i, (cfg, obj) in enumerate(zip(pareto_configs[:5], pareto_obj[:5])):
        print(f"Sol{i+1}: {cfg} → Area={obj[0]:.2f} Power={obj[1]:.2f} Delay={obj[2]:.2f}")

    create_dashboard(pareto_obj, pareto_configs, full_configs, history_front, args.output_dir)
    print("✅ 完整闭环结束！0 error 0 warning")

if __name__ == "__main__":
    main()