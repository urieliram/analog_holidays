"""One-time patch: replace the 3-line stub in cell 8c-bis with the full analysis.

Instructions:
  1. In VS Code, close M_identify_holidays.ipynb  (File > Close Editor or Ctrl+W)
  2. From a terminal in c:\GIT\analog_holidays run:   py -3 patch_notebook.py
  3. Reopen the notebook in VS Code.
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("M_identify_holidays.ipynb")
CELL_ID = "8148126c"

NEW_SOURCE = [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from IPython.display import display\n",
    "\n",
    "PREVIOUSLY_W_HOURS = 14\n",
    "N_CLUSTERS_PHOL    = 3\n",
    "_CLUSTER_LABELS    = list('CDEFGHIJ')  # C, D, E ... agnostic to DOW\n",
    "\n",
    "# Feature columns: last 14 h of eve + full 24 h of holiday = 38 h\n",
    "eve_hour_cols = HOUR_COLS[-PREVIOUSLY_W_HOURS:]\n",
    "hol_hour_cols = HOUR_COLS\n",
    "feat_cols = (\n",
    "    [f'eve_{c}' for c in eve_hour_cols]\n",
    "    + [f'hol_{c}' for c in hol_hour_cols]\n",
    ")\n",
    "\n",
    "# Holiday name lookup\n",
    "_date_to_name = dict(zip(\n",
    "    pd.to_datetime(df_holidays_display['date']).dt.normalize(),\n",
    "    df_holidays_display['holiday_name'],\n",
    "))\n",
    "\n",
    "# Build one 38-h vector per confirmed holiday event\n",
    "_wide_idx = set(df_wide.index.normalize())\n",
    "rows = []\n",
    "for d in sorted(match_dates_set):\n",
    "    eve = d - pd.Timedelta(days=1)\n",
    "    if eve not in _wide_idx:\n",
    "        continue\n",
    "    row_eve = df_wide.loc[df_wide.index.normalize() == eve].squeeze()\n",
    "    row_hol = df_wide.loc[df_wide.index.normalize() == d].squeeze()\n",
    "    eve_vals = row_eve[eve_hour_cols].values.astype(float)\n",
    "    hol_vals = row_hol[hol_hour_cols].values.astype(float)\n",
    "    if np.isnan(eve_vals).any() or np.isnan(hol_vals).any():\n",
    "        continue\n",
    "    rows.append({\n",
    "        'date': d,\n",
    "        'holiday_name': _date_to_name.get(d, str(d.date())),\n",
    "        **dict(zip([f'eve_{c}' for c in eve_hour_cols], eve_vals)),\n",
    "        **dict(zip([f'hol_{c}' for c in hol_hour_cols], hol_vals)),\n",
    "    })\n",
    "\n",
    "df_phol = pd.DataFrame(rows).set_index('date')\n",
    "print(f'Events with complete {PREVIOUSLY_W_HOURS + 24}-h profile: {len(df_phol)}')\n",
    "\n",
    "# KMeans\n",
    "X        = df_phol[feat_cols].values\n",
    "X_scaled = StandardScaler().fit_transform(X)\n",
    "kmeans_phol = KMeans(n_clusters=N_CLUSTERS_PHOL, random_state=42, n_init=20).fit(X_scaled)\n",
    "df_phol['cluster']      = kmeans_phol.labels_\n",
    "df_phol['cluster_type'] = [_CLUSTER_LABELS[k] for k in kmeans_phol.labels_]\n",
    "\n",
    "centroids_phol = np.array([\n",
    "    df_phol.loc[df_phol['cluster'] == k, feat_cols].values.mean(axis=0)\n",
    "    for k in range(N_CLUSTERS_PHOL)\n",
    "])\n",
    "\n",
    "# Per-cluster day list\n",
    "day_rows = []\n",
    "for d, row in df_phol.iterrows():\n",
    "    day_rows.append({'type': row['cluster_type'], 'date': d,\n",
    "                     'dow': d.day_name()[:3], 'holiday_name': row['holiday_name']})\n",
    "df_days_phol = pd.DataFrame(day_rows).sort_values(['type', 'date']).reset_index(drop=True)\n",
    "\n",
    "def _day_bg_phol(row):\n",
    "    k  = _CLUSTER_LABELS.index(row['type'])\n",
    "    return [f'background-color: {CLUSTER_COLORS[k % len(CLUSTER_COLORS)]}44; color: black'] * len(row)\n",
    "\n",
    "print('\\nDays per cluster')\n",
    "display(df_days_phol.style\n",
    "        .apply(_day_bg_phol, axis=1)\n",
    "        .format({'date': lambda d: d.strftime('%Y-%m-%d')})\n",
    "        .hide(axis='index'))\n",
    "\n",
    "# 38-h profile plot\n",
    "x_axis = list(range(-PREVIOUSLY_W_HOURS, 24))  # -14..-1=eve, 0..23=holiday\n",
    "fig, axes = plt.subplots(1, N_CLUSTERS_PHOL, figsize=(5.5*N_CLUSTERS_PHOL, 5), sharey=True)\n",
    "if N_CLUSTERS_PHOL == 1:\n",
    "    axes = [axes]\n",
    "fig.suptitle(\n",
    "    f'38-h event profiles  (eve last {PREVIOUSLY_W_HOURS} h + holiday 24 h)  |  {UNIQUE_ID}',\n",
    "    fontsize=13, y=1.02)\n",
    "for k, ax in enumerate(axes):\n",
    "    label = _CLUSTER_LABELS[k]\n",
    "    color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]\n",
    "    mask  = df_phol['cluster'] == k\n",
    "    data  = df_phol.loc[mask, feat_cols].values\n",
    "    for profile in data:\n",
    "        ax.plot(x_axis, profile, color=color, alpha=0.30, lw=0.9)\n",
    "    ax.plot(x_axis, centroids_phol[k], color=color, lw=3,\n",
    "            label=f'Centroid {label}', zorder=5)\n",
    "    ax.axvline(0, color='grey', ls='--', lw=0.9, label='midnight')\n",
    "    ax.set_title(f'Type {label}  (n={mask.sum()})', fontsize=12)\n",
    "    ax.set_xlabel('Hour relative to holiday start (h=0)', fontsize=10)\n",
    "    ax.set_xticks(range(-PREVIOUSLY_W_HOURS, 24, 4))\n",
    "    if k == 0:\n",
    "        ax.set_ylabel(f'Demand [{UNIQUE_ID}]', fontsize=10)\n",
    "    ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patched = False
for cell in nb["cells"]:
    if cell.get("id", "") == CELL_ID:
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    raise RuntimeError(f"Cell {CELL_ID!r} not found in notebook. IDs present: "
                       + str([c.get('id') for c in nb['cells']]))

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched cell {CELL_ID!r} in {NB_PATH.name}  — reopen the notebook in VS Code.")
