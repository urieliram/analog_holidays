"""38-h event profile clustering: eve last 14 h + holiday 24 h.

Run from the M_identify_holidays.ipynb cell via:  %run cluster_38h.py
Requires notebook variables: df_wide, df_holidays_display, match_dates_set,
                              HOUR_COLS, CLUSTER_COLORS, UNIQUE_ID
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from IPython.display import display

PREVIOUSLY_W_HOURS = 14
N_CLUSTERS_PHOL    = 3
_CLUSTER_LABELS    = list('CDEFGHIJ')   # C, D, E, ... agnostic to DOW

# ── Feature columns: last 14 h of eve + full 24 h of holiday = 38 h ──────────
eve_hour_cols = HOUR_COLS[-PREVIOUSLY_W_HOURS:]
hol_hour_cols = HOUR_COLS
feat_cols = (
    [f'eve_{c}' for c in eve_hour_cols]
    + [f'hol_{c}' for c in hol_hour_cols]
)

# ── Holiday name lookup ───────────────────────────────────────────────────────
_date_to_name = dict(zip(
    pd.to_datetime(df_holidays_display['date']).dt.normalize(),
    df_holidays_display['holiday_name'],
))

# ── Build one 38-h vector per confirmed holiday event ─────────────────────────
_wide_idx = set(df_wide.index.normalize())
rows = []
for d in sorted(match_dates_set):
    eve = d - pd.Timedelta(days=1)
    if eve not in _wide_idx:
        continue
    row_eve = df_wide.loc[df_wide.index.normalize() == eve].squeeze()
    row_hol = df_wide.loc[df_wide.index.normalize() == d].squeeze()
    eve_vals = row_eve[eve_hour_cols].values.astype(float)
    hol_vals = row_hol[hol_hour_cols].values.astype(float)
    if np.isnan(eve_vals).any() or np.isnan(hol_vals).any():
        continue
    rows.append({
        'date': d,
        'holiday_name': _date_to_name.get(d, str(d.date())),
        **dict(zip([f'eve_{c}' for c in eve_hour_cols], eve_vals)),
        **dict(zip([f'hol_{c}' for c in hol_hour_cols], hol_vals)),
    })

df_phol = pd.DataFrame(rows).set_index('date')
print(f'Events with complete {PREVIOUSLY_W_HOURS + 24}-h profile: {len(df_phol)}')

# ── KMeans ────────────────────────────────────────────────────────────────────
X        = df_phol[feat_cols].values
X_scaled = StandardScaler().fit_transform(X)
kmeans_phol = KMeans(n_clusters=N_CLUSTERS_PHOL, random_state=42, n_init=20).fit(X_scaled)
df_phol['cluster']      = kmeans_phol.labels_
df_phol['cluster_type'] = [_CLUSTER_LABELS[k] for k in kmeans_phol.labels_]

centroids_phol = np.array([
    df_phol.loc[df_phol['cluster'] == k, feat_cols].values.mean(axis=0)
    for k in range(N_CLUSTERS_PHOL)
])

# ── Per-cluster day list ──────────────────────────────────────────────────────
day_rows = []
for d, row in df_phol.iterrows():
    day_rows.append({
        'type': row['cluster_type'],
        'date': d,
        'dow': d.day_name()[:3],
        'holiday_name': row['holiday_name'],
    })
df_days_phol = (
    pd.DataFrame(day_rows)
    .sort_values(['type', 'date'])
    .reset_index(drop=True)
)

def _day_bg_phol(row):
    k  = _CLUSTER_LABELS.index(row['type'])
    bg = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
    return [f'background-color: {bg}44; color: black'] * len(row)

print('\nDays per cluster')
display(
    df_days_phol.style
    .apply(_day_bg_phol, axis=1)
    .format({'date': lambda d: d.strftime('%Y-%m-%d')})
    .hide(axis='index')
)

# ── 38-h profile plot ─────────────────────────────────────────────────────────
x_axis = list(range(-PREVIOUSLY_W_HOURS, 24))   # -14..-1 = eve, 0..23 = holiday

fig, axes = plt.subplots(
    1, N_CLUSTERS_PHOL,
    figsize=(5.5 * N_CLUSTERS_PHOL, 5),
    sharey=True,
)
if N_CLUSTERS_PHOL == 1:
    axes = [axes]

fig.suptitle(
    f'38-h event profiles  (eve last {PREVIOUSLY_W_HOURS} h + holiday 24 h)  |  {UNIQUE_ID}',
    fontsize=13, y=1.02,
)

for k, ax in enumerate(axes):
    label = _CLUSTER_LABELS[k]
    color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
    mask  = df_phol['cluster'] == k
    data  = df_phol.loc[mask, feat_cols].values
    for profile in data:
        ax.plot(x_axis, profile, color=color, alpha=0.30, linewidth=0.9)
    ax.plot(x_axis, centroids_phol[k], color=color, lw=3,
            label=f'Centroid {label}', zorder=5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.9, label='midnight')
    ax.set_title(f'Type {label}  (n = {mask.sum()})', fontsize=12)
    ax.set_xlabel('Hour relative to holiday start (h = 0)', fontsize=10)
    ax.set_xticks(range(-PREVIOUSLY_W_HOURS, 24, 4))
    ax.tick_params(labelsize=9)
    if k == 0:
        ax.set_ylabel(f'Demand  [{UNIQUE_ID}]', fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
