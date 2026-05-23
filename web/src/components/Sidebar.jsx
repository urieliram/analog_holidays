import React from 'react'
import { BRAND, UI } from '../constants.js'

const s = {
  sidebar: {
    width: 220,
    minHeight: '100vh',
    background: UI.header_bg,
    display: 'flex',
    flexDirection: 'column',
    padding: '16px 12px',
    flexShrink: 0,
    overflowY: 'auto',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  brandText: {
    fontSize: '1.05rem',
    lineHeight: 1.3,
  },
  divider: {
    border: 'none',
    borderTop: `1px solid rgba(255,255,255,0.15)`,
    margin: '10px 0',
  },
  label: {
    color: '#aad', fontSize: 11, marginBottom: 3, fontWeight: 'bold',
  },
  select: {
    width: '100%',
    padding: '5px 8px',
    borderRadius: 6,
    border: 'none',
    background: 'rgba(255,255,255,0.12)',
    color: '#fff',
    fontSize: 13,
    marginBottom: 10,
    cursor: 'pointer',
  },
  countRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12,
    color: '#ccc',
    marginBottom: 2,
  },
  btn: {
    width: '100%',
    padding: '7px 0',
    borderRadius: 6,
    border: 'none',
    cursor: 'pointer',
    fontWeight: 'bold',
    fontSize: 13,
    marginBottom: 6,
  },
  btnPrimary: {
    background: UI.accent,
    color: '#014754',
  },
  btnSecondary: {
    background: 'rgba(255,255,255,0.12)',
    color: '#fff',
  },
  sectionLabel: {
    color: UI.accent,
    fontSize: 11,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 6,
    marginTop: 10,
  },
  unsaved: {
    background: '#ff9800',
    color: '#fff',
    borderRadius: 6,
    padding: '4px 8px',
    fontSize: 11,
    fontWeight: 'bold',
    marginBottom: 6,
    textAlign: 'center',
  },
}

const LABEL_COLOR_MAP = {
  holiday:     '#2ca02c',
  special_day: '#d62728',
  normal_day:  '#1f77b4',
}
const LABEL_SYMBOL = { holiday: '🟢', special_day: '🔴', normal_day: '🔵' }

export default function Sidebar({
  regions, region, onRegionChange,
  rows,
  calMonth, onMonthChange,
  selectedDate, onDateClick,
  labelCounts,
  unsaved,
  onSave,
  onExportCSV,
  onExportNixtla,
  CalendarComp,
}) {
  return (
    <aside style={s.sidebar}>
      {/* Branding */}
      <div style={s.logo}>
        <div style={{ fontSize: 28 }}>📅</div>
        <div>
          <div style={{ ...s.brandText }}>
            <b>
              <span style={{ color: BRAND.white }}>Forecast</span>
              <span style={{ color: BRAND.electric_green }}>Energ</span>
              <span style={{ color: BRAND.deep_purple }}>AI</span>
            </b>
          </div>
          <div style={{ fontSize: 10, color: '#9274ff', marginTop: 1 }}>
            Holiday Feature Audit
          </div>
        </div>
      </div>
      <hr style={s.divider} />

      {/* Region selector */}
      <div style={s.label}>Region</div>
      <select
        value={region}
        onChange={e => onRegionChange(e.target.value)}
        style={s.select}
      >
        {regions.map(r => (
          <option key={r} value={r} style={{ background: UI.header_bg }}>
            {r.replace('SEN_demand_', '')}
          </option>
        ))}
      </select>

      {/* Calendar */}
      <CalendarComp />

      <hr style={s.divider} />

      {/* Label counts */}
      <div style={s.sectionLabel}>Counts</div>
      {Object.entries(labelCounts).map(([lbl, cnt]) => (
        <div key={lbl} style={s.countRow}>
          <span>
            {LABEL_SYMBOL[lbl]}&nbsp;
            <span style={{ color: LABEL_COLOR_MAP[lbl] }}>
              {lbl.replace('_', ' ')}
            </span>
          </span>
          <span style={{ fontWeight: 'bold', color: '#fff' }}>{cnt}</span>
        </div>
      ))}

      <hr style={s.divider} />

      {/* Actions */}
      {unsaved && <div style={s.unsaved}>⚠ Unsaved changes</div>}

      <button
        style={{ ...s.btn, ...s.btnPrimary }}
        onClick={onSave}
      >
        💾 {unsaved ? 'Save ●' : 'Save'}
      </button>
      <button style={{ ...s.btn, ...s.btnSecondary }} onClick={onExportCSV}>
        📤 Export CSV
      </button>
      <button style={{ ...s.btn, ...s.btnSecondary }} onClick={onExportNixtla}>
        📤 Export Nixtla
      </button>

      <hr style={s.divider} />

      {/* Footer */}
      <div style={{ marginTop: 'auto', fontSize: 10, color: 'rgba(255,255,255,0.35)', textAlign: 'center' }}>
        ForeSight PML<br />
        ForecastEnergAI ForeSight Platform
      </div>
    </aside>
  )
}
