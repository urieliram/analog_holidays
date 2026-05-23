import React from 'react'
import { LABEL_COLOR, LABEL_SYMBOL, MONTH_ABBR_EN, UI } from '../constants.js'

const pill = (bg, color, content) => (
  <span style={{
    background: bg,
    color,
    padding: '1px 8px',
    borderRadius: 10,
    fontSize: '0.78rem',
    fontWeight: 600,
    marginRight: 4,
    display: 'inline-block',
  }}>
    {content}
  </span>
)

export default function MetaBadges({ row, region, flagPos, flagTotal }) {
  if (!row) return null

  const label = row.label || 'normal_day'
  const labelColor = LABEL_COLOR[label] || '#888'
  const labelSymbol = LABEL_SYMBOL[label] || ''
  const labelText = label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  const dt = new Date(row.date + 'T12:00:00')
  const dayText = `${MONTH_ABBR_EN[dt.getMonth()]} ${dt.getDate()}, ${dt.getFullYear()}`
  const dowName = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][dt.getDay()]
  const title = (row.holiday_name || '').trim() || `${dowName}, ${row.date}`

  const isDeclared = Boolean(row.is_declared_holiday)
  const isOutlier = Boolean(row.is_outlier)
  const outlierScore = row.outlier_score != null ? Number(row.outlier_score).toFixed(4) : '—'
  const regShort = (region || '').replace('SEN_demand_', '')

  return (
    <div style={{ marginBottom: 6 }}>
      <span style={{ fontSize: '1.35rem', fontWeight: 700, marginRight: 6 }}>{title}</span>
      <span style={{
        background: labelColor, color: 'white',
        padding: '2px 10px', borderRadius: 12,
        fontSize: '0.85rem', fontWeight: 'bold',
        marginRight: 6,
      }}>
        {labelSymbol} {labelText}
      </span>
      {pill('#e8eaf6', '#3949ab', `📍 ${regShort}`)}
      {pill('#f3e5f5', '#7b1fa2', `🚩 ${flagPos}/${flagTotal}`)}
      {pill('#fafafa', '#555', `📅 ${dayText} · ${dowName}`)}
      {pill(isDeclared ? '#e8f5e9' : '#fce4ec', isDeclared ? '#2e7d32' : '#c62828', `${isDeclared ? '✅' : '❌'} Declared`)}
      {pill(isOutlier ? '#fce4ec' : '#f5f5f5', isOutlier ? '#c62828' : '#757575', `${isOutlier ? '🔴' : '⚪'} ${isOutlier ? 'Outlier' : 'Normal'} (${outlierScore})`)}
    </div>
  )
}
