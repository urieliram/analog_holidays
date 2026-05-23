import React, { useMemo } from 'react'
import { LABEL_COLOR, HOURS } from '../constants.js'
import { UI } from '../constants.js'

const s = {
  wrap: { overflowX: 'auto', marginTop: 8 },
  table: {
    borderCollapse: 'collapse',
    width: '100%',
    fontSize: 11,
    fontFamily: 'monospace',
  },
  th: {
    background: UI.header_bg,
    color: '#fff',
    padding: '3px 6px',
    textAlign: 'center',
    border: `1px solid ${UI.border}`,
    fontWeight: 'bold',
  },
  td: {
    padding: '2px 4px',
    border: `1px solid ${UI.border}`,
    textAlign: 'center',
  },
  tdActive: {
    background: '#d4edda',
    color: '#155724',
    fontWeight: 'bold',
  },
  tdInactive: { color: '#aaa' },
}

export default function FeatureTable({ label }) {
  const isHoliday = label === 'holiday'
  const isSpecial = label === 'special_day'

  const rows = [
    { feature: 'holiday',     value: isHoliday ? 1 : 0 },
    { feature: 'special_day', value: isSpecial ? 1 : 0 },
  ]

  return (
    <div style={s.wrap}>
      <table style={s.table}>
        <thead>
          <tr>
            <th style={s.th}>feature</th>
            {HOURS.map(h => (
              <th key={h} style={s.th}>{String(h).padStart(2,'0')}h</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.feature}>
              <td style={{ ...s.td, textAlign: 'left', fontWeight: 'bold' }}>{row.feature}</td>
              {HOURS.map(h => (
                <td
                  key={h}
                  style={{ ...s.td, ...(row.value === 1 ? s.tdActive : s.tdInactive) }}
                >
                  {row.value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
