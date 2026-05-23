import React, { useMemo } from 'react'
import { BRAND, LABEL_BG, LABEL_TEXT_COLOR, MONTH_ABBR_EN, DOW_SHORT_EN, UI } from '../constants.js'

const s = {
  wrap: { fontFamily: 'sans-serif', fontSize: 11 },
  nav: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginBottom: 2,
  },
  navBtn: {
    background: 'none', border: 'none', color: UI.accent, cursor: 'pointer',
    fontSize: 16, padding: '0 6px', fontWeight: 'bold',
  },
  label: { color: BRAND.white, fontWeight: 900, fontSize: '0.9rem', textAlign: 'center', flex: 1 },
  grid: {
    display: 'grid', gridTemplateColumns: 'repeat(7,1fr)', gap: 2, marginTop: 2,
  },
  hdr: { textAlign: 'center', color: '#aaa', fontWeight: 'bold', padding: '2px 0' },
  day: {
    textAlign: 'center', padding: '4px 1px', borderRadius: 4,
    cursor: 'pointer', position: 'relative',
  },
  nodata: { color: '#555', cursor: 'default' },
}

export default function Calendar({ rows, region, selectedDate, calMonth, onDateClick, onMonthChange }) {
  const year  = calMonth.year
  const month = calMonth.month   // 1-indexed

  // Build date → label lookup for this region
  const dateInfo = useMemo(() => {
    const map = {}
    rows.filter(r => r.unique_id === region).forEach(r => {
      map[r.date] = { label: r.label, name: r.holiday_name || '' }
    })
    return map
  }, [rows, region])

  // Calendar grid
  const firstDOW = new Date(year, month - 1, 1).getDay()  // 0=Sun
  const firstMon = firstDOW === 0 ? 6 : firstDOW - 1       // convert to Mon-first
  const daysInMonth = new Date(year, month, 0).getDate()

  const cells = []
  for (let i = 0; i < firstMon; i++) cells.push(null)
  for (let d = 1; d <= daysInMonth; d++) {
    const key = `${year}-${String(month).padStart(2,'0')}-${String(d).padStart(2,'0')}`
    cells.push({ day: d, key })
  }

  const monthLabel = `${MONTH_ABBR_EN[month - 1]} ${year}`

  function prevYear()  { onMonthChange({ year: year - 1, month }) }
  function nextYear()  { onMonthChange({ year: year + 1, month }) }
  function prevMonth() {
    if (month === 1) onMonthChange({ year: year - 1, month: 12 })
    else             onMonthChange({ year, month: month - 1 })
  }
  function nextMonth() {
    if (month === 12) onMonthChange({ year: year + 1, month: 1 })
    else              onMonthChange({ year, month: month + 1 })
  }

  return (
    <div style={s.wrap}>
      {/* Year nav */}
      <div style={s.nav}>
        <button style={s.navBtn} onClick={prevYear}>◀</button>
        <span style={s.label}>{year}</span>
        <button style={s.navBtn} onClick={nextYear}>▶</button>
      </div>
      {/* Month nav */}
      <div style={s.nav}>
        <button style={s.navBtn} onClick={prevMonth}>◀</button>
        <span style={{ ...s.label, fontWeight: 'bold', fontSize: '0.82rem', color: '#ccc' }}>
          {MONTH_ABBR_EN[month - 1]}
        </span>
        <button style={s.navBtn} onClick={nextMonth}>▶</button>
      </div>
      {/* Day-of-week headers */}
      <div style={s.grid}>
        {DOW_SHORT_EN.map(d => (
          <div key={d} style={s.hdr}>{d}</div>
        ))}
        {cells.map((cell, i) => {
          if (!cell) return <div key={`e${i}`} />
          const info = dateInfo[cell.key]
          if (!info) {
            return (
              <div key={cell.key} style={{ ...s.day, ...s.nodata }}>
                {cell.day}
              </div>
            )
          }
          const isSelected = cell.key === selectedDate
          const bg    = LABEL_BG[info.label]   || '#eee'
          const clr   = LABEL_TEXT_COLOR[info.label] || '#333'
          return (
            <div
              key={cell.key}
              title={info.name || undefined}
              onClick={() => onDateClick(cell.key)}
              style={{
                ...s.day,
                background: bg,
                color: clr,
                outline: isSelected ? '2px solid #333' : 'none',
                fontWeight: isSelected ? 'bold' : 'normal',
              }}
            >
              {cell.day}
            </div>
          )
        })}
      </div>
    </div>
  )
}
