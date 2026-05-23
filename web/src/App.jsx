import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  loadData,
  getRegionDates,
  getDayRow,
  saveLocalLabel,
  exportLabelsCSV,
  exportNixtlaLong,
} from './dataStore.js'
import { REGIONS, LABEL_COLOR, MONTH_ABBR_EN, UI } from './constants.js'
import Sidebar       from './components/Sidebar.jsx'
import Calendar      from './components/Calendar.jsx'
import MetaBadges    from './components/MetaBadges.jsx'
import FeatureTable  from './components/FeatureTable.jsx'
import { MainChart, ClusterChart } from './components/Charts.jsx'

const LABEL_SYMBOL = { holiday: '🟢', special_day: '🔴', normal_day: '🔵' }

const s = {
  app: { display: 'flex', minHeight: '100vh', background: UI.bg_soft },
  main: { flex: 1, padding: '14px 20px', minWidth: 0 },
  loading: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontSize: 18, color: UI.text_sec },
  error: { color: 'red', padding: 32 },
  navRow: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 },
  navBtn: {
    background: UI.header_bg, color: '#fff',
    border: 'none', borderRadius: 6,
    padding: '6px 14px', cursor: 'pointer',
    fontWeight: 'bold', fontSize: 13,
  },
  labelBtn: (active, color) => ({
    flex: 1,
    padding: '8px 0',
    borderRadius: 8,
    border: `2px solid ${color}`,
    background: active ? color : 'transparent',
    color: active ? '#fff' : color,
    fontWeight: 'bold',
    fontSize: 14,
    cursor: 'pointer',
    transition: 'all 0.15s',
  }),
  labelRow: { display: 'flex', gap: 10, marginTop: 12, marginBottom: 12 },
  sectionTitle: {
    fontSize: 12, fontWeight: 'bold', color: UI.text_sec,
    textTransform: 'uppercase', letterSpacing: 1,
    borderBottom: `1px solid ${UI.border}`,
    paddingBottom: 3, marginTop: 10, marginBottom: 6,
  },
  footer: {
    marginTop: 16,
    borderTop: `1px solid ${UI.border}`,
    paddingTop: 8,
    fontSize: 11,
    color: UI.text_sec,
    display: 'flex',
    justifyContent: 'space-between',
  },
  hamburger: {
    position: 'fixed', top: 6, left: 6, zIndex: 9999,
    background: UI.header_bg, color: '#fff',
    border: 'none', borderRadius: 6,
    width: 32, height: 32,
    fontSize: 18, cursor: 'pointer',
    display: 'none',
  },
}

export default function App() {
  const [allRows, setAllRows]     = useState(null)
  const [error, setError]         = useState(null)
  const [region, setRegion]       = useState(REGIONS[0])
  const [dateIdx, setDateIdx]     = useState(0)
  const [unsaved, setUnsaved]     = useState(false)
  const [sidebarOpen, setSidebar] = useState(true)

  // Calendar month state
  const [calMonth, setCalMonth] = useState({ year: new Date().getFullYear(), month: new Date().getMonth() + 1 })

  // Load data on mount
  useEffect(() => {
    loadData()
      .then(rows => {
        setAllRows(rows)
        // Init cal month from first date in default region
        const dates = getRegionDates(rows, REGIONS[0])
        if (dates.length) {
          const d = new Date(dates[0])
          setCalMonth({ year: d.getFullYear(), month: d.getMonth() + 1 })
        }
      })
      .catch(e => setError(e.message))
  }, [])

  const dates = useMemo(
    () => allRows ? getRegionDates(allRows, region) : [],
    [allRows, region],
  )

  const clampedIdx = Math.min(dateIdx, Math.max(0, dates.length - 1))
  const selectedDate = dates[clampedIdx] || null

  const currentRow = useMemo(
    () => allRows && selectedDate ? getDayRow(allRows, region, selectedDate) : null,
    [allRows, region, selectedDate],
  )

  // Flagged dates (holiday + special_day) for prev/next navigation
  const flagged = useMemo(() => {
    if (!allRows) return []
    return allRows
      .filter(r => r.unique_id === region && (r.label === 'holiday' || r.label === 'special_day'))
      .map(r => r.date)
      .sort()
  }, [allRows, region])

  const flagPos   = flagged.filter(d => d <= selectedDate).length
  const flagTotal = flagged.length

  function prevFlagged() {
    const prev = [...flagged].reverse().find(d => d < selectedDate)
    if (prev) {
      const i = dates.indexOf(prev)
      if (i >= 0) { setDateIdx(i); setCalMonth(dateToMonth(prev)) }
    }
  }
  function nextFlagged() {
    const next = flagged.find(d => d > selectedDate)
    if (next) {
      const i = dates.indexOf(next)
      if (i >= 0) { setDateIdx(i); setCalMonth(dateToMonth(next)) }
    }
  }

  function dateToMonth(dateStr) {
    const d = new Date(dateStr)
    return { year: d.getFullYear(), month: d.getMonth() + 1 }
  }

  function handleRegionChange(r) {
    setRegion(r)
    setDateIdx(0)
  }

  function handleDateClick(dateStr) {
    const i = dates.indexOf(dateStr)
    if (i >= 0) {
      setDateIdx(i)
      setCalMonth(dateToMonth(dateStr))
    }
  }

  function handleLabel(newLabel) {
    if (!allRows || !selectedDate) return
    setAllRows(prev =>
      prev.map(r =>
        r.unique_id === region && r.date === selectedDate
          ? { ...r, label: newLabel }
          : r,
      ),
    )
    saveLocalLabel(region, selectedDate, newLabel)
    setUnsaved(true)
  }

  function handleSave() {
    setUnsaved(false)
    alert('Labels were saved to localStorage. Use Export CSV for a permanent file.')
  }

  // Label counts for sidebar
  const labelCounts = useMemo(() => {
    if (!allRows) return {}
    const rdf = allRows.filter(r => r.unique_id === region)
    return {
      holiday:     rdf.filter(r => r.label === 'holiday').length,
      special_day: rdf.filter(r => r.label === 'special_day').length,
      normal_day:  rdf.filter(r => r.label === 'normal_day').length,
    }
  }, [allRows, region])

  if (error)    return <div style={s.error}>Error loading data: {error}<br/>Run export_to_json.py first.</div>
  if (!allRows) return <div style={s.loading}>⏳ Loading data…</div>

  const calendarComp = () => (
    <Calendar
      rows={allRows}
      region={region}
      selectedDate={selectedDate}
      calMonth={calMonth}
      onDateClick={handleDateClick}
      onMonthChange={setCalMonth}
    />
  )

  return (
    <div style={s.app}>
      {sidebarOpen && (
        <Sidebar
          regions={REGIONS}
          region={region}
          onRegionChange={handleRegionChange}
          rows={allRows}
          calMonth={calMonth}
          onMonthChange={setCalMonth}
          selectedDate={selectedDate}
          onDateClick={handleDateClick}
          labelCounts={labelCounts}
          unsaved={unsaved}
          onSave={handleSave}
          onExportCSV={() => exportLabelsCSV(allRows)}
          onExportNixtla={() => exportNixtlaLong(allRows)}
          CalendarComp={calendarComp}
        />
      )}

      <main style={s.main}>
        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebar(v => !v)}
          style={{
            background: UI.header_bg, color: '#fff',
            border: 'none', borderRadius: 6,
            padding: '4px 10px', cursor: 'pointer',
            fontSize: 16, marginBottom: 8,
          }}
          title="Toggle sidebar"
        >
          ☰
        </button>

        {/* Metadata badges */}
        {currentRow && (
          <MetaBadges
            row={currentRow}
            region={region}
            flagPos={flagPos}
            flagTotal={flagTotal}
          />
        )}

        {/* Navigation + Main chart */}
        <div style={s.navRow}>
          <button style={s.navBtn} onClick={prevFlagged}>◀ PREV</button>
          <div style={{ flex: 1, minWidth: 0 }}>
            {selectedDate && (
              <MainChart
                rows={allRows}
                region={region}
                selectedDate={selectedDate}
              />
            )}
          </div>
          <button style={s.navBtn} onClick={nextFlagged}>NEXT ▶</button>
        </div>

        {/* Cluster reference grid */}
        {currentRow && (
          <>
            <div style={s.sectionTitle}>
              Historical profiles — {MONTH_ABBR_EN[currentRow.month - 1]} × weekday
            </div>
            <ClusterChart
              rows={allRows}
              region={region}
              month={currentRow.month}
              dow={currentRow.dow}
              selectedDate={selectedDate}
            />
          </>
        )}

        {/* Classification buttons */}
        {currentRow && (
          <div style={s.labelRow}>
            {['holiday', 'special_day', 'normal_day'].map(lbl => (
              <button
                key={lbl}
                style={s.labelBtn(currentRow.label === lbl, LABEL_COLOR[lbl])}
                onClick={() => handleLabel(lbl)}
              >
                {LABEL_SYMBOL[lbl]} {lbl.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                {currentRow.label === lbl ? ' ✓' : ''}
              </button>
            ))}
          </div>
        )}

        {/* Binary feature table */}
        {currentRow && (
          <>
            <div style={s.sectionTitle}>Binary feature</div>
            <FeatureTable label={currentRow.label} />
          </>
        )}

        {/* Footer */}
        <div style={s.footer}>
          <span>Precomputed by identify_HOLIDAYS.py · Metric: PEARSON · p97 · Excl. 2022</span>
          <span>ForeSight PML · ForecastEnergAI ForeSight Platform</span>
        </div>
      </main>
    </div>
  )
}
