/**
 * dataStore.js
 * Load audit_data.json from /public/data/ and expose access helpers.
 * Edited labels are stored in localStorage for local persistence.
 */

const LS_KEY = 'pml_audit_labels_v1'

export async function loadData() {
  const res = await fetch('./data/audit_data.json')
  if (!res.ok) throw new Error(`Could not load audit_data.json: ${res.status}`)
  const rows = await res.json()
  // Merge localStorage labels on top
  const saved = loadLocalLabels()
  if (Object.keys(saved).length > 0) {
    rows.forEach(r => {
      const key = `${r.unique_id}|${r.date}`
      if (saved[key]) r.label = saved[key]
    })
  }
  return rows
}

export function loadLocalLabels() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || '{}')
  } catch {
    return {}
  }
}

export function saveLocalLabel(unique_id, date, label) {
  const labels = loadLocalLabels()
  labels[`${unique_id}|${date}`] = label
  localStorage.setItem(LS_KEY, JSON.stringify(labels))
}

export function exportLabelsJSON(rows) {
  const out = rows.map(r => ({ unique_id: r.unique_id, date: r.date, label: r.label }))
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `audit_labels_${new Date().toISOString().slice(0,10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

export function exportLabelsCSV(rows) {
  const header = 'unique_id,date,label\n'
  const body   = rows.map(r => `${r.unique_id},${r.date},${r.label}`).join('\n')
  const blob   = new Blob([header + body], { type: 'text/csv' })
  const url    = URL.createObjectURL(blob)
  const a      = document.createElement('a')
  a.href       = url
  a.download   = `audit_labels_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function exportNixtlaLong(rows) {
  const lines = ['unique_id,ds,y']
  rows.forEach(r => {
    for (let h = 0; h < 24; h++) {
      const hStr = String(h).padStart(2, '0')
      const ds   = `${r.date} ${hStr}:00:00`
      lines.push(`${r.unique_id}_holiday,${ds},${r.label === 'holiday' ? 1 : 0}`)
      lines.push(`${r.unique_id}_special_day,${ds},${r.label === 'special_day' ? 1 : 0}`)
    }
  })
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `holiday_audit_nixtla_${new Date().toISOString().slice(0,10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function getRegionDates(rows, region) {
  return [...new Set(rows.filter(r => r.unique_id === region).map(r => r.date))].sort()
}

export function getDayRow(rows, region, date) {
  return rows.find(r => r.unique_id === region && r.date === date) || null
}
