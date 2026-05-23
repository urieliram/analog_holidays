import React, { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { LABEL_COLOR, HOURS } from '../constants.js'

const HOUR_COUNT = 24

/**
 * Multi-day context chart — continuous ±N-day window around selectedDate.
 * Selected day shown bold + dark border frame.
 */
export function MainChart({ rows, region, selectedDate }) {
  const { data, layout } = useMemo(() => {
    const rdf = rows
      .filter(r => r.unique_id === region)
      .sort((a, b) => a.date.localeCompare(b.date))

    const idx  = rdf.findIndex(r => r.date === selectedDate)
    const half = 4
    const start = Math.max(0, idx - half)
    const end   = Math.min(rdf.length, idx + half + 1)
    const window = rdf.slice(start, end)

    const traces = []
    const shapes = []

    window.forEach(r => {
      const isSel = r.date === selectedDate
      const color = LABEL_COLOR[r.label] || '#888'
      const xs    = r.hours.map((_, h) => `${r.date}T${String(h).padStart(2,'0')}:00:00`)
      traces.push({
        x: xs,
        y: r.hours,
        type: 'scatter',
        mode: isSel ? 'lines+markers' : 'lines',
        line: { color, width: isSel ? 3 : 1.2 },
        marker: isSel ? { size: 4, color } : { size: 0 },
        opacity: isSel ? 1 : 0.55,
        showlegend: false,
        hovertemplate: `${r.date} %{x|%H:%M}<br>%{y:,.1f} MW<extra></extra>`,
      })
      // Day border shape for selected
      if (isSel) {
        const x0 = `${r.date}T00:00:00`
        const x1 = `${r.date}T23:59:00`
        shapes.push({
          type: 'rect', x0, x1, y0: 0, y1: 1, yref: 'paper',
          line: { color: '#333333', width: 2 },
          fillcolor: 'rgba(0,0,0,0)',
        })
      }
    })

    // Midnight dashed lines
    window.slice(1).forEach(r => {
      shapes.push({
        type: 'line',
        x0: `${r.date}T00:00:00`, x1: `${r.date}T00:00:00`,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: '#cccccc', width: 1, dash: 'dot' },
      })
    })

    // Tick per day at noon
    const tickvals = window.map(r => `${r.date}T12:00:00`)
    const ticktext = window.map(r => {
      const d   = new Date(r.date)
      const dow = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()]
      return `${dow}\n${String(d.getDate()).padStart(2,'0')} ${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()]}`
    })

    const layout = {
      xaxis: { tickvals, ticktext, tickfont: { size: 10 }, showgrid: false },
      yaxis: { title: 'Demand (MW)' },
      height: 340,
      margin: { l: 55, r: 20, t: 40, b: 45 },
      showlegend: false,
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'transparent',
      shapes,
    }

    return { data: traces, layout }
  }, [rows, region, selectedDate])

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  )
}

/**
 * Weekly context: 7 mini subplots centred on selectedDate.
 */
export function WeeklyContextChart({ rows, region, selectedDate }) {
  const { data, layout } = useMemo(() => {
    const rdf = rows
      .filter(r => r.unique_id === region)
      .sort((a, b) => a.date.localeCompare(b.date))

    const idx   = rdf.findIndex(r => r.date === selectedDate)
    const half  = 3
    const start = Math.max(0, idx - half)
    const end   = Math.min(rdf.length, start + 7)
    const window = rdf.slice(start, end)

    const traces = []
    const subplotTitles = window.map(r => {
      const d   = new Date(r.date)
      const dow = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()]
      const star = r.date === selectedDate ? ' ★' : ''
      return `${dow}<br>${String(d.getMonth()+1).padStart(2,'0')}/${String(d.getDate()).padStart(2,'0')}${star}`
    })

    window.forEach((r, j) => {
      const isCur = r.date === selectedDate
      const color = LABEL_COLOR[r.label] || '#888'
      traces.push({
        x: HOURS,
        y: r.hours,
        type: 'scatter',
        mode: 'lines',
        line: { color, width: isCur ? 2.5 : 1.2 },
        opacity: isCur ? 1 : 0.6,
        showlegend: false,
        xaxis: j === 0 ? 'x' : `x${j+1}`,
        yaxis: j === 0 ? 'y' : `y${j+1}`,
        hovertemplate: `${r.date}<br>%{y:,.1f} MW<extra></extra>`,
      })
    })

    const n = window.length
    const axes = {}
    window.forEach((r, j) => {
      const isCur = r.date === selectedDate
      const xk = j === 0 ? 'xaxis' : `xaxis${j+1}`
      const yk = j === 0 ? 'yaxis' : `yaxis${j+1}`
      const dom = [j/n + 0.005, (j+1)/n - 0.005]
      axes[xk] = { domain: dom, tickvals: [0,12,23], tickfont: { size: 8 }, showgrid: false, anchor: j === 0 ? 'y' : `y${j+1}` }
      axes[yk] = { domain: [0,1], showticklabels: false, showgrid: true, gridcolor: '#eeeeee', anchor: j === 0 ? 'x' : `x${j+1}` }
      if (isCur) {
        axes[xk].linecolor = '#333'
        axes[xk].linewidth = 2
        axes[xk].mirror = true
        axes[yk].linecolor = '#333'
        axes[yk].linewidth = 2
        axes[yk].mirror = true
      }
    })

    const annotations = subplotTitles.map((t, j) => ({
      text: t,
      x: (j + 0.5) / n,
      y: 1.02,
      xref: 'paper', yref: 'paper',
      showarrow: false,
      font: { size: 9 },
    }))

    const layout = {
      ...axes,
      annotations,
      height: 200,
      margin: { l: 10, r: 10, t: 40, b: 10 },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'transparent',
      showlegend: false,
    }

    return { data: traces, layout }
  }, [rows, region, selectedDate])

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  )
}

/**
 * Cluster reference grid: 7 DOW subplots for current month.
 */
export function ClusterChart({ rows, region, month, dow, selectedDate }) {
  const { data, layout } = useMemo(() => {
    const rdf = rows.filter(r => r.unique_id === region && r.month === month)
    const traces = []
    const axes   = {}
    const annotations = []
    const DOW_EN = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    for (let ci = 0; ci < 7; ci++) {
      const sub    = rdf.filter(r => r.dow === ci)
      const isCur  = ci === dow
      const profiles = []
      const xk = ci === 0 ? 'xaxis' : `xaxis${ci+1}`
      const yk = ci === 0 ? 'yaxis' : `yaxis${ci+1}`
      const dom = [ci/7 + 0.004, (ci+1)/7 - 0.004]

      sub.forEach(r => {
        const isToday = r.date === selectedDate
        const color   = LABEL_COLOR[r.label] || '#888'
        traces.push({
          x: HOURS, y: r.hours,
          type: 'scatter', mode: 'lines',
          line: { color, width: isToday ? 2.5 : 0.9 },
          opacity: isToday ? 1 : 0.25,
          showlegend: false,
          xaxis: ci === 0 ? 'x' : `x${ci+1}`,
          yaxis: ci === 0 ? 'y' : `y${ci+1}`,
          hovertemplate: `${r.date}<br>%{y:,.1f} MW<extra></extra>`,
        })
        if (!r.hours.some(v => v === null || isNaN(v))) profiles.push(r.hours)
      })

      // Centroid
      if (profiles.length > 0) {
        const centroid = HOURS.map(h => {
          const vals = profiles.map(p => p[h]).filter(v => v !== null && !isNaN(v))
          return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
        })
        traces.push({
          x: HOURS, y: centroid,
          type: 'scatter', mode: 'lines',
          line: { color: 'black', width: 2, dash: 'dot' },
          opacity: 0.85,
          showlegend: false,
          xaxis: ci === 0 ? 'x' : `x${ci+1}`,
          yaxis: ci === 0 ? 'y' : `y${ci+1}`,
          hovertemplate: `Centroide<br>%{y:,.1f} MW<extra></extra>`,
        })
      }

      axes[xk] = {
        domain: dom,
        tickvals: [0, 6, 12, 18, 23],
        ticktext: ['0h','6h','12h','18h','23h'],
        tickfont: { size: 7 }, showgrid: false,
        anchor: ci === 0 ? 'y' : `y${ci+1}`,
        ...(isCur ? { linecolor: '#333', linewidth: 2, mirror: true } : {}),
      }
      axes[yk] = {
        domain: [0, 1],
        showticklabels: false, showgrid: true, gridcolor: '#eeeeee',
        anchor: ci === 0 ? 'x' : `x${ci+1}`,
        ...(isCur ? { linecolor: '#333', linewidth: 2, mirror: true } : {}),
      }

      annotations.push({
        text: DOW_EN[ci],
        x: (ci + 0.5) / 7,
        y: 1.04,
        xref: 'paper', yref: 'paper',
        showarrow: false,
        font: { size: 10, color: isCur ? '#014754' : '#555' },
      })
    }

    const layout = {
      ...axes,
      annotations,
      height: 240,
      margin: { l: 20, r: 10, t: 44, b: 8 },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'transparent',
      showlegend: false,
    }

    return { data: traces, layout }
  }, [rows, region, month, dow, selectedDate])

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  )
}
