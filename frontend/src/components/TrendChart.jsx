import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine, Brush,
} from 'recharts'

function fmtDate(s) {
  const d = new Date(s + 'T00:00:00')
  return `${d.getMonth() + 1}/${d.getDate()}`
}

const SERIES = [
  { key: 'Actual',          color: '#43a047', dash: undefined, width: 3, dot: true  },
  { key: 'Random Forest',   color: '#1e88e5', dash: '6 3',    width: 1.5, dot: false },
  { key: 'Gradient Boost',  color: '#8e24aa', dash: '4 2',    width: 1.5, dot: false },
  { key: 'Ridge',           color: '#f4511e', dash: '2 2',    width: 1.5, dot: false },
  { key: 'Extra Trees',     color: '#fb8c00', dash: '8 2',    width: 1.5, dot: false },
  { key: 'Ensemble (mean)', color: '#00897b', dash: undefined, width: 2.5, dot: false },
]

const MODEL_LINES = [
  { key: 'pred_random_forest',  label: 'Random Forest'   },
  { key: 'pred_gradient_boost', label: 'Gradient Boost'  },
  { key: 'pred_ridge',          label: 'Ridge'           },
  { key: 'pred_extra_trees',    label: 'Extra Trees'     },
  { key: 'pred_ensemble',       label: 'Ensemble (mean)' },
]

const ALL_KEYS = SERIES.map(s => s.key)

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="tooltip-date">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{p.value ?? '—'}</strong>
        </p>
      ))}
    </div>
  )
}

export default function TrendChart({ data }) {
  const [visible, setVisible] = useState(new Set(ALL_KEYS))

  const chartData = [...data].reverse().map(d => {
    const row = { date: fmtDate(d.log_date), Actual: d.tiredness_score ?? null }
    for (const m of MODEL_LINES) {
      row[m.label] = d[m.key] != null ? +d[m.key].toFixed(2) : null
    }
    return row
  })

  function toggleSeries(key) {
    setVisible(prev => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  function toggleAll() {
    setVisible(prev =>
      prev.size === ALL_KEYS.length ? new Set() : new Set(ALL_KEYS)
    )
  }

  const allOn = visible.size === ALL_KEYS.length

  return (
    <div className="chart-container">
      <h2>Tiredness Score Over Time</h2>
      <p className="subtitle">1 = very tired / dizzy &nbsp;·&nbsp; 10 = fully energised</p>

      <div className="series-toggles">
        <button
          className={`toggle-btn toggle-all ${allOn ? 'active' : ''}`}
          onClick={toggleAll}
        >
          All
        </button>
        {SERIES.map(s => (
          <button
            key={s.key}
            className={`toggle-btn ${visible.has(s.key) ? 'active' : ''}`}
            style={{
              '--toggle-color': s.color,
            }}
            onClick={() => toggleSeries(s.key)}
          >
            {s.key}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eeeeee" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis domain={[1, 10]} ticks={[1,2,3,4,5,6,7,8,9,10]} tick={{ fontSize: 11 }} width={24} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={5} stroke="#bdbdbd" strokeDasharray="4 4" />
          <Brush dataKey="date" height={24} stroke="#90caf9" fill="#f5f5f5" travellerWidth={8} />

          {SERIES.map(s => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.color}
              strokeWidth={s.width}
              strokeDasharray={s.dash}
              dot={s.dot ? { r: 4, fill: s.color } : false}
              activeDot={s.dot ? { r: 6 } : undefined}
              connectNulls={false}
              hide={!visible.has(s.key)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
