import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { fetchHistoryDates, fetchPredictionHistory } from '../api'

const MODEL_LINES = [
  { key: 'pred_random_forest',  label: 'Random Forest',   color: '#1e88e5' },
  { key: 'pred_gradient_boost', label: 'Gradient Boost',  color: '#8e24aa' },
  { key: 'pred_ridge',          label: 'Ridge',           color: '#f4511e' },
  { key: 'pred_extra_trees',    label: 'Extra Trees',     color: '#fb8c00' },
  { key: 'pred_ensemble',       label: 'Ensemble',        color: '#00897b' },
]

function fmtRunAt(iso) {
  const d = new Date(iso)
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="tooltip-date">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{p.value?.toFixed(2) ?? '—'}</strong>
        </p>
      ))}
    </div>
  )
}

export default function PredictionHistoryChart({ dashboardData }) {
  const [dates, setDates]         = useState([])
  const [selectedDate, setDate]   = useState('')
  const [history, setHistory]     = useState([])
  const [loading, setLoading]     = useState(true)
  const [histLoading, setHLoading] = useState(false)
  const [error, setError]         = useState(null)

  // Load available dates on mount
  useEffect(() => {
    fetchHistoryDates()
      .then(d => {
        setDates(d)
        if (d.length) setDate(d[0])
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Load history whenever selected date changes
  useEffect(() => {
    if (!selectedDate) return
    setHLoading(true)
    fetchPredictionHistory(selectedDate)
      .then(setHistory)
      .catch(e => setError(e.message))
      .finally(() => setHLoading(false))
  }, [selectedDate])

  if (loading) return <div className="state-msg">Loading dates…</div>
  if (error)   return <div className="state-msg error">Error: {error}</div>
  if (!dates.length) {
    return (
      <div className="empty">
        No prediction history yet — it builds up after the pipeline has run at least once.
      </div>
    )
  }

  // Look up the ground-truth tiredness score for the selected date
  const actual = dashboardData.find(d => d.log_date === selectedDate)?.tiredness_score ?? null

  const chartData = history.map(row => {
    const point = { run_at: fmtRunAt(row.run_at) }
    for (const m of MODEL_LINES) point[m.label] = row[m.key] ?? null
    return point
  })

  const onlyOneRun = history.length === 1

  return (
    <div className="chart-container">
      <div className="history-header">
        <div>
          <h2>Prediction History</h2>
          <p className="subtitle">
            How each model's prediction for a given date evolved as more training data was added.
          </p>
        </div>
        <select
          className="date-select"
          value={selectedDate}
          onChange={e => setDate(e.target.value)}
        >
          {dates.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>

      {actual !== null && (
        <p className="history-actual">
          Actual tiredness score for {selectedDate}: <strong>{actual} / 10</strong>
        </p>
      )}

      {onlyOneRun && (
        <p className="history-note">
          Only one inference run so far for this date. The chart becomes meaningful
          after the model has been retrained at least once with new data.
        </p>
      )}

      {histLoading
        ? <div className="state-msg">Loading…</div>
        : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eeeeee" />
              <XAxis dataKey="run_at" tick={{ fontSize: 11 }} />
              <YAxis domain={[1, 10]} ticks={[1,2,3,4,5,6,7,8,9,10]} tick={{ fontSize: 11 }} width={24} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />

              {actual !== null && (
                <ReferenceLine
                  y={actual}
                  stroke="#43a047"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  label={{ value: `Actual: ${actual}`, position: 'right', fontSize: 11, fill: '#43a047' }}
                />
              )}

              {MODEL_LINES.map(m => (
                <Line
                  key={m.key}
                  type="monotone"
                  dataKey={m.label}
                  stroke={m.color}
                  strokeWidth={m.label === 'Ensemble' ? 2.5 : 1.5}
                  dot={{ r: 4, fill: m.color }}
                  activeDot={{ r: 6 }}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )
      }
    </div>
  )
}
