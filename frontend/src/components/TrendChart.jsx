import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

function fmtDate(s) {
  const d = new Date(s + 'T00:00:00')
  return `${d.getMonth() + 1}/${d.getDate()}`
}

const MODEL_LINES = [
  { key: 'pred_random_forest',  label: 'Random Forest',    color: '#1e88e5', dash: '6 3' },
  { key: 'pred_gradient_boost', label: 'Gradient Boost',   color: '#8e24aa', dash: '4 2' },
  { key: 'pred_ridge',          label: 'Ridge',            color: '#f4511e', dash: '2 2' },
  { key: 'pred_extra_trees',    label: 'Extra Trees',      color: '#fb8c00', dash: '8 2' },
  { key: 'pred_ensemble',       label: 'Ensemble (mean)',  color: '#00897b', dash: undefined },
]

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
  const chartData = [...data].reverse().map(d => {
    const row = { date: fmtDate(d.log_date), Actual: d.tiredness_score ?? null }
    for (const m of MODEL_LINES) {
      row[m.label] = d[m.key] != null ? +d[m.key].toFixed(2) : null
    }
    return row
  })

  return (
    <div className="chart-container">
      <h2>Tiredness Score Over Time</h2>
      <p className="subtitle">1 = very tired / dizzy &nbsp;·&nbsp; 10 = fully energised</p>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eeeeee" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis domain={[1, 10]} ticks={[1,2,3,4,5,6,7,8,9,10]} tick={{ fontSize: 11 }} width={24} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={5} stroke="#bdbdbd" strokeDasharray="4 4" />

          {/* Actual score — thicker, solid, prominent */}
          <Line
            type="monotone"
            dataKey="Actual"
            stroke="#43a047"
            strokeWidth={3}
            dot={{ r: 4, fill: '#43a047' }}
            activeDot={{ r: 6 }}
            connectNulls={false}
          />

          {/* Four model predictions + ensemble */}
          {MODEL_LINES.map(m => (
            <Line
              key={m.key}
              type="monotone"
              dataKey={m.label}
              stroke={m.color}
              strokeWidth={m.dash ? 1.5 : 2.5}
              strokeDasharray={m.dash}
              dot={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
