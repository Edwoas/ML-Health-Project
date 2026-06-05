import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, Cell, ResponsiveContainer,
} from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { label, correlation } = payload[0].payload
  const sign = correlation >= 0 ? 'positive' : 'negative'
  const abs  = Math.abs(correlation)
  const strength = abs >= 0.6 ? 'strong' : abs >= 0.3 ? 'moderate' : 'weak'
  return (
    <div className="chart-tooltip">
      <p className="tooltip-date">Metric {label}</p>
      <p style={{ color: correlation >= 0 ? '#1e88e5' : '#e53935' }}>
        r = <strong>{correlation.toFixed(3)}</strong>
      </p>
      <p style={{ fontSize: '0.78rem', color: '#757575', marginTop: 4 }}>
        {strength} {sign} correlation
      </p>
    </div>
  )
}

export default function CorrelationChart({ data }) {
  if (!data.length) {
    return (
      <div className="empty">
        Not enough data yet — need at least 5 logged days with a tiredness score.
      </div>
    )
  }

  const chartHeight = 60 + data.length * 44

  return (
    <div className="chart-container">
      <h2>Correlation with Tiredness Score</h2>
      <p className="subtitle">
        Each letter is one anonymised health metric. Blue bars = positive correlation
        (higher value → more tired). Red bars = negative (higher value → less tired).
        Sorted by strength.
      </p>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, left: 12, bottom: 4 }}
          barCategoryGap="30%"
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eeeeee" />
          <XAxis
            type="number"
            domain={[-1, 1]}
            tickCount={9}
            tick={{ fontSize: 11 }}
            tickFormatter={v => v.toFixed(1)}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fontSize: 15, fontWeight: 700 }}
            width={30}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.04)' }} />
          <ReferenceLine x={0} stroke="#9e9e9e" strokeWidth={2} />
          <Bar dataKey="correlation" radius={[0, 4, 4, 0]} isAnimationActive>
            {data.map(entry => (
              <Cell
                key={entry.label}
                fill={entry.correlation >= 0 ? '#1e88e5' : '#e53935'}
                fillOpacity={0.55 + Math.abs(entry.correlation) * 0.45}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
