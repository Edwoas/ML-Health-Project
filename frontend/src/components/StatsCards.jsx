// Scale is 1-10 (10 = best)
function scoreColor(v) {
  if (v == null) return '#9e9e9e'
  return v >= 7 ? '#43a047' : v >= 5 ? '#fb8c00' : '#e53935'
}

function Card({ value, label }) {
  const display = value != null ? value : '—'
  return (
    <div className="stat-card">
      <div className="stat-value" style={{ color: scoreColor(value) }}>{display}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

export default function StatsCards({ data }) {
  if (!data.length) return null

  const latest = data[0]
  const withBoth = data.filter(d => d.tiredness_score != null && d.pred_ensemble != null)
  const mae = withBoth.length
    ? (withBoth.reduce((s, d) => s + Math.abs(d.tiredness_score - d.pred_ensemble), 0) / withBoth.length).toFixed(2)
    : null

  return (
    <div className="stats-row">
      <Card value={latest.tiredness_score} label="Latest Actual" />
      <Card value={latest.pred_ensemble?.toFixed(1)} label="Latest Ensemble Prediction" />
      <Card value={mae} label={`Ensemble MAE (${withBoth.length} days)`} />
      <Card value={data.length} label="Days Logged" />
    </div>
  )
}
