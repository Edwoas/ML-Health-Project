const WEATHER_ICON = { sunny: '☀️', cloudy: '☁️', rainy: '🌧️', snowy: '❄️' }

// Scale 1-10; higher = better for most metrics, but some are inverted
// For display, green = high value (good for sleep/social, bad for stress/conflicts)
function ScoreBadge({ value, invert = false }) {
  if (value == null) return <span className="na">—</span>
  const n = parseFloat(value)
  const pct = invert ? (11 - n) / 10 : n / 10   // invert: low value = good
  const cls = pct >= 0.7 ? 'badge-good' : pct >= 0.4 ? 'badge-mid' : 'badge-bad'
  return <span className={`badge ${cls}`}>{n % 1 === 0 ? n : n.toFixed(1)}</span>
}

function BoolCell({ value }) {
  if (value == null) return <span className="na">—</span>
  return <span className={value ? 'bool-yes' : 'bool-no'}>{value ? '✓' : '✗'}</span>
}

function DiffCell({ actual, predicted }) {
  if (actual == null || predicted == null) return <span className="na">—</span>
  const diff = predicted - actual
  const cls = Math.abs(diff) < 0.5 ? 'diff-ok' : Math.abs(diff) < 1.5 ? 'diff-warn' : 'diff-bad'
  return <span className={cls}>{diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)}</span>
}

const PRED_COLS = [
  { key: 'pred_random_forest',  label: 'RF' },
  { key: 'pred_gradient_boost', label: 'GB' },
  { key: 'pred_ridge',          label: 'Ridge' },
  { key: 'pred_extra_trees',    label: 'ET' },
  { key: 'pred_ensemble',       label: 'Ens.' },
]

export default function LogTable({ data }) {
  if (!data.length) {
    return (
      <div className="empty">
        No data yet — fill your Google Sheet and wait for the Airflow pipeline to run at midnight.
      </div>
    )
  }

  return (
    <div className="table-card">
      <h2>Daily Log</h2>
      <p className="subtitle" style={{ marginBottom: 12 }}>
        RF = Random Forest &nbsp;·&nbsp; GB = Gradient Boost &nbsp;·&nbsp;
        ET = Extra Trees &nbsp;·&nbsp; Ens. = Ensemble mean
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Weather</th>
              <th>Temp °C</th>
              <th>Meals</th>
              <th>Junk</th>
              <th>Lunch</th>
              <th>Dinner</th>
              <th>Sugar</th>
              <th>Stomach</th>
              <th>Sleep</th>
              <th>Gym</th>
              <th>AM Ex.</th>
              <th>Stress</th>
              <th>Social</th>
              <th>Conflicts</th>
              <th>Actual</th>
              {PRED_COLS.map(c => <th key={c.key}>{c.label}</th>)}
              <th>Δ Ens.</th>
            </tr>
          </thead>
          <tbody>
            {data.map(row => (
              <tr key={row.log_date}>
                <td className="td-date">{row.log_date}</td>
                <td>
                  {row.weather
                    ? `${WEATHER_ICON[row.weather?.toLowerCase()] ?? ''} ${row.weather}`
                    : <span className="na">—</span>}
                </td>
                <td>{row.temperature ?? <span className="na">—</span>}</td>
                <td>{row.num_meals    ?? <span className="na">—</span>}</td>
                <td>{row.num_junkfood ?? <span className="na">—</span>}</td>
                <td><ScoreBadge value={row.meal_size_lunch}  /></td>
                <td><ScoreBadge value={row.meal_size_dinner} /></td>
                <td><ScoreBadge value={row.sugar}            invert /></td>
                <td><ScoreBadge value={row.stomach_issues}   invert /></td>
                <td><ScoreBadge value={row.sleep_quality}    /></td>
                <td><BoolCell   value={row.exercise_gym}     /></td>
                <td><BoolCell   value={row.exercise_morning} /></td>
                <td><ScoreBadge value={row.stress}           invert /></td>
                <td><ScoreBadge value={row.social_activity}  /></td>
                <td><ScoreBadge value={row.social_conflicts} invert /></td>
                <td><ScoreBadge value={row.tiredness_score}  /></td>
                {PRED_COLS.map(c => (
                  <td key={c.key}>
                    <ScoreBadge value={row[c.key]} />
                  </td>
                ))}
                <td>
                  <DiffCell actual={row.tiredness_score} predicted={row.pred_ensemble} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
