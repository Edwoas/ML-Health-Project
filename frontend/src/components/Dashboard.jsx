import { useState, useEffect } from 'react'
import { fetchDashboard, fetchCorrelations } from '../api'
import StatsCards            from './StatsCards'
import TrendChart            from './TrendChart'
import LogTable              from './LogTable'
import CorrelationChart      from './CorrelationChart'
import PredictionHistoryChart from './PredictionHistoryChart'

export default function Dashboard() {
  const [data, setData]       = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  // Correlation panel — loaded lazily on first reveal
  const [showCorr, setShowCorr]       = useState(false)
  const [corrData, setCorrData]       = useState(null)
  const [corrLoading, setCorrLoading] = useState(false)
  const [corrError, setCorrError]     = useState(null)

  // Prediction history panel
  const [showHistory, setShowHistory] = useState(false)

  useEffect(() => {
    fetchDashboard({ limit: 90 })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  function handleRevealCorr() {
    if (!showCorr && corrData === null && !corrLoading) {
      setCorrLoading(true)
      fetchCorrelations()
        .then(setCorrData)
        .catch(e => setCorrError(e.message))
        .finally(() => setCorrLoading(false))
    }
    setShowCorr(v => !v)
  }

  if (loading) return <div className="state-msg">Loading dashboard…</div>
  if (error)   return <div className="state-msg error">Error: {error}</div>

  return (
    <div className="dashboard">
      <StatsCards data={data} />
      <TrendChart data={data} />
      <LogTable   data={data} />

      {/* ── Prediction history ── */}
      <div className="corr-section">
        <button className="reveal-btn" onClick={() => setShowHistory(v => !v)}>
          {showHistory ? '▲ Hide Prediction History' : '▼ Show Prediction History'}
        </button>
        {showHistory && <PredictionHistoryChart dashboardData={data} />}
      </div>

      {/* ── Correlation analysis ── */}
      <div className="corr-section">
        <button className="reveal-btn" onClick={handleRevealCorr}>
          {showCorr ? '▲ Hide Correlation Analysis' : '▼ Reveal Correlation Analysis'}
        </button>

        {showCorr && (
          <div className="corr-disclaimer">
            <strong>Heads up:</strong> metric names have been anonymised (A, B, C…) to
            prevent this chart from influencing how you log future data.
            The mapping is fixed — a given letter always represents the same metric.
          </div>
        )}

        {showCorr && (
          corrLoading
            ? <div className="state-msg">Computing correlations…</div>
            : corrError
              ? <div className="state-msg error">Error: {corrError}</div>
              : corrData !== null && <CorrelationChart data={corrData} />
        )}
      </div>
    </div>
  )
}
