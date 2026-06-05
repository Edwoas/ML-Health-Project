const BASE = '/api'

export async function fetchDashboard({ limit = 90, startDate, endDate } = {}) {
  const params = new URLSearchParams({ limit })
  if (startDate) params.set('start_date', startDate)
  if (endDate)   params.set('end_date', endDate)

  const res = await fetch(`${BASE}/dashboard/?${params}`)
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function fetchCorrelations() {
  const res = await fetch(`${BASE}/correlations/`)
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function fetchHistoryDates() {
  const res = await fetch(`${BASE}/prediction-history/dates`)
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function fetchPredictionHistory(logDate) {
  const res = await fetch(`${BASE}/prediction-history/?log_date=${logDate}`)
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}
