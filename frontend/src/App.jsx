import Dashboard from './components/Dashboard'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Health ML Tracker</h1>
        <p>Daily tiredness &amp; dizziness prediction from health metrics</p>
      </header>

      <main className="app-main">
        <Dashboard />
      </main>

      <footer className="app-footer">
        Data fetched daily from Google Sheets &nbsp;·&nbsp; Model retrained every Sunday
      </footer>
    </div>
  )
}
