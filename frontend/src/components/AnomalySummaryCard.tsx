import { AlertTriangle, CheckCircle, Activity, TrendingUp, Hash } from 'lucide-react'
import type { AnomalySummary } from '../api'

interface Props {
  summary: AnomalySummary | null
  threshold?: number
}

function StatBox({
  icon,
  label,
  value,
  sub,
  alert,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  alert?: boolean
}) {
  return (
    <div
      className={`rounded-xl p-4 border ${
        alert
          ? 'bg-amber-500/5 border-amber-500/30'
          : 'bg-slate-900/60 border-slate-700/60'
      }`}
    >
      <div className={`mb-2.5 ${alert ? 'text-amber-400' : 'text-slate-500'}`}>{icon}</div>
      <p className="text-xl font-bold text-white leading-none">{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      <p className="text-xs text-slate-500 mt-1.5">{label}</p>
    </div>
  )
}

export default function AnomalySummaryCard({ summary, threshold }: Props) {
  if (!summary) {
    return (
      <div className="bg-[#1e293b] rounded-xl p-5 border border-slate-700/60 h-full flex flex-col">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
          Asset Summary
        </h2>
        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
          No data
        </div>
      </div>
    )
  }

  const anomalyPct = (summary.anomaly_rate * 100).toFixed(1)
  const isHighAlert = summary.anomaly_rate > 0.2
  const latestScore = summary.recent_anomalies[0]?.anomaly_score?.score
  const latestIsAnomaly = !!(latestScore !== undefined && threshold !== undefined && latestScore > threshold)

  return (
    <div className="bg-[#1e293b] rounded-xl p-5 border border-slate-700/60">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          Asset Summary
        </h2>
        {isHighAlert ? (
          <span className="inline-flex items-center gap-1 text-xs text-amber-400 bg-amber-400/10 border border-amber-400/20 px-2.5 py-0.5 rounded-full">
            <AlertTriangle size={10} />
            High anomaly rate
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-2.5 py-0.5 rounded-full">
            <CheckCircle size={10} />
            Normal
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatBox
          icon={<Activity size={16} />}
          label="Total Readings"
          value={summary.total_readings.toLocaleString()}
        />
        <StatBox
          icon={<Hash size={16} />}
          label="Anomalies Flagged"
          value={summary.total_anomalies.toLocaleString()}
          alert={summary.total_anomalies > 0}
        />
        <StatBox
          icon={<TrendingUp size={16} />}
          label="Anomaly Rate"
          value={`${anomalyPct}%`}
          alert={isHighAlert}
        />
        <StatBox
          icon={<Activity size={16} />}
          label="Latest Score"
          value={latestScore !== undefined ? latestScore.toFixed(5) : '—'}
          sub={latestIsAnomaly ? '⚠ above threshold' : undefined}
          alert={latestIsAnomaly}
        />
      </div>
    </div>
  )
}
