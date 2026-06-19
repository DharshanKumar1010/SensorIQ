import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, LogOut, RefreshCw } from 'lucide-react'
import {
  api,
  type Asset,
  type ScoredReading,
  type AnomalySummary,
  type ModelInfo,
} from '../api'
import AssetSelector from '../components/AssetSelector'
import SensorTimelineChart from '../components/SensorTimelineChart'
import AnomalyFlagsChart from '../components/AnomalyFlagsChart'
import AnomalySummaryCard from '../components/AnomalySummaryCard'
import ModelInfoCard from '../components/ModelInfoCard'

export default function DashboardPage() {
  const navigate = useNavigate()
  const [assets, setAssets] = useState<Asset[]>([])
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const [readings, setReadings] = useState<ScoredReading[]>([])
  const [summary, setSummary] = useState<AnomalySummary | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Load asset list once on mount
  useEffect(() => {
    api.getAssets()
      .then((data) => {
        setAssets(data)
        if (data.length > 0) setSelectedAsset(data[0])
      })
      .catch(() => setError('Failed to load assets. Is the API running?'))
  }, [])

  // Re-fetch data whenever the selected asset changes
  useEffect(() => {
    if (!selectedAsset) return
    setLoading(true)
    setError('')

    Promise.all([
      api.getScoredReadings(selectedAsset.id).catch(() => [] as ScoredReading[]),
      api.getAnomalySummary(selectedAsset.id).catch(() => null),
      api.getModelInfo(selectedAsset.id).catch(() => null),
    ]).then(([r, s, m]) => {
      setReadings(r)
      setSummary(s)
      setModelInfo(m)
    }).finally(() => setLoading(false))
  }, [selectedAsset])

  function logout() {
    localStorage.removeItem('token')
    navigate('/login')
  }

  function refresh() {
    if (selectedAsset) {
      // Re-trigger the useEffect by creating a new reference
      setSelectedAsset({ ...selectedAsset })
    }
  }

  return (
    <div className="min-h-screen bg-[#0f172a] text-white">
      {/* Top bar */}
      <header className="bg-[#1e293b] border-b border-slate-700/60 px-5 py-3.5 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center shrink-0">
            <Activity size={16} className="text-amber-400" />
          </div>
          <span className="font-bold text-base text-white tracking-tight">SensorIQ</span>
          <span className="text-slate-600 hidden sm:block text-sm">/ Dashboard</span>
        </div>

        <div className="flex items-center gap-3">
          <AssetSelector assets={assets} selected={selectedAsset} onChange={setSelectedAsset} />
          <button
            onClick={refresh}
            title="Refresh"
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700 transition-colors"
          >
            <RefreshCw size={15} />
          </button>
          <button
            onClick={logout}
            className="flex items-center gap-1.5 text-slate-400 hover:text-white transition-colors text-sm px-2 py-1.5 rounded-lg hover:bg-slate-700"
          >
            <LogOut size={14} />
            <span className="hidden sm:inline">Sign out</span>
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="px-5 py-5 max-w-7xl mx-auto space-y-5">
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-5 py-3">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {!selectedAsset && !error && (
          <div className="text-center py-24 text-slate-500">
            <Activity size={40} className="mx-auto mb-4 opacity-20" />
            <p className="text-sm">No assets found. Create one via the API to get started.</p>
          </div>
        )}

        {selectedAsset && loading && (
          <div className="text-center py-24 text-slate-500">
            <div className="w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm">Loading sensor data…</p>
          </div>
        )}

        {selectedAsset && !loading && (
          <>
            {/* Asset label */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 uppercase tracking-widest">Asset</span>
              <span className="text-sm font-medium text-white">{selectedAsset.name}</span>
              <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">
                {selectedAsset.asset_type}
              </span>
            </div>

            {/* Summary + Model info */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <AnomalySummaryCard summary={summary} threshold={modelInfo?.threshold} />
              </div>
              <ModelInfoCard modelInfo={modelInfo} />
            </div>

            {/* Sensor timeline */}
            <div className="bg-[#1e293b] rounded-xl p-5 border border-slate-700/60">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                  Sensor Timeline
                </h2>
                <span className="text-xs text-slate-600">s2 · s4 · s7</span>
              </div>
              <SensorTimelineChart readings={readings} />
            </div>

            {/* Anomaly flags */}
            <div className="bg-[#1e293b] rounded-xl p-5 border border-slate-700/60">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                  Anomaly Score Over Time
                </h2>
                {modelInfo && (
                  <span className="text-xs text-slate-600">
                    threshold&nbsp;
                    <span className="text-amber-500 font-mono">{modelInfo.threshold.toFixed(5)}</span>
                  </span>
                )}
              </div>
              <AnomalyFlagsChart
                readings={readings}
                threshold={modelInfo?.threshold ?? 0.005618}
              />
            </div>
          </>
        )}
      </main>
    </div>
  )
}
