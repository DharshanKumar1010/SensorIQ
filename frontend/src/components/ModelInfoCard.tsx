import { Cpu, BarChart3 } from 'lucide-react'
import type { ModelInfo } from '../api'

interface Props {
  modelInfo: ModelInfo | null
}

function Row({ label, value, mono, color }: { label: string; value: string; mono?: boolean; color?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-700/40 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-xs ${mono ? 'font-mono' : 'font-medium'} ${color ?? 'text-slate-300'}`}>
        {value}
      </span>
    </div>
  )
}

export default function ModelInfoCard({ modelInfo }: Props) {
  return (
    <div className="bg-[#1e293b] rounded-xl p-5 border border-slate-700/60 flex flex-col">
      <div className="flex items-center gap-2 mb-3">
        <Cpu size={14} className="text-amber-400" />
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          Active Model
        </h2>
      </div>

      {!modelInfo ? (
        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
          No model data
        </div>
      ) : (
        <div className="flex flex-col gap-0">
          <Row
            label="Type"
            value={modelInfo.model_type === 'autoencoder' ? 'LSTM Autoencoder' : 'Isolation Forest'}
          />
          <Row label="Version" value={modelInfo.model_version} mono />
          <Row label="Threshold" value={modelInfo.threshold.toFixed(6)} mono color="text-amber-400" />
          {modelInfo.pearson_corr !== null && (
            <Row
              label="Pearson r (RUL)"
              value={modelInfo.pearson_corr.toFixed(4)}
              mono
              color={modelInfo.pearson_corr < 0 ? 'text-emerald-400' : 'text-red-400'}
            />
          )}

          {modelInfo.rul_bucket_stats && modelInfo.rul_bucket_stats.length > 0 && (
            <div className="mt-4 pt-3 border-t border-slate-700/60">
              <div className="flex items-center gap-1.5 mb-3">
                <BarChart3 size={11} className="text-slate-500" />
                <span className="text-xs text-slate-500">Anomaly rate by RUL bucket</span>
              </div>
              <div className="space-y-2">
                {modelInfo.rul_bucket_stats.map((stat) => {
                  const pct = Math.min(100, stat.anomaly_rate * 100)
                  const isHot = pct > 40
                  return (
                    <div key={stat.bucket} className="flex items-center gap-2">
                      <span className="text-xs text-slate-600 w-12 shrink-0 font-mono">
                        {stat.bucket}
                      </span>
                      <div className="flex-1 bg-slate-700/50 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${isHot ? 'bg-amber-400' : 'bg-emerald-500'}`}
                          style={{ width: `${pct.toFixed(1)}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-500 w-8 text-right font-mono">
                        {pct.toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
