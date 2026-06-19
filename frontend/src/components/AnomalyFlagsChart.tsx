import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { ScoredReading } from '../api'

interface Props {
  readings: ScoredReading[]
  threshold: number
}

export default function AnomalyFlagsChart({ readings, threshold }: Props) {
  const data = readings
    .filter((r) => r.anomaly_score !== null)
    .sort((a, b) => a.reading.cycle - b.reading.cycle)
    .map((r) => ({
      cycle: r.reading.cycle,
      score: r.anomaly_score!.score,
      is_anomaly: r.anomaly_score!.is_anomaly,
    }))

  if (data.length === 0) {
    return (
      <div className="h-52 flex items-center justify-center text-slate-600 text-sm">
        No scored readings yet
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={210}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: -10 }}>
        <defs>
          <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" vertical={false} />
        <XAxis
          dataKey="cycle"
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          label={{ value: 'Cycle', position: 'insideBottom', offset: -4, fill: '#475569', fontSize: 10 }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={52}
          tickFormatter={(v: number) => v.toFixed(4)}
        />
        <Tooltip
          contentStyle={{
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: '#94a3b8', marginBottom: 4 }}
          labelFormatter={(v) => `Cycle ${v}`}
          formatter={(value: unknown) => [(value as number).toFixed(6), 'Score']}
        />
        <ReferenceLine
          y={threshold}
          stroke="#ef4444"
          strokeDasharray="6 3"
          strokeWidth={1.5}
          label={{
            value: `Threshold ${threshold.toFixed(5)}`,
            fill: '#ef4444',
            fontSize: 9,
            position: 'insideTopRight',
          }}
        />
        <Area
          type="monotone"
          dataKey="score"
          stroke="#f59e0b"
          strokeWidth={1.5}
          fill="url(#scoreGrad)"
          dot={false}
          activeDot={{ r: 3, fill: '#f59e0b' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
