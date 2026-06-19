import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { ScoredReading } from '../api'

interface Props {
  readings: ScoredReading[]
}

const SENSORS = ['s2', 's4', 's7'] as const
const COLORS: Record<string, string> = {
  s2: '#60a5fa',
  s4: '#34d399',
  s7: '#f59e0b',
}

export default function SensorTimelineChart({ readings }: Props) {
  const data = readings
    .slice()
    .sort((a, b) => a.reading.cycle - b.reading.cycle)
    .map((r) => ({
      cycle: r.reading.cycle,
      s2: r.reading.sensor_data['s2'] ?? null,
      s4: r.reading.sensor_data['s4'] ?? null,
      s7: r.reading.sensor_data['s7'] ?? null,
    }))

  if (data.length === 0) {
    return (
      <div className="h-52 flex items-center justify-center text-slate-600 text-sm">
        No readings yet
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={210}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 8, left: -10 }}>
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
          width={36}
        />
        <Tooltip
          contentStyle={{
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: '#94a3b8', marginBottom: 4 }}
          itemStyle={{ color: '#e2e8f0' }}
          labelFormatter={(v) => `Cycle ${v}`}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 8 }}
          iconType="line"
        />
        {SENSORS.map((s) => (
          <Line
            key={s}
            type="monotone"
            dataKey={s}
            stroke={COLORS[s]}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            activeDot={{ r: 3, fill: COLORS[s] }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
