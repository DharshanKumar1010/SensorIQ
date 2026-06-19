import { ChevronDown } from 'lucide-react'
import type { Asset } from '../api'

interface Props {
  assets: Asset[]
  selected: Asset | null
  onChange: (asset: Asset) => void
}

export default function AssetSelector({ assets, selected, onChange }: Props) {
  return (
    <div className="relative">
      <select
        value={selected?.id ?? ''}
        onChange={(e) => {
          const asset = assets.find((a) => a.id === e.target.value)
          if (asset) onChange(asset)
        }}
        className="appearance-none bg-slate-900 border border-slate-700 text-white rounded-lg pl-3 pr-8 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 cursor-pointer"
      >
        {assets.length === 0 && <option value="">No assets</option>}
        {assets.map((a) => (
          <option key={a.id} value={a.id}>
            {a.name}
          </option>
        ))}
      </select>
      <ChevronDown
        size={13}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none"
      />
    </div>
  )
}
