import axios from 'axios'

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

const client = axios.create({ baseURL: BASE_URL })

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── Types ────────────────────────────────────────────────────────────────────

export interface Asset {
  id: string
  user_id: string
  name: string
  asset_type: string
  created_at: string
}

export interface AnomalyScoreOut {
  id: string
  reading_id: string
  model_version: string
  score: number
  is_anomaly: boolean
  created_at: string
}

export interface ScoredReading {
  reading: {
    id: string
    asset_id: string
    timestamp: string
    cycle: number
    sensor_data: Record<string, number>
    created_at: string
  }
  anomaly_score: AnomalyScoreOut | null
}

export interface AnomalySummary {
  asset_id: string
  total_readings: number
  total_anomalies: number
  anomaly_rate: number
  recent_anomalies: ScoredReading[]
}

export interface RulBucketStat {
  bucket: string
  mean_score: number
  anomaly_rate: number
  n_windows: number
}

export interface ModelInfo {
  model_type: string
  model_version: string
  threshold: number
  pearson_corr: number | null
  rul_bucket_stats: RulBucketStat[] | null
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  async login(email: string, password: string): Promise<string> {
    const { data } = await client.post<{ access_token: string }>('/auth/login', { email, password })
    return data.access_token
  },

  async getAssets(): Promise<Asset[]> {
    const { data } = await client.get<Asset[]>('/assets')
    return data
  },

  async getScoredReadings(assetId: string): Promise<ScoredReading[]> {
    const { data } = await client.get<ScoredReading[]>(`/anomalies/${assetId}`)
    return data
  },

  async getAnomalySummary(assetId: string): Promise<AnomalySummary> {
    const { data } = await client.get<AnomalySummary>(`/anomalies/${assetId}/summary`)
    return data
  },

  async getModelInfo(assetId: string): Promise<ModelInfo> {
    const { data } = await client.get<ModelInfo>(`/anomalies/${assetId}/model-info`)
    return data
  },
}
