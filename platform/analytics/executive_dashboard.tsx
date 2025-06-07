'use client'

import { useState, useEffect } from 'react'
import { Line, Bar, Scatter, Doughnut } from 'react-chartjs-2'
import { motion } from 'framer-motion'

export function ExecutiveDashboard() {
  const [metrics, setMetrics] = useState<BusinessMetrics | null>(null)
  const [dateRange, setDateRange] = useState('30d')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchMetrics()
  }, [dateRange])

  const fetchMetrics = async () => {
    setIsLoading(true)
    const data = await fetch(`/api/analytics/executive?range=${dateRange}`)
    const metrics = await data.json()
    setMetrics(metrics)
    setIsLoading(false)
  }

  if (isLoading) return <LoadingState />
  if (!metrics) return null

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="MRR"
          value={`$${metrics.mrr.toLocaleString()}`}
          change={metrics.mrrGrowth}
          icon="💰"
        />
        <MetricCard
          title="Active Users"
          value={metrics.activeUsers.toLocaleString()}
          change={metrics.userGrowth}
          icon="👥"
        />
        <MetricCard
          title="Churn Rate"
          value={`${metrics.churnRate.toFixed(1)}%`}
          change={-metrics.churnDelta}
          inverse
          icon="📉"
        />
        <MetricCard
          title="LTV:CAC"
          value={`${metrics.ltvCacRatio.toFixed(1)}x`}
          change={metrics.ltvCacDelta}
          icon="📊"
        />
      </div>

      {/* Revenue Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <Card title="Revenue Breakdown">
          <Doughnut
            data={{
              labels: metrics.revenueByTier.map(t => t.tier),
              datasets: [{
                data: metrics.revenueByTier.map(t => t.amount),
                backgroundColor: [
                  '#3B82F6', // Free
                  '#10B981', // Pro
                  '#F59E0B', // Elite
                  '#EF4444', // Enterprise
                ],
              }]
            }}
            options={{
              plugins: {
                legend: { position: 'bottom' }
              }
            }}
          />
        </Card>

        <Card title="Growth Metrics">
          <Line
            data={{
              labels: metrics.growthTrend.map(d => d.date),
              datasets: [
                {
                  label: 'MRR',
                  data: metrics.growthTrend.map(d => d.mrr),
                  borderColor: '#3B82F6',
                  tension: 0.4,
                },
                {
                  label: 'Users',
                  data: metrics.growthTrend.map(d => d.users),
                  borderColor: '#10B981',
                  tension: 0.4,
                  yAxisID: 'y1',
                }
              ]
            }}
            options={{
              scales: {
                y: {
                  type: 'linear',
                  display: true,
                  position: 'left',
                },
                y1: {
                  type: 'linear',
                  display: true,
                  position: 'right',
                  grid: { drawOnChartArea: false },
                }
              }
            }}
          />
        </Card>
      </div>

      {/* Cohort Analysis */}
      <Card title="Cohort Retention Analysis" className="mb-8">
        <CohortChart data={metrics.cohortData} />
      </Card>

      {/* Product Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <Card title="Feature Adoption">
          <Bar
            data={{
              labels: metrics.featureAdoption.map(f => f.feature),
              datasets: [{
                label: 'Adoption Rate',
                data: metrics.featureAdoption.map(f => f.adoptionRate),
                backgroundColor: '#3B82F6',
              }]
            }}
            options={{
              indexAxis: 'y',
              scales: {
                x: {
                  beginAtZero: true,
                  max: 100,
                }
              }
            }}
          />
        </Card>

        <Card title="Mission Performance">
          <Scatter
            data={{
              datasets: [{
                label: 'Missions',
                data: metrics.missionPerformance.map(m => ({
                  x: m.duration,
                  y: m.briefsGenerated,
                })),
                backgroundColor: '#10B981',
              }]
            }}
            options={{
              scales: {
                x: { title: { display: true, text: 'Duration (hours)' } },
                y: { title: { display: true, text: 'Briefs Generated' } },
              }
            }}
          />
        </Card>

        <Card title="API Usage">
          <Line
            data={{
              labels: metrics.apiUsage.map(d => d.hour),
              datasets: [{
                label: 'API Calls',
                data: metrics.apiUsage.map(d => d.calls),
                borderColor: '#F59E0B',
                fill: true,
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
              }]
            }}
          />
        </Card>
      </div>

      {/* Predictive Analytics */}
      <Card title="Revenue Forecast" className="mb-8">
        <RevenueForcast data={metrics.forecast} />
      </Card>

      {/* Action Items */}
      <Card title="AI-Generated Insights & Actions">
        <div className="space-y-4">
          {metrics.insights.map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </Card>
    </div>
  )
}

function MetricCard({ title, value, change, icon, inverse = false }) {
  const isPositive = inverse ? change < 0 : change > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900 rounded-lg p-6 border border-gray-800"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        <span className="text-2xl">{icon}</span>
      </div>
      <div className="text-3xl font-bold mb-2">{value}</div>
      <div className={`flex items-center text-sm ${
        isPositive ? 'text-green-500' : 'text-red-500'
      }`}>
        <span>{isPositive ? '↑' : '↓'}</span>
        <span className="ml-1">{Math.abs(change).toFixed(1)}%</span>
      </div>
    </motion.div>
  )
}

function CohortChart({ data }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left p-2">Cohort</th>
            {data.periods.map(period => (
              <th key={period} className="text-center p-2">{period}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.cohorts.map(cohort => (
            <tr key={cohort.month} className="border-t border-gray-800">
              <td className="p-2 font-medium">{cohort.month}</td>
              {cohort.retention.map((value, i) => (
                <td
                  key={i}
                  className="p-2 text-center"
                  style={{
                    backgroundColor: `rgba(59, 130, 246, ${value / 100})`,
                  }}
                >
                  {value}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
