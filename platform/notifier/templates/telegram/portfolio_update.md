💰 **Portfolio Performance Update**

**Total Value:** ${{.total_value}} ({{if gt .total_change_percent 0}}+{{end}}{{.total_change_percent}}%)
**24h PnL:** {{if gt .pnl_24h 0}}+{{end}}${{.pnl_24h}} ({{if gt .pnl_24h_percent 0}}+{{end}}{{.pnl_24h_percent}}%)

{{- if .top_performers }}

**🔥 Top Performers:**
{{- range .top_performers }}
{{- if gt .pnl 0 }}
🟢 {{.symbol}}: +${{.pnl}} (+{{.pnl_percent}}%)
{{- else }}
🔴 {{.symbol}}: ${{.pnl}} ({{.pnl_percent}}%)
{{- end }}
{{- end }}
{{- end }}

{{- if .strategies }}

**🎯 Active Strategies:**
{{- range .strategies }}
• {{.name}}: `{{.status}}`
{{- if .profit }}
  └ Profit: {{if gt .profit 0}}+{{end}}${{.profit}}
{{- end }}
{{- end }}
{{- end }}

{{- if .risk_metrics }}

**📊 Risk Metrics:**
• Portfolio Risk: `{{.risk_metrics.level}}` ({{.risk_metrics.score}}/10)
• Max Drawdown: {{.risk_metrics.max_drawdown}}%
• Sharpe Ratio: {{.risk_metrics.sharpe_ratio}}
{{- if .risk_metrics.var }}
• VaR (24h): ${{.risk_metrics.var}}
{{- end }}
{{- end }}

{{- if .alerts }}

**⚠️ Active Alerts:**
{{- range .alerts }}
• {{.type}}: {{.message}}
{{- end }}
{{- end }}

*Last updated: {{.timestamp}}*