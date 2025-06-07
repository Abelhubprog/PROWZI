{{- if eq .severity "critical" -}}
🚨 *CRITICAL TRADING ALERT* 🚨
{{- else if eq .severity "warning" -}}
⚠️ *Trading Alert* ⚠️
{{- else if eq .severity "success" -}}
✅ *Trading Success* ✅
{{- else -}}
ℹ️ *Trading Update* ℹ️
{{- end }}

**{{.title}}**

{{.description}}

{{- if .position }}

**Position Details:**
• Symbol: `{{.position.symbol}}`
• Size: {{.position.size}}
• Entry: ${{.position.entry_price}}
{{- if .position.current_price }}
• Current: ${{.position.current_price}}
{{- end }}
{{- if .position.pnl }}
• PnL: {{if gt .position.pnl 0}}+{{end}}${{.position.pnl}} ({{if gt .position.pnl_percent 0}}+{{end}}{{.position.pnl_percent}}%)
{{- end }}
{{- end }}

{{- if .risk }}

**Risk Metrics:**
• Portfolio Risk: {{.risk.level}} ({{.risk.score}}/10)
{{- if .risk.max_drawdown }}
• Max Drawdown: {{.risk.max_drawdown}}%
{{- end }}
{{- if .risk.var }}
• VaR (24h): ${{.risk.var}}
{{- end }}
{{- end }}

{{- if .action }}

**Recommended Action:**
{{.action}}
{{- end }}

{{- if .timestamp }}
*Generated at {{.timestamp}}*
{{- end }}