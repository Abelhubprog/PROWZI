🎯 **Mission Status Update**

**Mission:** {{.mission.name}}
**Status:** {{if eq .mission.status "active"}}🟢 Active{{else if eq .mission.status "paused"}}🟡 Paused{{else if eq .mission.status "completed"}}✅ Completed{{else if eq .mission.status "failed"}}🔴 Failed{{else}}⚪ {{.mission.status}}{{end}}

{{- if .mission.progress }}
**Progress:** {{.mission.progress}}% ({{.mission.completed_tasks}}/{{.mission.total_tasks}} tasks)
{{- end }}

{{- if .mission.profit }}
**Performance:**
• Profit: {{if gt .mission.profit 0}}+{{end}}${{.mission.profit}}
{{- if .mission.roi }}
• ROI: {{if gt .mission.roi 0}}+{{end}}{{.mission.roi}}%
{{- end }}
{{- if .mission.win_rate }}
• Win Rate: {{.mission.win_rate}}%
{{- end }}
{{- end }}

{{- if .mission.risk }}
**Risk Level:** {{.mission.risk.level}} ({{.mission.risk.score}}/10)
{{- if .mission.risk.exposure }}
• Exposure: ${{.mission.risk.exposure}}
{{- end }}
{{- if .mission.risk.max_loss }}
• Max Loss: ${{.mission.risk.max_loss}}
{{- end }}
{{- end }}

{{- if .mission.details }}
{{.mission.details}}
{{- end }}

{{- if .mission.next_action }}

**Next Action:** {{.mission.next_action}}
{{- end }}

{{- if .mission.estimated_completion }}
**Est. Completion:** {{.mission.estimated_completion}}
{{- end }}

*Mission ID: {{.mission.id}}*
*Updated: {{.timestamp}}*