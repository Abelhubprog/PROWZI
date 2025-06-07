package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all the Prometheus metrics for the notification service
type Metrics struct {
	notificationsReceived  prometheus.Counter
	notificationsProcessed prometheus.Counter
	notificationsSent      *prometheus.CounterVec
	notificationsFailed    *prometheus.CounterVec
	notificationDuration   *prometheus.HistogramVec
	queueSize             prometheus.Gauge
	channelHealth         *prometheus.GaugeVec
	templateRenderTime    *prometheus.HistogramVec
	httpRequestsTotal     *prometheus.CounterVec
	httpRequestDuration   *prometheus.HistogramVec
	activeConnections     prometheus.Gauge
	retryAttempts         *prometheus.CounterVec
}

// NewMetrics creates a new Metrics instance with all Prometheus metrics initialized
func NewMetrics() *Metrics {
	return &Metrics{
		notificationsReceived: promauto.NewCounter(prometheus.CounterOpts{
			Name: "prowzi_notifications_received_total",
			Help: "Total number of notifications received",
		}),
		notificationsProcessed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "prowzi_notifications_processed_total",
			Help: "Total number of notifications processed",
		}),
		notificationsSent: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "prowzi_notifications_sent_total",
			Help: "Total number of notifications sent by channel",
		}, []string{"channel"}),
		notificationsFailed: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "prowzi_notifications_failed_total",
			Help: "Total number of failed notifications by channel and reason",
		}, []string{"channel", "reason"}),
		notificationDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "prowzi_notification_duration_seconds",
			Help:    "Duration of notification processing by channel",
			Buckets: prometheus.DefBuckets,
		}, []string{"channel"}),
		queueSize: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "prowzi_queue_size",
			Help: "Current size of the notification queue",
		}),
		channelHealth: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "prowzi_channel_health",
			Help: "Health status of notification channels (1 = healthy, 0 = unhealthy)",
		}, []string{"channel"}),
		templateRenderTime: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "prowzi_template_render_duration_seconds",
			Help:    "Duration of template rendering by template name",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0},
		}, []string{"template"}),
		httpRequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "prowzi_http_requests_total",
			Help: "Total number of HTTP requests by method and status",
		}, []string{"method", "endpoint", "status"}),
		httpRequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "prowzi_http_request_duration_seconds",
			Help:    "Duration of HTTP requests by method and endpoint",
			Buckets: prometheus.DefBuckets,
		}, []string{"method", "endpoint"}),
		activeConnections: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "prowzi_active_connections",
			Help: "Number of active HTTP connections",
		}),
		retryAttempts: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "prowzi_retry_attempts_total",
			Help: "Total number of retry attempts by channel and attempt number",
		}, []string{"channel", "attempt"}),
	}
}

// IncrementNotificationsReceived increments the notifications received counter
func (m *Metrics) IncrementNotificationsReceived() {
	m.notificationsReceived.Inc()
}

// IncrementNotificationsProcessed increments the notifications processed counter
func (m *Metrics) IncrementNotificationsProcessed() {
	m.notificationsProcessed.Inc()
}

// IncrementNotificationsSent increments the notifications sent counter for a specific channel
func (m *Metrics) IncrementNotificationsSent(channel string) {
	m.notificationsSent.WithLabelValues(channel).Inc()
}

// IncrementNotificationsFailed increments the notifications failed counter for a specific channel and reason
func (m *Metrics) IncrementNotificationsFailed(channel, reason string) {
	m.notificationsFailed.WithLabelValues(channel, reason).Inc()
}

// ObserveNotificationDuration records the duration of notification processing for a specific channel
func (m *Metrics) ObserveNotificationDuration(channel string, duration time.Duration) {
	m.notificationDuration.WithLabelValues(channel).Observe(duration.Seconds())
}

// SetQueueSize sets the current queue size
func (m *Metrics) SetQueueSize(size float64) {
	m.queueSize.Set(size)
}

// SetChannelHealth sets the health status of a notification channel
func (m *Metrics) SetChannelHealth(channel string, healthy bool) {
	value := float64(0)
	if healthy {
		value = 1
	}
	m.channelHealth.WithLabelValues(channel).Set(value)
}

// ObserveTemplateRenderTime records the duration of template rendering
func (m *Metrics) ObserveTemplateRenderTime(template string, duration time.Duration) {
	m.templateRenderTime.WithLabelValues(template).Observe(duration.Seconds())
}

// IncrementHTTPRequests increments the HTTP requests counter
func (m *Metrics) IncrementHTTPRequests(method, endpoint, status string) {
	m.httpRequestsTotal.WithLabelValues(method, endpoint, status).Inc()
}

// ObserveHTTPRequestDuration records the duration of HTTP requests
func (m *Metrics) ObserveHTTPRequestDuration(method, endpoint string, duration time.Duration) {
	m.httpRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

// SetActiveConnections sets the number of active HTTP connections
func (m *Metrics) SetActiveConnections(count float64) {
	m.activeConnections.Set(count)
}

// IncrementRetryAttempts increments the retry attempts counter
func (m *Metrics) IncrementRetryAttempts(channel string, attempt int) {
	m.retryAttempts.WithLabelValues(channel, string(rune(attempt))).Inc()
}

// Timer is a utility struct for timing operations
type Timer struct {
	start time.Time
}

// NewTimer creates a new timer
func NewTimer() *Timer {
	return &Timer{start: time.Now()}
}

// ObserveDurationFor observes the duration since the timer was created for a specific metric
func (t *Timer) ObserveDurationFor(observeFunc func(time.Duration)) {
	duration := time.Since(t.start)
	observeFunc(duration)
}

// MetricsMiddleware returns a Gin middleware for recording HTTP metrics
func (m *Metrics) MetricsMiddleware() func(c *gin.Context) {
	return func(c *gin.Context) {
		start := time.Now()
		m.activeConnections.Inc()

		c.Next()

		duration := time.Since(start)
		status := string(rune(c.Writer.Status()))
		
		m.IncrementHTTPRequests(c.Request.Method, c.FullPath(), status)
		m.ObserveHTTPRequestDuration(c.Request.Method, c.FullPath(), duration)
		m.activeConnections.Dec()
	}
}

// ChannelMetrics holds channel-specific metrics
type ChannelMetrics struct {
	metrics *Metrics
	channel string
}

// NewChannelMetrics creates a new ChannelMetrics instance
func (m *Metrics) NewChannelMetrics(channel string) *ChannelMetrics {
	return &ChannelMetrics{
		metrics: m,
		channel: channel,
	}
}

// RecordSuccess records a successful notification send
func (cm *ChannelMetrics) RecordSuccess(duration time.Duration) {
	cm.metrics.IncrementNotificationsSent(cm.channel)
	cm.metrics.ObserveNotificationDuration(cm.channel, duration)
	cm.metrics.SetChannelHealth(cm.channel, true)
}

// RecordFailure records a failed notification send
func (cm *ChannelMetrics) RecordFailure(reason string, duration time.Duration) {
	cm.metrics.IncrementNotificationsFailed(cm.channel, reason)
	cm.metrics.ObserveNotificationDuration(cm.channel, duration)
	cm.metrics.SetChannelHealth(cm.channel, false)
}

// RecordRetry records a retry attempt
func (cm *ChannelMetrics) RecordRetry(attempt int) {
	cm.metrics.IncrementRetryAttempts(cm.channel, attempt)
}

// Stats represents current service statistics
type Stats struct {
	NotificationsReceived  uint64            `json:"notifications_received"`
	NotificationsProcessed uint64            `json:"notifications_processed"`
	NotificationsSent      map[string]uint64 `json:"notifications_sent"`
	NotificationsFailed    map[string]uint64 `json:"notifications_failed"`
	QueueSize             float64           `json:"queue_size"`
	ChannelHealth         map[string]bool   `json:"channel_health"`
	ActiveConnections     float64           `json:"active_connections"`
	Uptime                time.Duration     `json:"uptime"`
}

// GetStats returns current service statistics (this would need to be implemented
// by collecting values from the Prometheus metrics registry in a real implementation)
func (m *Metrics) GetStats() *Stats {
	// Note: In a real implementation, you would collect these values from
	// the Prometheus metrics registry. This is a simplified version.
	return &Stats{
		NotificationsReceived:  0, // Would be collected from registry
		NotificationsProcessed: 0,
		NotificationsSent:      make(map[string]uint64),
		NotificationsFailed:    make(map[string]uint64),
		QueueSize:             0,
		ChannelHealth:         make(map[string]bool),
		ActiveConnections:     0,
		Uptime:                0, // Would be calculated from service start time
	}
}