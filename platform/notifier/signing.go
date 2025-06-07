package signing

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "time"
)

type WebhookSigner struct {
    secrets map[string]string // channelID -> secret
}

func NewWebhookSigner(secrets map[string]string) *WebhookSigner {
    return &WebhookSigner{secrets: secrets}
}

func (w *WebhookSigner) Sign(channelID string, payload []byte) (string, error) {
    secret, ok := w.secrets[channelID]
    if !ok {
        return "", fmt.Errorf("no secret for channel %s", channelID)
    }

    // Include timestamp to prevent replay
    timestamp := fmt.Sprintf("%d", time.Now().Unix())
    message := fmt.Sprintf("%s.%s", timestamp, string(payload))

    h := hmac.New(sha256.New, []byte(secret))
    h.Write([]byte(message))
    signature := hex.EncodeToString(h.Sum(nil))

    // Format: t=timestamp,v1=signature
    return fmt.Sprintf("t=%s,v1=%s", timestamp, signature), nil
}

// Client verification
func VerifyWebhook(secret, header string, payload []byte) bool {
    parts := parseHeader(header)
    timestamp := parts["t"]
    signature := parts["v1"]

    if timestamp == "" || signature == "" {
        return false
    }

    // Check timestamp (5 minute window)
    ts, _ := strconv.ParseInt(timestamp, 10, 64)
    if time.Now().Unix()-ts > 300 {
        return false
    }

    // Verify signature
    expected := computeSignature(secret, timestamp, payload)
    return hmac.Equal([]byte(signature), []byte(expected))
}
