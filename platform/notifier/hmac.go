package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"strconv"
	"time"
)

// HMACValidator provides HMAC signature validation for webhooks
type HMACValidator struct {
	secretKey []byte
	hashFunc  func() hash.Hash
}

// NewHMACValidator creates a new HMAC validator with the given secret key
func NewHMACValidator(secretKey string) *HMACValidator {
	return &HMACValidator{
		secretKey: []byte(secretKey),
		hashFunc:  sha256.New,
	}
}

// GenerateSignature generates an HMAC signature for the given payload
func (hv *HMACValidator) GenerateSignature(payload []byte) string {
	h := hmac.New(hv.hashFunc, hv.secretKey)
	h.Write(payload)
	return hex.EncodeToString(h.Sum(nil))
}

// ValidateSignature validates an HMAC signature against the given payload
func (hv *HMACValidator) ValidateSignature(payload []byte, signature string) bool {
	expectedSignature := hv.GenerateSignature(payload)
	return hmac.Equal([]byte(signature), []byte(expectedSignature))
}

// GenerateTimestampedSignature generates an HMAC signature with timestamp for replay protection
func (hv *HMACValidator) GenerateTimestampedSignature(payload []byte, timestamp time.Time) string {
	timestampStr := strconv.FormatInt(timestamp.Unix(), 10)
	message := append([]byte(timestampStr+"."), payload...)
	
	h := hmac.New(hv.hashFunc, hv.secretKey)
	h.Write(message)
	return fmt.Sprintf("t=%s,v1=%s", timestampStr, hex.EncodeToString(h.Sum(nil)))
}

// ValidateTimestampedSignature validates an HMAC signature with timestamp
func (hv *HMACValidator) ValidateTimestampedSignature(payload []byte, signature string, tolerance time.Duration) error {
	// Parse the signature header (format: "t=<timestamp>,v1=<signature>")
	timestamp, sig, err := parseTimestampedSignature(signature)
	if err != nil {
		return fmt.Errorf("invalid signature format: %w", err)
	}

	// Check timestamp tolerance
	now := time.Now()
	if now.Sub(timestamp) > tolerance {
		return fmt.Errorf("signature timestamp too old")
	}
	
	if timestamp.After(now.Add(5 * time.Minute)) {
		return fmt.Errorf("signature timestamp too far in the future")
	}

	// Validate signature
	timestampStr := strconv.FormatInt(timestamp.Unix(), 10)
	message := append([]byte(timestampStr+"."), payload...)
	
	h := hmac.New(hv.hashFunc, hv.secretKey)
	h.Write(message)
	expectedSignature := hex.EncodeToString(h.Sum(nil))

	if !hmac.Equal([]byte(sig), []byte(expectedSignature)) {
		return fmt.Errorf("signature validation failed")
	}

	return nil
}

// parseTimestampedSignature parses a timestamped signature header
func parseTimestampedSignature(signature string) (time.Time, string, error) {
	var timestamp int64
	var sig string
	
	// Simple parsing of "t=<timestamp>,v1=<signature>" format
	if len(signature) < 10 {
		return time.Time{}, "", fmt.Errorf("signature too short")
	}

	// Find timestamp
	tStart := 2 // Skip "t="
	tEnd := tStart
	for tEnd < len(signature) && signature[tEnd] != ',' {
		tEnd++
	}
	
	if tEnd >= len(signature) {
		return time.Time{}, "", fmt.Errorf("no timestamp found")
	}

	var err error
	timestamp, err = strconv.ParseInt(signature[tStart:tEnd], 10, 64)
	if err != nil {
		return time.Time{}, "", fmt.Errorf("invalid timestamp: %w", err)
	}

	// Find signature
	vStart := tEnd + 1
	for vStart < len(signature) && signature[vStart] != '=' {
		vStart++
	}
	vStart++ // Skip "="
	
	if vStart >= len(signature) {
		return time.Time{}, "", fmt.Errorf("no signature found")
	}

	sig = signature[vStart:]
	
	return time.Unix(timestamp, 0), sig, nil
}

// WebhookSigner provides webhook signing functionality
type WebhookSigner struct {
	validator *HMACValidator
}

// NewWebhookSigner creates a new webhook signer
func NewWebhookSigner(secretKey string) *WebhookSigner {
	return &WebhookSigner{
		validator: NewHMACValidator(secretKey),
	}
}

// SignWebhook signs a webhook payload with timestamp
func (ws *WebhookSigner) SignWebhook(payload []byte) string {
	return ws.validator.GenerateTimestampedSignature(payload, time.Now())
}

// VerifyWebhook verifies a webhook signature
func (ws *WebhookSigner) VerifyWebhook(payload []byte, signature string, tolerance time.Duration) error {
	return ws.validator.ValidateTimestampedSignature(payload, signature, tolerance)
}

// DefaultTolerance is the default tolerance for timestamp validation (5 minutes)
const DefaultTolerance = 5 * time.Minute

// QuickValidation provides a simple interface for webhook validation
func QuickValidation(secretKey string, payload []byte, signature string) bool {
	if secretKey == "" {
		return false
	}
	
	validator := NewHMACValidator(secretKey)
	return validator.ValidateSignature(payload, signature)
}

// QuickTimestampedValidation provides a simple interface for timestamped webhook validation
func QuickTimestampedValidation(secretKey string, payload []byte, signature string) error {
	if secretKey == "" {
		return fmt.Errorf("secret key is required")
	}
	
	validator := NewHMACValidator(secretKey)
	return validator.ValidateTimestampedSignature(payload, signature, DefaultTolerance)
}