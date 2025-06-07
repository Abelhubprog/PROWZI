CREATE TABLE evi_weights (
  id SERIAL PRIMARY KEY,
  domain VARCHAR(50) NOT NULL,
  weights JSONB NOT NULL,
  confidence_score FLOAT NOT NULL,
  canary_percentage INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  active BOOLEAN DEFAULT true
);

CREATE INDEX idx_evi_weights_active ON evi_weights(domain, active) WHERE active = true;

ALTER TABLE feedback ADD COLUMN freshness FLOAT;
ALTER TABLE feedback ADD COLUMN novelty FLOAT;
ALTER TABLE feedback ADD COLUMN impact FLOAT;
ALTER TABLE feedback ADD COLUMN confidence FLOAT;
ALTER TABLE feedback ADD COLUMN gap FLOAT;
