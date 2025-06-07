use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::collections::HashMap;

#[derive(Debug, Clone, sqlx::FromRow)]
struct FeedbackRecord {
    brief_id: String,
    user_id: String,
    rating: String, // "positive" or "negative"
    created_at: DateTime<Utc>,
    freshness: f32,
    novelty: f32,
    impact: f32,
    confidence: f32,
    gap: f32,
    domain: String,
}

#[derive(Debug, Clone)]
struct WeightUpdate {
    domain: String,
    weights: EVIWeights,
    confidence_score: f64,
    training_samples: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;

    let nats = async_nats::connect(&std::env::var("NATS_URL")?).await?;

    // Run training
    let updates = train_weights(&pool).await?;

    // Deploy with canary
    for update in updates {
        deploy_canary(&pool, &nats, update).await?;
    }

    Ok(())
}

async fn train_weights(pool: &PgPool) -> Result<Vec<WeightUpdate>, Box<dyn std::error::Error>> {
    let feedback = sqlx::query_as::<_, FeedbackRecord>(
        r#"
        SELECT f.*, b.freshness, b.novelty, b.impact, b.confidence, b.gap, e.domain
        FROM feedback f
        JOIN briefs b ON f.brief_id = b.brief_id
        JOIN events e ON b.event_ids[1] = e.event_id
        WHERE f.created_at > NOW() - INTERVAL '7 days'
        "#
    )
    .fetch_all(pool)
    .await?;

    let mut domain_groups: HashMap<String, Vec<FeedbackRecord>> = HashMap::new();
    for record in feedback {
        domain_groups.entry(record.domain.clone()).or_default().push(record);
    }

    let mut updates = Vec::new();

    for (domain, records) in domain_groups {
        if records.len() < 100 {
            continue; // Skip insufficient data
        }

        let (features, labels) = prepare_training_data(&records);
        let model = train_ridge_regression(features, labels);

        let weights = extract_weights(&model);
        let confidence = calculate_confidence(&records);

        updates.push(WeightUpdate {
            domain,
            weights,
            confidence_score: confidence,
            training_samples: records.len(),
        });
    }

    Ok(updates)
}

fn prepare_training_data(records: &[FeedbackRecord]) -> (Array2<f64>, Array1<f64>) {
    let n = records.len();
    let mut features = Array2::<f64>::zeros((n, 5));
    let mut labels = Array1::<f64>::zeros(n);

    for (i, record) in records.iter().enumerate() {
        // Apply time decay
        let age_days = (Utc::now() - record.created_at).num_days() as f64;
        let decay = 0.9_f64.powf(age_days);

        features[[i, 0]] = record.freshness as f64;
        features[[i, 1]] = record.novelty as f64;
        features[[i, 2]] = record.impact as f64;
        features[[i, 3]] = record.confidence as f64;
        features[[i, 4]] = record.gap as f64;

        labels[i] = match record.rating.as_str() {
            "positive" => 1.0 * decay,
            "negative" => -1.0 * decay,
            _ => 0.0,
        };
    }

    (features, labels)
}

fn train_ridge_regression(features: Array2<f64>, labels: Array1<f64>) -> RidgeRegression<f64, Array1<f64>> {
    let params = RidgeRegressionParameters::default()
        .with_alpha(0.1); // Regularization strength

    RidgeRegression::fit(&features, &labels, params).unwrap()
}

fn extract_weights(model: &RidgeRegression<f64, Array1<f64>>) -> EVIWeights {
    let coef = model.coefficients();

    // Normalize to sum to 1
    let sum: f64 = coef.iter().map(|&x| x.abs()).sum();

    EVIWeights {
        freshness: (coef[0].abs() / sum) as f32,
        novelty: (coef[1].abs() / sum) as f32,
        impact: (coef[2].abs() / sum) as f32,
        confidence: (coef[3].abs() / sum) as f32,
        gap: (coef[4].abs() / sum) as f32,
    }
}

async fn deploy_canary(
    pool: &PgPool,
    nats: &async_nats::Client,
    update: WeightUpdate,
) -> Result<(), Box<dyn std::error::Error>> {
    // Store new weights
    sqlx::query(
        r#"
        INSERT INTO evi_weights (domain, weights, confidence_score, canary_percentage, created_at)
        VALUES ($1, $2, $3, 10, NOW())
        RETURNING id
        "#
    )
    .bind(&update.domain)
    .bind(serde_json::to_value(&update.weights)?)
    .bind(update.confidence_score)
    .fetch_one(pool)
    .await?;

    // Broadcast update
    nats.publish(
        "config.update",
        serde_json::to_vec(&ConfigUpdate {
            component: "evaluator",
            domain: update.domain,
            change: "weights_canary",
        })?.into(),
    ).await?;

    // Monitor for 30 minutes
    tokio::time::sleep(tokio::time::Duration::from_secs(1800)).await;

    // Check metrics
    let metrics = check_canary_metrics(pool, &update.domain).await?;

    if metrics.sn_ratio_drop < 0.05 {
        // Promote to 100%
        sqlx::query(
            "UPDATE evi_weights SET canary_percentage = 100 WHERE domain = $1 AND canary_percentage = 10"
        )
        .bind(&update.domain)
        .execute(pool)
        .await?;

        println!("Promoted weights for domain {} to 100%", update.domain);
    } else {
        // Rollback
        sqlx::query(
            "DELETE FROM evi_weights WHERE domain = $1 AND canary_percentage = 10"
        )
        .bind(&update.domain)
        .execute(pool)
        .await?;

        println!("Rolled back weights for domain {} (S/N drop: {:.2}%)", 
            update.domain, metrics.sn_ratio_drop * 100.0);
    }

    Ok(())
}
