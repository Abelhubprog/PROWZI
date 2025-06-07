// services/risk/src/var_calculator.rs

use solana_sdk::{
    instruction::Instruction,
    pubkey::Pubkey,
    transaction::VersionedTransaction,
};
use std::sync::Arc;
use ndarray::{Array1, Array2};
use tokio::sync::RwLock;
use statrs::distribution::{Normal, ContinuousCDF};

const CONFIDENCE_LEVELS: &[f64] = &[0.95, 0.99, 0.999];
const HISTORICAL_WINDOW: usize = 252; // One trading year

pub struct VaRCalculator {
    config: VaRConfig,
    historical_data: Arc<HistoricalDataProvider>,
    model_engine: Arc<RiskModelEngine>,
    correlation_engine: Arc<CorrelationEngine>,
    metrics: Arc<RwLock<VaRMetrics>>,
}

impl VaRCalculator {
    pub async fn calculate_portfolio_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<VaRMetrics, VaRError> {
        // Get historical data
        let historical_returns = self.historical_data
            .get_portfolio_returns(portfolio, HISTORICAL_WINDOW)
            .await?;

        // Calculate parametric VaR
        let parametric_var = self.calculate_parametric_var(
            &historical_returns,
            confidence_level,
        )?;

        // Calculate historical VaR
        let historical_var = self.calculate_historical_var(
            &historical_returns,
            confidence_level,
        )?;

        // Calculate Monte Carlo VaR
        let monte_carlo_var = self.calculate_monte_carlo_var(
            portfolio,
            &historical_returns,
            confidence_level,
        ).await?;

        // Combine VaR estimates
        let combined_var = self.combine_var_estimates(
            parametric_var,
            historical_var,
            monte_carlo_var,
        )?;

        Ok(VaRMetrics {
            var_estimate: combined_var,
            confidence_level,
            time_horizon: self.config.time_horizon,
            calculation_method: VaRMethod::Combined,
            component_vars: vec![
                parametric_var,
                historical_var,
                monte_carlo_var,
            ],
        })
    }

    async fn calculate_monte_carlo_var(
        &self,
        portfolio: &Portfolio,
        historical_returns: &Array2<f64>,
        confidence_level: f64,
    ) -> Result<f64, VaRError> {
        // Generate correlation matrix
        let correlations = self.correlation_engine
            .calculate_correlations(historical_returns)?;

        // Generate Cholesky decomposition
        let cholesky = self.calculate_cholesky_decomposition(&correlations)?;

        // Run Monte Carlo simulation
        let simulated_returns = self.run_monte_carlo_simulation(
            portfolio,
            &cholesky,
            self.config.simulation_runs,
        )?;

        // Calculate VaR from simulated returns
        let var = self.calculate_var_from_simulation(
            &simulated_returns,
            confidence_level,
        )?;

        Ok(var)
    }

    fn run_monte_carlo_simulation(
        &self,
        portfolio: &Portfolio,
        cholesky: &Array2<f64>,
        num_simulations: usize,
    ) -> Result<Array1<f64>, VaRError> {
        let mut simulated_returns = Array1::zeros(num_simulations);
        
        for i in 0..num_simulations {
            // Generate random normal variates
            let random_variates = self.generate_random_variates(
                portfolio.positions.len(),
            )?;

            // Apply Cholesky decomposition
            let correlated_returns = cholesky.dot(&random_variates);

            // Calculate portfolio return
            let portfolio_return = self.calculate_portfolio_return(
                portfolio,
                &correlated_returns,
            )?;

            simulated_returns[i] = portfolio_return;
        }

        Ok(simulated_returns)
    }

    fn calculate_var_from_simulation(
        &self,
        returns: &Array1<f64>,
        confidence_level: f64,
    ) -> Result<f64, VaRError> {
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
        Ok(-sorted_returns[index])
    }

    pub async fn calculate_conditional_var(
        &self,
        portfolio: &Portfolio,
        confidence_level: f64,
    ) -> Result<CVaRMetrics, VaRError> {
        // Calculate VaR first
        let var = self.calculate_portfolio_var(
            portfolio,
            confidence_level,
        ).await?;

        // Calculate expected shortfall
        let shortfall = self.calculate_expected_shortfall(
            portfolio,
            var.var_estimate,
            confidence_level,
        ).await?;

        Ok(CVaRMetrics {
            cvar_estimate: shortfall,
            var_metrics: var,
            tail_loss_mean: self.calculate_tail_loss_mean(shortfall)?,
            tail_loss_std: self.calculate_tail_loss_std(shortfall)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_var_calculation() {
        let calculator = VaRCalculator::new(VaRConfig::default()).await.unwrap();
        let portfolio = create_test_portfolio();
        
        for confidence_level in CONFIDENCE_LEVELS {
            let var = calculator.calculate_portfolio_var(&portfolio, *confidence_level)
                .await
                .unwrap();
            
            assert!(var.var_estimate > 0.0);
            assert_eq!(var.confidence_level, *confidence_level);
            assert_eq!(var.component_vars.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_conditional_var() {
        let calculator = VaRCalculator::new(VaRConfig::default()).await.unwrap();
        let portfolio = create_test_portfolio();

        let cvar = calculator.calculate_conditional_var(&portfolio, 0.99)
            .await
            .unwrap();
            
        assert!(cvar.cvar_estimate > cvar.var_metrics.var_estimate);
        assert!(cvar.tail_loss_mean > 0.0);
    }
}