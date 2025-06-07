//! Portfolio management and analysis utilities

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::{Position, RiskError, RiskResult};

/// Portfolio management and analysis
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub id: Uuid,
    pub name: String,
    pub positions: HashMap<Uuid, Position>,
    pub cash_balance: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Portfolio {
    /// Create new portfolio
    pub fn new(name: String, initial_balance: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            positions: HashMap::new(),
            cash_balance: initial_balance,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Add position to portfolio
    pub fn add_position(&mut self, position: Position) -> RiskResult<()> {
        self.positions.insert(position.id, position);
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Remove position from portfolio
    pub fn remove_position(&mut self, position_id: &Uuid) -> RiskResult<Position> {
        self.updated_at = Utc::now();
        self.positions.remove(position_id)
            .ok_or_else(|| RiskError::Position(format!("Position {} not found", position_id)))
    }

    /// Calculate total portfolio value
    pub fn total_value(&self) -> f64 {
        let positions_value: f64 = self.positions.values()
            .map(|pos| pos.current_price * pos.size.abs())
            .sum();
        positions_value + self.cash_balance
    }

    /// Calculate total unrealized P&L
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values()
            .map(|pos| pos.unrealized_pnl)
            .sum()
    }

    /// Calculate portfolio allocation by symbol
    pub fn allocation_by_symbol(&self) -> HashMap<String, f64> {
        let total_value = self.total_value();
        if total_value == 0.0 {
            return HashMap::new();
        }

        let mut allocations = HashMap::new();
        for position in self.positions.values() {
            let position_value = position.current_price * position.size.abs();
            let allocation = position_value / total_value;
            allocations.insert(position.symbol.clone(), allocation);
        }
        allocations
    }

    /// Get largest position by value
    pub fn largest_position(&self) -> Option<&Position> {
        self.positions.values()
            .max_by(|a, b| {
                let a_value = a.current_price * a.size.abs();
                let b_value = b.current_price * b.size.abs();
                a_value.partial_cmp(&b_value).unwrap()
            })
    }

    /// Calculate portfolio concentration risk
    pub fn concentration_risk(&self) -> f64 {
        let allocations = self.allocation_by_symbol();
        let max_allocation = allocations.values()
            .fold(0.0, |max, &allocation| max.max(allocation));
        max_allocation
    }
}

/// Portfolio performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPerformance {
    pub total_return: f64,
    pub daily_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub calmar_ratio: f64,
}

/// Portfolio analytics engine
pub struct PortfolioAnalytics {
    portfolio: Portfolio,
    historical_values: Vec<PortfolioSnapshot>,
}

impl PortfolioAnalytics {
    /// Create new analytics engine
    pub fn new(portfolio: Portfolio) -> Self {
        Self {
            portfolio,
            historical_values: Vec::new(),
        }
    }

    /// Add portfolio snapshot for historical tracking
    pub fn add_snapshot(&mut self, snapshot: PortfolioSnapshot) {
        self.historical_values.push(snapshot);
        
        // Keep only last 252 snapshots (approximately 1 year of daily data)
        if self.historical_values.len() > 252 {
            self.historical_values.remove(0);
        }
    }

    /// Calculate portfolio performance metrics
    pub fn calculate_performance(&self) -> RiskResult<PortfolioPerformance> {
        if self.historical_values.len() < 2 {
            return Err(RiskError::Assessment("Insufficient historical data".to_string()));
        }

        let returns = self.calculate_returns()?;
        let total_return = self.calculate_total_return(&returns);
        let daily_return = self.calculate_average_daily_return(&returns);
        let volatility = self.calculate_volatility(&returns)?;
        let sharpe_ratio = self.calculate_sharpe_ratio(daily_return, volatility);
        let max_drawdown = self.calculate_max_drawdown()?;
        let (win_rate, profit_factor) = self.calculate_win_metrics(&returns);
        let calmar_ratio = self.calculate_calmar_ratio(daily_return, max_drawdown);

        Ok(PortfolioPerformance {
            total_return,
            daily_return,
            volatility,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            calmar_ratio,
        })
    }

    /// Calculate daily returns
    fn calculate_returns(&self) -> RiskResult<Vec<f64>> {
        let mut returns = Vec::new();
        
        for i in 1..self.historical_values.len() {
            let prev_value = self.historical_values[i - 1].total_value;
            let curr_value = self.historical_values[i].total_value;
            
            if prev_value > 0.0 {
                let return_rate = (curr_value - prev_value) / prev_value;
                returns.push(return_rate);
            }
        }
        
        Ok(returns)
    }

    /// Calculate total return
    fn calculate_total_return(&self, returns: &[f64]) -> f64 {
        if self.historical_values.is_empty() {
            return 0.0;
        }
        
        let initial_value = self.historical_values[0].total_value;
        let final_value = self.historical_values.last().unwrap().total_value;
        
        if initial_value > 0.0 {
            (final_value - initial_value) / initial_value
        } else {
            0.0
        }
    }

    /// Calculate average daily return
    fn calculate_average_daily_return(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        returns.iter().sum::<f64>() / returns.len() as f64
    }

    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self, returns: &[f64]) -> RiskResult<f64> {
        if returns.len() < 2 {
            return Ok(0.0);
        }

        let mean = self.calculate_average_daily_return(returns);
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }

    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, daily_return: f64, volatility: f64) -> f64 {
        let risk_free_rate = 0.02 / 252.0; // 2% annual risk-free rate
        
        if volatility > 0.0 {
            (daily_return - risk_free_rate) / volatility
        } else {
            0.0
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self) -> RiskResult<f64> {
        if self.historical_values.is_empty() {
            return Ok(0.0);
        }

        let mut max_drawdown = 0.0;
        let mut peak_value = self.historical_values[0].total_value;

        for snapshot in &self.historical_values {
            if snapshot.total_value > peak_value {
                peak_value = snapshot.total_value;
            }
            
            let drawdown = (peak_value - snapshot.total_value) / peak_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        Ok(max_drawdown)
    }

    /// Calculate win rate and profit factor
    fn calculate_win_metrics(&self, returns: &[f64]) -> (f64, f64) {
        if returns.is_empty() {
            return (0.0, 0.0);
        }

        let winning_trades: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losing_trades: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let win_rate = winning_trades.len() as f64 / returns.len() as f64;
        
        let total_wins: f64 = winning_trades.iter().sum();
        let total_losses: f64 = losing_trades.iter().map(|r| r.abs()).sum();
        
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        (win_rate, profit_factor)
    }

    /// Calculate Calmar ratio
    fn calculate_calmar_ratio(&self, daily_return: f64, max_drawdown: f64) -> f64 {
        let annual_return = daily_return * 252.0; // Annualize daily return
        
        if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        }
    }
}

/// Portfolio snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_value: f64,
    pub cash_balance: f64,
    pub positions_count: usize,
    pub largest_position_size: f64,
    pub concentration_risk: f64,
}

impl From<&Portfolio> for PortfolioSnapshot {
    fn from(portfolio: &Portfolio) -> Self {
        let total_value = portfolio.total_value();
        let largest_position = portfolio.largest_position();
        let largest_position_size = largest_position
            .map(|pos| pos.current_price * pos.size.abs())
            .unwrap_or(0.0);

        Self {
            timestamp: Utc::now(),
            total_value,
            cash_balance: portfolio.cash_balance,
            positions_count: portfolio.positions.len(),
            largest_position_size,
            concentration_risk: portfolio.concentration_risk(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new("Test Portfolio".to_string(), 10000.0);
        assert_eq!(portfolio.name, "Test Portfolio");
        assert_eq!(portfolio.cash_balance, 10000.0);
        assert_eq!(portfolio.total_value(), 10000.0);
    }

    #[test]
    fn test_portfolio_analytics() {
        let portfolio = Portfolio::new("Test".to_string(), 10000.0);
        let mut analytics = PortfolioAnalytics::new(portfolio);
        
        // Add some test snapshots
        analytics.add_snapshot(PortfolioSnapshot {
            timestamp: Utc::now(),
            total_value: 10000.0,
            cash_balance: 10000.0,
            positions_count: 0,
            largest_position_size: 0.0,
            concentration_risk: 0.0,
        });
        
        analytics.add_snapshot(PortfolioSnapshot {
            timestamp: Utc::now(),
            total_value: 10500.0,
            cash_balance: 10500.0,
            positions_count: 0,
            largest_position_size: 0.0,
            concentration_risk: 0.0,
        });
        
        let performance = analytics.calculate_performance().unwrap();
        assert!(performance.total_return > 0.0);
    }
}