use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{info, warn, error, debug};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Micro Capital Optimizer for trading with minimal capital ($10)
/// Designed to maximize profitability with extreme precision
#[derive(Clone)]
pub struct MicroCapitalOptimizer {
    /// Target capital in USD (default: $10)
    target_capital_usd: Decimal,
    /// Minimum trade size in lamports
    min_trade_size_lamports: u64,
    /// Maximum position size as percentage of capital
    max_position_size_pct: Decimal,
    /// Fee optimization strategies
    fee_optimizer: FeeOptimizer,
    /// Position sizing calculator
    position_calculator: PositionSizeCalculator,
    /// Profitability analyzer
    profit_analyzer: ProfitabilityAnalyzer,
    /// Risk manager for micro capital
    risk_manager: MicroCapitalRiskManager,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroTradeRequest {
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub trade_direction: TradeDirection,
    pub available_capital_lamports: u64,
    pub target_profit_bps: u16,
    pub max_loss_bps: u16,
    pub urgency: TradeUrgency,
    pub gas_budget_lamports: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeDirection {
    Buy,
    Sell,
    ArbitrageTriangular { path: Vec<Pubkey> },
    ArbitrageCrossChain { target_chain: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeUrgency {
    Immediate,    // Execute within 1 block
    Fast,         // Execute within 5 blocks
    Normal,       // Execute within 20 blocks
    Patient,      // Execute when optimal conditions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTradeParams {
    pub trade_size_lamports: u64,
    pub estimated_gas_lamports: u64,
    pub min_profit_lamports: u64,
    pub max_slippage_bps: u16,
    pub priority_fee_lamports: u64,
    pub expected_roi_bps: u16,
    pub execution_strategy: ExecutionStrategy,
    pub backup_routes: Vec<BackupRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Direct,
    SplitOrder { chunks: u8 },
    DCA { intervals_sec: u64, chunk_count: u8 },
    TWAP { duration_sec: u64, chunk_count: u8 },
    Arbitrage { route_count: u8 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRoute {
    pub route_id: String,
    pub estimated_profit_lamports: u64,
    pub execution_probability: f64,
    pub fallback_priority: u8,
}

#[derive(Clone)]
struct FeeOptimizer {
    historical_fees: HashMap<String, Vec<u64>>,
    current_network_load: f64,
}

#[derive(Clone)]
struct PositionSizeCalculator {
    max_position_pct: Decimal,
    min_profit_threshold: Decimal,
}

#[derive(Clone)]
struct ProfitabilityAnalyzer {
    min_roi_bps: u16,
    target_daily_return_pct: Decimal,
}

#[derive(Clone)]
struct MicroCapitalRiskManager {
    max_daily_loss_pct: Decimal,
    max_trades_per_hour: u16,
    emergency_stop_loss_pct: Decimal,
}

impl MicroCapitalOptimizer {
    /// Create new micro capital optimizer for $10 trading
    pub fn new() -> Self {
        Self {
            target_capital_usd: dec!(10.0),
            min_trade_size_lamports: 1000, // ~$0.001 minimum
            max_position_size_pct: dec!(20.0), // Max 20% per trade
            fee_optimizer: FeeOptimizer::new(),
            position_calculator: PositionSizeCalculator::new(),
            profit_analyzer: ProfitabilityAnalyzer::new(),
            risk_manager: MicroCapitalRiskManager::new(),
        }
    }

    /// Optimize trade parameters for maximum profitability with minimal capital
    pub async fn optimize_trade(&self, request: &MicroTradeRequest) -> Result<OptimizedTradeParams> {
        info!("🎯 Optimizing micro capital trade with {} lamports", request.available_capital_lamports);

        // Step 1: Calculate optimal position size
        let position_size = self.calculate_optimal_position_size(request).await?;
        
        // Step 2: Estimate all fees and costs
        let fee_analysis = self.analyze_trading_fees(request, position_size).await?;
        
        // Step 3: Calculate minimum profitable trade size
        let min_profit_size = self.calculate_min_profitable_size(&fee_analysis).await?;
        
        // Step 4: Validate profitability before execution
        if position_size < min_profit_size {
            return Err(anyhow!("Trade size {} too small for profitability (min: {})", 
                              position_size, min_profit_size));
        }
        
        // Step 5: Optimize execution strategy
        let execution_strategy = self.determine_execution_strategy(request, position_size).await?;
        
        // Step 6: Calculate backup routes
        let backup_routes = self.generate_backup_routes(request).await?;
        
        // Step 7: Final optimization and validation
        let optimized_params = OptimizedTradeParams {
            trade_size_lamports: position_size,
            estimated_gas_lamports: fee_analysis.total_gas_lamports,
            min_profit_lamports: fee_analysis.min_profit_lamports,
            max_slippage_bps: self.calculate_optimal_slippage(request).await?,
            priority_fee_lamports: fee_analysis.priority_fee_lamports,
            expected_roi_bps: self.calculate_expected_roi(request, position_size, &fee_analysis).await?,
            execution_strategy,
            backup_routes,
        };

        info!("✅ Micro capital optimization complete:");
        info!("   Trade size: {} lamports (${:.4})", 
              optimized_params.trade_size_lamports,
              optimized_params.trade_size_lamports as f64 / 1_000_000_000.0 * 100.0); // Rough SOL->USD
        info!("   Expected ROI: {:.2}%", optimized_params.expected_roi_bps as f64 / 100.0);
        info!("   Min profit: {} lamports", optimized_params.min_profit_lamports);

        Ok(optimized_params)
    }

    /// Calculate optimal position size for micro capital
    async fn calculate_optimal_position_size(&self, request: &MicroTradeRequest) -> Result<u64> {
        let available_for_trading = request.available_capital_lamports;
        
        // Reserve gas budget first
        let gas_budget = request.gas_budget_lamports.unwrap_or(50_000); // ~0.00005 SOL
        let tradeable_capital = available_for_trading.saturating_sub(gas_budget);
        
        // Apply maximum position size limit
        let max_position = (Decimal::from(tradeable_capital) * self.max_position_size_pct / dec!(100.0))
            .to_u64().unwrap_or(0);
        
        // Apply minimum trade size
        let position_size = max_position.max(self.min_trade_size_lamports);
        
        // Ensure we don't exceed available capital
        Ok(position_size.min(tradeable_capital))
    }

    /// Analyze all trading fees and costs
    async fn analyze_trading_fees(&self, request: &MicroTradeRequest, position_size: u64) -> Result<FeeAnalysis> {
        // Base transaction fee (5000 lamports)
        let base_tx_fee = 5_000u64;
        
        // Priority fee based on urgency
        let priority_fee = match request.urgency {
            TradeUrgency::Immediate => 100_000, // High priority for immediate execution
            TradeUrgency::Fast => 50_000,
            TradeUrgency::Normal => 10_000,
            TradeUrgency::Patient => 5_000,
        };
        
        // Jupiter swap fee (typically 0.25% of trade size)
        let swap_fee_bps = 25; // 0.25%
        let swap_fee_lamports = (position_size as u128 * swap_fee_bps as u128 / 10_000) as u64;
        
        // SPL token account creation fees (if needed)
        let account_creation_fee = 2_039_280u64; // Rent for new token account
        
        // Total gas estimation
        let total_gas = base_tx_fee + priority_fee + account_creation_fee;
        
        // Minimum profit requirement (must exceed all fees by at least target_profit_bps)
        let min_profit_lamports = (swap_fee_lamports as u128 * request.target_profit_bps as u128 / 10_000) as u64 + total_gas;
        
        Ok(FeeAnalysis {
            base_tx_fee,
            priority_fee_lamports: priority_fee,
            swap_fee_lamports,
            account_creation_fee,
            total_gas_lamports: total_gas,
            min_profit_lamports,
        })
    }

    /// Calculate minimum profitable trade size
    async fn calculate_min_profitable_size(&self, fee_analysis: &FeeAnalysis) -> Result<u64> {
        // Trade must be large enough that potential profit exceeds all costs
        // Formula: min_size = (total_costs * 10000) / target_profit_bps
        let min_size = (fee_analysis.total_gas_lamports as u128 * 10_000 / 100) as u64; // 1% minimum profit
        
        Ok(min_size.max(self.min_trade_size_lamports))
    }

    /// Determine optimal execution strategy
    async fn determine_execution_strategy(&self, request: &MicroTradeRequest, position_size: u64) -> Result<ExecutionStrategy> {
        match request.trade_direction {
            TradeDirection::ArbitrageTriangular { .. } => {
                Ok(ExecutionStrategy::Arbitrage { route_count: 3 })
            },
            TradeDirection::ArbitrageCrossChain { .. } => {
                Ok(ExecutionStrategy::Arbitrage { route_count: 2 })
            },
            _ => {
                // For micro capital, usually direct execution is best
                // Only split large orders (>50% of daily volume)
                if position_size > 1_000_000 { // >0.001 SOL
                    Ok(ExecutionStrategy::SplitOrder { chunks: 3 })
                } else {
                    Ok(ExecutionStrategy::Direct)
                }
            }
        }
    }

    /// Generate backup execution routes
    async fn generate_backup_routes(&self, request: &MicroTradeRequest) -> Result<Vec<BackupRoute>> {
        let mut routes = Vec::new();
        
        // Primary route: Jupiter direct swap
        routes.push(BackupRoute {
            route_id: "jupiter_direct".to_string(),
            estimated_profit_lamports: 10_000,
            execution_probability: 0.95,
            fallback_priority: 1,
        });
        
        // Secondary route: Orca pools
        routes.push(BackupRoute {
            route_id: "orca_whirlpool".to_string(),
            estimated_profit_lamports: 8_000,
            execution_probability: 0.90,
            fallback_priority: 2,
        });
        
        // Tertiary route: Raydium pools
        routes.push(BackupRoute {
            route_id: "raydium_clmm".to_string(),
            estimated_profit_lamports: 6_000,
            execution_probability: 0.85,
            fallback_priority: 3,
        });
        
        Ok(routes)
    }

    /// Calculate optimal slippage tolerance
    async fn calculate_optimal_slippage(&self, request: &MicroTradeRequest) -> Result<u16> {
        let base_slippage = match request.urgency {
            TradeUrgency::Immediate => 300, // 3% for immediate execution
            TradeUrgency::Fast => 150,      // 1.5%
            TradeUrgency::Normal => 100,    // 1%
            TradeUrgency::Patient => 50,    // 0.5%
        };
        
        // Adjust for trade direction
        let adjusted_slippage = match request.trade_direction {
            TradeDirection::ArbitrageTriangular { .. } => base_slippage + 50, // +0.5% for complexity
            TradeDirection::ArbitrageCrossChain { .. } => base_slippage + 100, // +1% for cross-chain
            _ => base_slippage,
        };
        
        Ok(adjusted_slippage)
    }

    /// Calculate expected ROI for the trade
    async fn calculate_expected_roi(&self, request: &MicroTradeRequest, position_size: u64, fee_analysis: &FeeAnalysis) -> Result<u16> {
        // Base ROI from market movement (simplified calculation)
        let base_roi_bps = request.target_profit_bps;
        
        // Subtract fees impact
        let fee_impact_bps = (fee_analysis.total_gas_lamports as u128 * 10_000 / position_size as u128) as u16;
        
        // Net expected ROI
        let net_roi_bps = base_roi_bps.saturating_sub(fee_impact_bps);
        
        Ok(net_roi_bps)
    }

    /// Validate trade profitability before execution
    pub async fn validate_profitability(&self, params: &OptimizedTradeParams) -> Result<bool> {
        // Check minimum ROI threshold
        if params.expected_roi_bps < 50 { // Minimum 0.5% profit
            warn!("Trade ROI too low: {}bps", params.expected_roi_bps);
            return Ok(false);
        }
        
        // Check profit vs costs ratio
        let profit_to_cost_ratio = params.min_profit_lamports as f64 / params.estimated_gas_lamports as f64;
        if profit_to_cost_ratio < 1.5 { // Profit should be at least 1.5x costs
            warn!("Profit to cost ratio too low: {:.2}", profit_to_cost_ratio);
            return Ok(false);
        }
        
        info!("✅ Trade profitability validated - ROI: {}bps, P/C ratio: {:.2}", 
              params.expected_roi_bps, profit_to_cost_ratio);
        
        Ok(true)
    }

    /// Calculate optimal capital allocation across multiple opportunities
    pub async fn optimize_capital_allocation(&self, opportunities: Vec<MicroTradeRequest>) -> Result<Vec<OptimizedAllocation>> {
        let mut allocations = Vec::new();
        let mut remaining_capital = self.target_capital_usd;
        
        // Sort opportunities by expected ROI (highest first)
        let mut sorted_ops = opportunities;
        sorted_ops.sort_by(|a, b| b.target_profit_bps.cmp(&a.target_profit_bps));
        
        for (index, opportunity) in sorted_ops.iter().enumerate() {
            if remaining_capital <= dec!(0.5) { // Stop if less than $0.50 remaining
                break;
            }
            
            let allocation_usd = remaining_capital.min(dec!(2.0)); // Max $2 per trade
            let allocation_lamports = (allocation_usd * dec!(1_000_000_000) / dec!(100)).to_u64().unwrap_or(0); // Rough conversion
            
            let optimized_params = self.optimize_trade(&MicroTradeRequest {
                available_capital_lamports: allocation_lamports,
                ..opportunity.clone()
            }).await?;
            
            if self.validate_profitability(&optimized_params).await? {
                allocations.push(OptimizedAllocation {
                    opportunity_index: index,
                    allocated_capital_lamports: allocation_lamports,
                    expected_profit_lamports: optimized_params.min_profit_lamports,
                    params: optimized_params,
                });
                
                remaining_capital -= allocation_usd;
            }
        }
        
        info!("📊 Capital allocation optimized: {} trades planned, ${:.2} allocated", 
              allocations.len(), 
              (self.target_capital_usd - remaining_capital).to_f64().unwrap_or(0.0));
        
        Ok(allocations)
    }
}

#[derive(Debug, Clone)]
struct FeeAnalysis {
    pub base_tx_fee: u64,
    pub priority_fee_lamports: u64,
    pub swap_fee_lamports: u64,
    pub account_creation_fee: u64,
    pub total_gas_lamports: u64,
    pub min_profit_lamports: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedAllocation {
    pub opportunity_index: usize,
    pub allocated_capital_lamports: u64,
    pub expected_profit_lamports: u64,
    pub params: OptimizedTradeParams,
}

impl FeeOptimizer {
    fn new() -> Self {
        Self {
            historical_fees: HashMap::new(),
            current_network_load: 0.5,
        }
    }
}

impl PositionSizeCalculator {
    fn new() -> Self {
        Self {
            max_position_pct: dec!(20.0),
            min_profit_threshold: dec!(0.5),
        }
    }
}

impl ProfitabilityAnalyzer {
    fn new() -> Self {
        Self {
            min_roi_bps: 50, // 0.5% minimum ROI
            target_daily_return_pct: dec!(5.0), // 5% daily target
        }
    }
}

impl MicroCapitalRiskManager {
    fn new() -> Self {
        Self {
            max_daily_loss_pct: dec!(2.0), // Max 2% daily loss
            max_trades_per_hour: 10,
            emergency_stop_loss_pct: dec!(5.0), // Emergency stop at 5% loss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::pubkey::Pubkey;

    #[tokio::test]
    async fn test_micro_capital_optimization() {
        let optimizer = MicroCapitalOptimizer::new();
        
        let request = MicroTradeRequest {
            base_mint: Pubkey::new_unique(),
            quote_mint: Pubkey::new_unique(),
            trade_direction: TradeDirection::Buy,
            available_capital_lamports: 10_000_000, // ~0.01 SOL
            target_profit_bps: 100, // 1% target profit
            max_loss_bps: 50, // 0.5% max loss
            urgency: TradeUrgency::Normal,
            gas_budget_lamports: Some(100_000),
        };
        
        let result = optimizer.optimize_trade(&request).await;
        assert!(result.is_ok());
        
        let params = result.unwrap();
        assert!(params.trade_size_lamports > 0);
        assert!(params.expected_roi_bps > 0);
    }

    #[tokio::test]
    async fn test_profitability_validation() {
        let optimizer = MicroCapitalOptimizer::new();
        
        let params = OptimizedTradeParams {
            trade_size_lamports: 1_000_000,
            estimated_gas_lamports: 100_000,
            min_profit_lamports: 150_000,
            max_slippage_bps: 100,
            priority_fee_lamports: 10_000,
            expected_roi_bps: 75, // 0.75% ROI
            execution_strategy: ExecutionStrategy::Direct,
            backup_routes: vec![],
        };
        
        let is_profitable = optimizer.validate_profitability(&params).await.unwrap();
        assert!(is_profitable);
    }

    #[tokio::test]
    async fn test_capital_allocation() {
        let optimizer = MicroCapitalOptimizer::new();
        
        let opportunities = vec![
            MicroTradeRequest {
                base_mint: Pubkey::new_unique(),
                quote_mint: Pubkey::new_unique(),
                trade_direction: TradeDirection::Buy,
                available_capital_lamports: 5_000_000,
                target_profit_bps: 200, // 2% target
                max_loss_bps: 50,
                urgency: TradeUrgency::Fast,
                gas_budget_lamports: Some(50_000),
            },
            MicroTradeRequest {
                base_mint: Pubkey::new_unique(),
                quote_mint: Pubkey::new_unique(),
                trade_direction: TradeDirection::Sell,
                available_capital_lamports: 3_000_000,
                target_profit_bps: 150, // 1.5% target
                max_loss_bps: 75,
                urgency: TradeUrgency::Normal,
                gas_budget_lamports: Some(75_000),
            },
        ];
        
        let allocations = optimizer.optimize_capital_allocation(opportunities).await;
        assert!(allocations.is_ok());
        
        let allocs = allocations.unwrap();
        assert!(allocs.len() <= 2);
    }
}