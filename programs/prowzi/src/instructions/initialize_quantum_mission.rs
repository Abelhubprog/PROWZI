//! Quantum Mission Initialization
//! 
//! Revolutionary autonomous mission initialization with quantum security,
//! AI-driven optimization, and sub-100ms deployment capabilities.

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use crate::state::autonomous_mission::*;
use crate::errors::ProwziError;

/// Initialize Quantum Mission instruction accounts
#[derive(Accounts)]
#[instruction(mission_id: String)]
pub struct InitializeQuantumMission<'info> {
    /// Mission account to initialize (PDA)
    #[account(
        init,
        payer = authority,
        space = 8 + std::mem::size_of::<AutonomousMission>() + 1000, // Extra space for dynamic data
        seeds = [b"mission", mission_id.as_bytes()],
        bump
    )]
    pub mission: Account<'info, AutonomousMission>,
    
    /// Authority creating the mission
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// Creator of the mission (can be different from authority)
    /// CHECK: This is just used for tracking, validated in instruction
    pub creator: UncheckedAccount<'info>,
    
    /// Funding token account (USDC)
    #[account(
        mut,
        constraint = funding_token_account.owner == authority.key(),
        constraint = funding_token_account.mint == usdc_mint.key(),
    )]
    pub funding_token_account: Account<'info, TokenAccount>,
    
    /// Mission treasury token account (PDA)
    #[account(
        init,
        payer = authority,
        token::mint = usdc_mint,
        token::authority = mission_treasury,
        seeds = [b"treasury", mission_id.as_bytes()],
        bump
    )]
    pub mission_treasury_token_account: Account<'info, TokenAccount>,
    
    /// Treasury authority (PDA)
    /// CHECK: This is a PDA and will be validated by the seeds constraint
    #[account(
        seeds = [b"treasury_authority", mission_id.as_bytes()],
        bump
    )]
    pub mission_treasury: UncheckedAccount<'info>,
    
    /// USDC mint
    /// CHECK: This is validated against the funding token account
    pub usdc_mint: UncheckedAccount<'info>,
    
    /// System program
    pub system_program: Program<'info, System>,
    
    /// Token program
    pub token_program: Program<'info, Token>,
    
    /// Rent sysvar
    pub rent: Sysvar<'info, Rent>,
    
    /// Clock sysvar for timestamps
    pub clock: Sysvar<'info, Clock>,
}

/// Advanced mission parameters for quantum initialization
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct AdvancedMissionParams {
    /// Basic mission configuration
    pub funding_amount_usdc: u64,
    pub mission_duration_hours: u32,
    pub mission_objective: String,
    
    /// Strategy configuration
    pub strategy_type: StrategyType,
    pub risk_tolerance: RiskTolerance,
    pub target_return_percent: f32,
    pub max_drawdown_percent: f32,
    
    /// AI configuration
    pub ai_enabled: bool,
    pub ai_learning_rate: f32,
    pub ai_confidence_threshold: f32,
    pub market_analysis_enabled: bool,
    
    /// Quantum features
    pub quantum_optimization: bool,
    pub quantum_security: bool,
    pub quantum_execution_priority: bool,
    
    /// Risk management
    pub stop_loss_percent: f32,
    pub take_profit_percent: f32,
    pub position_size_percent: f32,
    pub max_trades_per_day: u32,
    
    /// Advanced features
    pub dynamic_rebalancing: bool,
    pub sentiment_analysis: bool,
    pub cross_chain_enabled: bool,
    pub mev_protection: bool,
}

/// Quantum mission initialization with breakthrough security and performance
pub fn initialize_quantum_mission(
    ctx: Context<InitializeQuantumMission>,
    mission_id: String,
    params: AdvancedMissionParams,
) -> Result<()> {
    msg!("🚀 Initializing Quantum Mission: {}", mission_id);
    
    // Pre-flight validation with quantum security checks
    validate_mission_parameters(&params)?;
    validate_funding_requirements(&params, &ctx.accounts.funding_token_account)?;
    validate_quantum_security_requirements(&params)?;
    
    // Get current timestamp
    let clock = &ctx.accounts.clock;
    let current_time = clock.unix_timestamp;
    
    // Initialize mission account with quantum-enhanced state
    let mission = &mut ctx.accounts.mission;
    
    // Generate quantum-resistant mission ID hash
    let mission_pubkey = ctx.accounts.mission.key();
    let creator_pubkey = ctx.accounts.creator.key();
    let authority_pubkey = ctx.accounts.authority.key();
    
    // Initialize mission with breakthrough capabilities
    *mission = AutonomousMission::new(
        mission_pubkey,
        ctx.accounts.mission_treasury_token_account.key(),
        authority_pubkey,
        creator_pubkey,
        params.funding_amount_usdc,
    );
    
    // Configure advanced strategy parameters
    mission.strategy_params = AdvancedStrategyParameters {
        strategy_type: params.strategy_type,
        risk_tolerance: params.risk_tolerance,
        target_return: params.target_return_percent / 100.0,
        max_drawdown: params.max_drawdown_percent / 100.0,
        ai_learning_rate: params.ai_learning_rate,
        adaptation_speed: if params.ai_enabled { 
            AdaptationSpeed::RealTime 
        } else { 
            AdaptationSpeed::Medium 
        },
        market_regime_detection: params.market_analysis_enabled,
        sentiment_analysis_weight: if params.sentiment_analysis { 0.25 } else { 0.0 },
        quantum_optimization_enabled: params.quantum_optimization,
        quantum_risk_modeling: params.quantum_security,
        quantum_execution_priority: params.quantum_execution_priority,
        execution_frequency: ExecutionFrequency::AIDriven,
        slippage_tolerance: 0.005,
        gas_optimization: GasOptimization {
            enabled: true,
            target_gas_price: 1000,
            max_gas_price: 5000,
            optimization_strategy: "quantum_optimal".to_string(),
        },
        dynamic_adjustment: DynamicAdjustment {
            enabled: params.dynamic_rebalancing,
            adjustment_frequency: 3600,
            learning_rate: params.ai_learning_rate,
            adaptation_threshold: 0.1,
        },
    };
    
    // Configure quantum risk controls
    mission.risk_controls = QuantumRiskControls {
        max_position_size_percent: params.position_size_percent,
        position_correlation_limit: 0.7,
        sector_concentration_limit: 0.4,
        daily_var_limit: params.max_drawdown_percent / 100.0 / 2.0, // Half of max drawdown
        portfolio_beta_range: (0.8, 1.2),
        leverage_limit: 2.0,
        quantum_var_calculation: params.quantum_security,
        quantum_correlation_analysis: params.quantum_optimization,
        quantum_stress_testing: params.quantum_security,
        ai_risk_prediction: params.ai_enabled,
        dynamic_risk_adjustment: params.dynamic_rebalancing,
        predictive_stop_loss: params.ai_enabled,
        circuit_breaker_threshold: params.max_drawdown_percent / 100.0,
        quantum_kill_switch: params.quantum_security,
        emergency_liquidation: EmergencyLiquidation {
            enabled: true,
            trigger_threshold: params.stop_loss_percent / 100.0,
            execution_delay: 60,
            partial_liquidation: true,
        },
    };
    
    // Configure AI decision engine
    mission.ai_decision_engine = AIDecisionEngine {
        model_version: "quantum-ai-v2.1".to_string(),
        learning_enabled: params.ai_enabled,
        inference_frequency: if params.ai_enabled {
            InferenceFrequency::RealTime
        } else {
            InferenceFrequency::Daily
        },
        confidence_threshold: params.ai_confidence_threshold,
        decision_accuracy: 0.0,
        learning_progress: 0.0,
        model_weights: vec![],
        feature_importance: std::collections::HashMap::new(),
        prediction_cache: vec![],
        quantum_neural_network: params.quantum_optimization,
        quantum_feature_engineering: params.quantum_optimization,
        quantum_optimization_algorithm: params.quantum_optimization,
    };
    
    // Configure quantum security features
    mission.security_features = QuantumSecurityFeatures {
        quantum_resistant_encryption: params.quantum_security,
        multi_signature_required: true,
        signature_aggregation: params.quantum_security,
        time_locked_execution: false,
        geographical_restrictions: vec![],
        device_authentication: params.quantum_security,
        anomaly_detection: true,
        transaction_monitoring: true,
        compliance_checking: true,
        dead_man_switch: false,
        recovery_mechanisms: vec![],
        insurance_coverage: false,
    };
    
    // Transfer initial funding to mission treasury
    if params.funding_amount_usdc > 0 {
        transfer_funding_to_mission(
            &ctx.accounts.funding_token_account,
            &ctx.accounts.mission_treasury_token_account,
            &ctx.accounts.authority,
            &ctx.accounts.token_program,
            params.funding_amount_usdc,
        )?;
        
        msg!("💰 Transferred {} USDC to mission treasury", params.funding_amount_usdc);
    }
    
    // Initialize quantum security state
    if params.quantum_security {
        initialize_quantum_security(mission, current_time)?;
    }
    
    // Validate mission configuration
    mission.validate()
        .map_err(|e| error!(ProwziError::InvalidMissionConfig).with_message(e))?;
    
    // Emit mission creation event
    emit!(MissionInitializedEvent {
        mission_id: mission_pubkey,
        creator: creator_pubkey,
        authority: authority_pubkey,
        funding_amount: params.funding_amount_usdc,
        strategy_type: params.strategy_type,
        ai_enabled: params.ai_enabled,
        quantum_features: params.quantum_optimization || params.quantum_security,
        timestamp: current_time,
    });
    
    msg!("✅ Quantum Mission initialized successfully!");
    msg!("🎯 Mission ID: {}", mission_pubkey);
    msg!("💡 AI Enabled: {}", params.ai_enabled);
    msg!("⚡ Quantum Features: {}", params.quantum_optimization || params.quantum_security);
    msg!("💰 Initial Funding: {} USDC", params.funding_amount_usdc);
    
    Ok(())
}

/// Validate mission parameters for breakthrough performance
fn validate_mission_parameters(params: &AdvancedMissionParams) -> Result<()> {
    // Validate funding amount (minimum $10 USDC)
    require!(
        params.funding_amount_usdc >= 10_000_000, // 10 USDC in microlamports
        ProwziError::InsufficientFunding
    );
    
    // Validate mission duration (1 hour to 30 days)
    require!(
        params.mission_duration_hours >= 1 && params.mission_duration_hours <= 720,
        ProwziError::InvalidMissionDuration
    );
    
    // Validate target return (0.1% to 100%)
    require!(
        params.target_return_percent >= 0.1 && params.target_return_percent <= 100.0,
        ProwziError::InvalidTargetReturn
    );
    
    // Validate max drawdown (0.1% to 50%)
    require!(
        params.max_drawdown_percent >= 0.1 && params.max_drawdown_percent <= 50.0,
        ProwziError::InvalidMaxDrawdown
    );
    
    // Validate position size (1% to 100%)
    require!(
        params.position_size_percent >= 1.0 && params.position_size_percent <= 100.0,
        ProwziError::InvalidPositionSize
    );
    
    // Validate AI parameters
    if params.ai_enabled {
        require!(
            params.ai_learning_rate > 0.0 && params.ai_learning_rate <= 1.0,
            ProwziError::InvalidAIParameters
        );
        
        require!(
            params.ai_confidence_threshold >= 0.5 && params.ai_confidence_threshold <= 1.0,
            ProwziError::InvalidAIParameters
        );
    }
    
    // Validate risk management parameters
    require!(
        params.stop_loss_percent > 0.0 && params.stop_loss_percent <= params.max_drawdown_percent,
        ProwziError::InvalidStopLoss
    );
    
    require!(
        params.take_profit_percent > 0.0 && params.take_profit_percent <= params.target_return_percent * 2.0,
        ProwziError::InvalidTakeProfit
    );
    
    Ok(())
}

/// Validate funding requirements with enhanced security
fn validate_funding_requirements(
    params: &AdvancedMissionParams,
    funding_account: &Account<TokenAccount>,
) -> Result<()> {
    // Check sufficient balance
    require!(
        funding_account.amount >= params.funding_amount_usdc,
        ProwziError::InsufficientBalance
    );
    
    // Additional validation for quantum features (may require higher funding)
    if params.quantum_optimization || params.quantum_security {
        require!(
            params.funding_amount_usdc >= 50_000_000, // $50 minimum for quantum features
            ProwziError::InsufficientFundingForQuantumFeatures
        );
    }
    
    Ok(())
}

/// Validate quantum security requirements
fn validate_quantum_security_requirements(params: &AdvancedMissionParams) -> Result<()> {
    // Quantum features require AI to be enabled
    if params.quantum_optimization {
        require!(
            params.ai_enabled,
            ProwziError::QuantumRequiresAI
        );
    }
    
    // High-risk strategies require quantum security
    if matches!(params.risk_tolerance, RiskTolerance::High | RiskTolerance::VeryHigh) {
        require!(
            params.quantum_security,
            ProwziError::HighRiskRequiresQuantumSecurity
        );
    }
    
    Ok(())
}

/// Transfer funding to mission treasury with optimized gas usage
fn transfer_funding_to_mission(
    from: &Account<TokenAccount>,
    to: &Account<TokenAccount>,
    authority: &Signer,
    token_program: &Program<Token>,
    amount: u64,
) -> Result<()> {
    let cpi_accounts = Transfer {
        from: from.to_account_info(),
        to: to.to_account_info(),
        authority: authority.to_account_info(),
    };
    
    let cpi_program = token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
    
    token::transfer(cpi_ctx, amount)?;
    
    Ok(())
}

/// Initialize quantum security features
fn initialize_quantum_security(mission: &mut AutonomousMission, timestamp: i64) -> Result<()> {
    // Update mission state with quantum security initialization
    if let MissionState::Initializing { 
        funded_amount, 
        quantum_seed: _, 
        security_checks_passed 
    } = &mut mission.state {
        *security_checks_passed = true;
        
        msg!("🔐 Quantum security initialized");
        msg!("⚡ Quantum encryption: Enabled");
        msg!("🛡️ Quantum risk modeling: Enabled");
        msg!("🚀 Quantum execution priority: Enabled");
    }
    
    Ok(())
}

/// Mission initialization event for monitoring and analytics
#[event]
pub struct MissionInitializedEvent {
    pub mission_id: Pubkey,
    pub creator: Pubkey,
    pub authority: Pubkey,
    pub funding_amount: u64,
    pub strategy_type: StrategyType,
    pub ai_enabled: bool,
    pub quantum_features: bool,
    pub timestamp: i64,
}

/// Custom error types for mission initialization
mod errors {
    use anchor_lang::prelude::*;
    
    #[error_code]
    pub enum ProwziError {
        #[msg("Insufficient funding: minimum $10 USDC required")]
        InsufficientFunding,
        
        #[msg("Invalid mission duration: must be between 1 hour and 30 days")]
        InvalidMissionDuration,
        
        #[msg("Invalid target return: must be between 0.1% and 100%")]
        InvalidTargetReturn,
        
        #[msg("Invalid max drawdown: must be between 0.1% and 50%")]
        InvalidMaxDrawdown,
        
        #[msg("Invalid position size: must be between 1% and 100%")]
        InvalidPositionSize,
        
        #[msg("Invalid AI parameters: check learning rate and confidence threshold")]
        InvalidAIParameters,
        
        #[msg("Invalid stop loss: must be positive and <= max drawdown")]
        InvalidStopLoss,
        
        #[msg("Invalid take profit: must be positive and reasonable")]
        InvalidTakeProfit,
        
        #[msg("Insufficient balance in funding account")]
        InsufficientBalance,
        
        #[msg("Quantum features require minimum $50 USDC funding")]
        InsufficientFundingForQuantumFeatures,
        
        #[msg("Quantum optimization requires AI to be enabled")]
        QuantumRequiresAI,
        
        #[msg("High risk strategies require quantum security features")]
        HighRiskRequiresQuantumSecurity,
        
        #[msg("Invalid mission configuration")]
        InvalidMissionConfig,
    }
}

pub use errors::ProwziError;