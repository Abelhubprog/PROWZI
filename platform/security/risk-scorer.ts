/**
 * Prowzi Platform Security Risk Scorer
 * 
 * This module provides comprehensive risk assessment capabilities for the Prowzi platform,
 * analyzing various risk factors across trading, technical, and operational domains.
 */

export interface RiskFactors {
  // Market Risk Factors
  volatility: number;           // 0-1 (market volatility level)
  correlation: number;          // 0-1 (correlation with other positions)
  liquidity: number;           // 0-1 (market liquidity level, 1 = highly liquid)
  concentration: number;        // 0-1 (position concentration risk)
  
  // Technical Risk Factors
  systemLoad: number;          // 0-1 (current system load)
  networkLatency: number;      // 0-1 (network latency risk)
  apiErrors: number;           // 0-1 (API error rate)
  dataQuality: number;         // 0-1 (data quality score, 1 = high quality)
  
  // Operational Risk Factors
  userBehavior: number;        // 0-1 (anomalous user behavior)
  complianceStatus: number;    // 0-1 (compliance violations, 0 = compliant)
  securityIncidents: number;   // 0-1 (recent security incidents)
  
  // External Risk Factors
  geopoliticalRisk: number;    // 0-1 (geopolitical instability)
  regulatoryRisk: number;      // 0-1 (regulatory uncertainty)
  cyberThreatLevel: number;    // 0-1 (current cyber threat level)
}

export interface RiskWeights {
  market: number;
  technical: number;
  operational: number;
  external: number;
}

export interface RiskScore {
  overall: number;             // 0-100 overall risk score
  category: RiskCategory;      // risk category
  components: {
    market: number;
    technical: number;
    operational: number;
    external: number;
  };
  confidence: number;          // 0-1 confidence in the score
  recommendation: RiskRecommendation;
  timestamp: Date;
}

export enum RiskCategory {
  LOW = 'low',           // 0-25
  MODERATE = 'moderate', // 26-50
  HIGH = 'high',         // 51-75
  CRITICAL = 'critical'  // 76-100
}

export enum RiskRecommendation {
  PROCEED = 'proceed',
  CAUTION = 'caution',
  REVIEW_REQUIRED = 'review_required',
  HALT = 'halt'
}

export interface TradingRiskFactors extends RiskFactors {
  // Trading-specific factors
  positionSize: number;        // 0-1 (position size relative to portfolio)
  leverage: number;            // 0-1 (leverage usage)
  stopLossDistance: number;    // 0-1 (stop loss distance)
  timeInPosition: number;      // 0-1 (time held in position)
  portfolioDrawdown: number;   // 0-1 (current drawdown level)
}

export interface OperationalRiskFactors extends RiskFactors {
  // Operational-specific factors
  staffingLevel: number;       // 0-1 (adequate staffing, 1 = fully staffed)
  systemUptime: number;        // 0-1 (system uptime percentage)
  backupStatus: number;        // 0-1 (backup system status, 1 = all good)
  processCompliance: number;   // 0-1 (process compliance level)
}

export class RiskScorer {
  private defaultWeights: RiskWeights = {
    market: 0.35,
    technical: 0.25,
    operational: 0.25,
    external: 0.15
  };

  private confidenceThresholds = {
    high: 0.8,
    medium: 0.6,
    low: 0.4
  };

  constructor(private customWeights?: Partial<RiskWeights>) {
    if (customWeights) {
      this.defaultWeights = { ...this.defaultWeights, ...customWeights };
    }
    
    // Normalize weights to sum to 1
    const totalWeight = Object.values(this.defaultWeights).reduce((sum, weight) => sum + weight, 0);
    Object.keys(this.defaultWeights).forEach(key => {
      this.defaultWeights[key as keyof RiskWeights] /= totalWeight;
    });
  }

  /**
   * Calculate overall risk score from risk factors
   */
  calculateRiskScore(factors: RiskFactors, weights?: RiskWeights): RiskScore {
    const finalWeights = weights || this.defaultWeights;
    
    // Calculate component scores (0-100)
    const marketScore = this.calculateMarketRisk(factors) * 100;
    const technicalScore = this.calculateTechnicalRisk(factors) * 100;
    const operationalScore = this.calculateOperationalRisk(factors) * 100;
    const externalScore = this.calculateExternalRisk(factors) * 100;

    // Calculate weighted overall score
    const overallScore = (
      marketScore * finalWeights.market +
      technicalScore * finalWeights.technical +
      operationalScore * finalWeights.operational +
      externalScore * finalWeights.external
    );

    // Calculate confidence based on data quality and completeness
    const confidence = this.calculateConfidence(factors);

    const result: RiskScore = {
      overall: Math.round(overallScore * 100) / 100,
      category: this.getRiskCategory(overallScore),
      components: {
        market: Math.round(marketScore * 100) / 100,
        technical: Math.round(technicalScore * 100) / 100,
        operational: Math.round(operationalScore * 100) / 100,
        external: Math.round(externalScore * 100) / 100
      },
      confidence: Math.round(confidence * 100) / 100,
      recommendation: this.getRecommendation(overallScore, confidence),
      timestamp: new Date()
    };

    return result;
  }

  /**
   * Calculate market risk component
   */
  private calculateMarketRisk(factors: RiskFactors): number {
    const volatilityRisk = factors.volatility * 0.4;
    const correlationRisk = factors.correlation * 0.2;
    const liquidityRisk = (1 - factors.liquidity) * 0.3; // Lower liquidity = higher risk
    const concentrationRisk = factors.concentration * 0.1;

    return volatilityRisk + correlationRisk + liquidityRisk + concentrationRisk;
  }

  /**
   * Calculate technical risk component
   */
  private calculateTechnicalRisk(factors: RiskFactors): number {
    const systemLoadRisk = factors.systemLoad * 0.3;
    const latencyRisk = factors.networkLatency * 0.3;
    const apiErrorRisk = factors.apiErrors * 0.2;
    const dataQualityRisk = (1 - factors.dataQuality) * 0.2; // Lower quality = higher risk

    return systemLoadRisk + latencyRisk + apiErrorRisk + dataQualityRisk;
  }

  /**
   * Calculate operational risk component
   */
  private calculateOperationalRisk(factors: RiskFactors): number {
    const behaviorRisk = factors.userBehavior * 0.4;
    const complianceRisk = factors.complianceStatus * 0.3;
    const securityRisk = factors.securityIncidents * 0.3;

    return behaviorRisk + complianceRisk + securityRisk;
  }

  /**
   * Calculate external risk component
   */
  private calculateExternalRisk(factors: RiskFactors): number {
    const geopoliticalRisk = factors.geopoliticalRisk * 0.4;
    const regulatoryRisk = factors.regulatoryRisk * 0.3;
    const cyberRisk = factors.cyberThreatLevel * 0.3;

    return geopoliticalRisk + regulatoryRisk + cyberRisk;
  }

  /**
   * Calculate confidence in the risk score
   */
  private calculateConfidence(factors: RiskFactors): number {
    // Base confidence on data completeness and quality
    const dataQualityWeight = factors.dataQuality;
    const completenessScore = this.calculateCompletenessScore(factors);
    
    // Adjust confidence based on the age of data and system stability
    const stabilityFactor = 1 - (factors.systemLoad * 0.1 + factors.apiErrors * 0.1);
    
    return (dataQualityWeight * 0.5 + completenessScore * 0.3 + stabilityFactor * 0.2);
  }

  /**
   * Calculate data completeness score
   */
  private calculateCompletenessScore(factors: RiskFactors): number {
    const requiredFields = Object.keys(factors);
    const validFields = requiredFields.filter(key => {
      const value = factors[key as keyof RiskFactors];
      return value !== undefined && value !== null && !isNaN(value);
    });

    return validFields.length / requiredFields.length;
  }

  /**
   * Determine risk category from score
   */
  private getRiskCategory(score: number): RiskCategory {
    if (score <= 25) return RiskCategory.LOW;
    if (score <= 50) return RiskCategory.MODERATE;
    if (score <= 75) return RiskCategory.HIGH;
    return RiskCategory.CRITICAL;
  }

  /**
   * Get recommendation based on score and confidence
   */
  private getRecommendation(score: number, confidence: number): RiskRecommendation {
    // Lower confidence requires more conservative recommendations
    const adjustedScore = confidence < this.confidenceThresholds.medium ? score * 1.2 : score;

    if (adjustedScore >= 75) return RiskRecommendation.HALT;
    if (adjustedScore >= 50) return RiskRecommendation.REVIEW_REQUIRED;
    if (adjustedScore >= 25) return RiskRecommendation.CAUTION;
    return RiskRecommendation.PROCEED;
  }

  /**
   * Calculate trading-specific risk score
   */
  calculateTradingRisk(factors: TradingRiskFactors): RiskScore {
    // Enhanced market risk calculation for trading
    const enhancedFactors = { ...factors };
    
    // Adjust market risk based on trading-specific factors
    enhancedFactors.volatility = Math.min(1, factors.volatility + factors.leverage * 0.3);
    enhancedFactors.concentration = Math.max(factors.concentration, factors.positionSize);
    
    // Add drawdown risk to operational component
    enhancedFactors.userBehavior = Math.max(factors.userBehavior, factors.portfolioDrawdown);

    return this.calculateRiskScore(enhancedFactors);
  }

  /**
   * Calculate operational-specific risk score
   */
  calculateOperationalRisk(factors: OperationalRiskFactors): RiskScore {
    const enhancedFactors = { ...factors };
    
    // Adjust technical risk based on operational factors
    enhancedFactors.systemLoad = Math.max(factors.systemLoad, 1 - factors.systemUptime);
    enhancedFactors.dataQuality = Math.min(factors.dataQuality, factors.backupStatus);
    
    // Enhance compliance component
    enhancedFactors.complianceStatus = Math.max(factors.complianceStatus, 1 - factors.processCompliance);

    const weights: RiskWeights = {
      market: 0.15,
      technical: 0.35,
      operational: 0.40,
      external: 0.10
    };

    return this.calculateRiskScore(enhancedFactors, weights);
  }

  /**
   * Perform trend analysis on historical risk scores
   */
  analyzeTrend(historicalScores: RiskScore[]): RiskTrendAnalysis {
    if (historicalScores.length < 2) {
      return {
        trend: 'insufficient_data',
        velocity: 0,
        acceleration: 0,
        prediction: null
      };
    }

    const scores = historicalScores.map(s => s.overall);
    const trend = this.calculateTrend(scores);
    const velocity = this.calculateVelocity(scores);
    const acceleration = this.calculateAcceleration(scores);
    
    return {
      trend: this.classifyTrend(velocity),
      velocity,
      acceleration,
      prediction: this.predictNextScore(scores)
    };
  }

  private calculateTrend(scores: number[]): number {
    const n = scores.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = scores.reduce((sum, score) => sum + score, 0);
    const sumXY = scores.reduce((sum, score, index) => sum + (index * score), 0);
    const sumX2 = scores.reduce((sum, _, index) => sum + (index * index), 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  }

  private calculateVelocity(scores: number[]): number {
    if (scores.length < 2) return 0;
    const recent = scores.slice(-5); // Use last 5 scores
    return this.calculateTrend(recent);
  }

  private calculateAcceleration(scores: number[]): number {
    if (scores.length < 3) return 0;
    const velocities: number[] = [];
    
    for (let i = 1; i < scores.length; i++) {
      velocities.push(scores[i] - scores[i - 1]);
    }
    
    return this.calculateTrend(velocities);
  }

  private classifyTrend(velocity: number): string {
    if (Math.abs(velocity) < 0.5) return 'stable';
    return velocity > 0 ? 'increasing' : 'decreasing';
  }

  private predictNextScore(scores: number[]): number | null {
    if (scores.length < 3) return null;
    
    const trend = this.calculateTrend(scores);
    const lastScore = scores[scores.length - 1];
    
    return Math.max(0, Math.min(100, lastScore + trend));
  }
}

export interface RiskTrendAnalysis {
  trend: string;
  velocity: number;
  acceleration: number;
  prediction: number | null;
}

export class RiskMonitor {
  private riskScorer: RiskScorer;
  private alertThresholds: Record<RiskCategory, number>;
  private historicalScores: RiskScore[] = [];

  constructor(
    customWeights?: Partial<RiskWeights>,
    alertThresholds?: Partial<Record<RiskCategory, number>>
  ) {
    this.riskScorer = new RiskScorer(customWeights);
    this.alertThresholds = {
      [RiskCategory.LOW]: 25,
      [RiskCategory.MODERATE]: 50,
      [RiskCategory.HIGH]: 75,
      [RiskCategory.CRITICAL]: 90,
      ...alertThresholds
    };
  }

  /**
   * Monitor risk factors and generate alerts
   */
  monitor(factors: RiskFactors): RiskMonitorResult {
    const currentScore = this.riskScorer.calculateRiskScore(factors);
    this.historicalScores.push(currentScore);
    
    // Keep only last 100 scores
    if (this.historicalScores.length > 100) {
      this.historicalScores = this.historicalScores.slice(-100);
    }

    const alerts = this.generateAlerts(currentScore);
    const trend = this.riskScorer.analyzeTrend(this.historicalScores);

    return {
      currentScore,
      alerts,
      trend,
      historicalScores: [...this.historicalScores]
    };
  }

  private generateAlerts(score: RiskScore): RiskAlert[] {
    const alerts: RiskAlert[] = [];

    // Overall risk alerts
    if (score.overall >= this.alertThresholds[RiskCategory.CRITICAL]) {
      alerts.push({
        level: 'critical',
        message: `Critical risk level detected: ${score.overall}%`,
        component: 'overall',
        threshold: this.alertThresholds[RiskCategory.CRITICAL],
        currentValue: score.overall
      });
    } else if (score.overall >= this.alertThresholds[RiskCategory.HIGH]) {
      alerts.push({
        level: 'high',
        message: `High risk level detected: ${score.overall}%`,
        component: 'overall',
        threshold: this.alertThresholds[RiskCategory.HIGH],
        currentValue: score.overall
      });
    }

    // Component-specific alerts
    Object.entries(score.components).forEach(([component, value]) => {
      if (value >= 80) {
        alerts.push({
          level: 'high',
          message: `High ${component} risk: ${value}%`,
          component,
          threshold: 80,
          currentValue: value
        });
      }
    });

    // Low confidence alert
    if (score.confidence < 0.5) {
      alerts.push({
        level: 'warning',
        message: `Low confidence in risk assessment: ${(score.confidence * 100).toFixed(1)}%`,
        component: 'confidence',
        threshold: 50,
        currentValue: score.confidence * 100
      });
    }

    return alerts;
  }
}

export interface RiskMonitorResult {
  currentScore: RiskScore;
  alerts: RiskAlert[];
  trend: RiskTrendAnalysis;
  historicalScores: RiskScore[];
}

export interface RiskAlert {
  level: 'info' | 'warning' | 'high' | 'critical';
  message: string;
  component: string;
  threshold: number;
  currentValue: number;
}

// Export default instance
export const defaultRiskScorer = new RiskScorer();
export const defaultRiskMonitor = new RiskMonitor();