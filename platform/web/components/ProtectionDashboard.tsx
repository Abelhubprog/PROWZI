import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  Shield, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Activity,
  Zap,
  Target,
  DollarSign
} from 'lucide-react';

interface ProtectionStatus {
  positionId: string;
  token: string;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  trailingStop: {
    active: boolean;
    currentStop: number;
    trailDistance: number;
  };
  circuitBreaker: {
    active: boolean;
    triggeredCount: number;
    lastTrigger?: string;
  };
  hedgePositions: Array<{
    id: string;
    token: string;
    size: number;
    effectiveness: number;
  }>;
  insurance: {
    active: boolean;
    coverage: number;
    premium: number;
  };
  mlOptimization: {
    confidence: number;
    lastUpdate: string;
    recommendedChanges: string[];
  };
}

interface ProtectionEvent {
  id: string;
  timestamp: string;
  type: 'stop_triggered' | 'trail_updated' | 'hedge_executed' | 'emergency_exit' | 'params_optimized';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  positionId?: string;
}

export default function ProtectionDashboard() {
  const [protectedPositions, setProtectedPositions] = useState<ProtectionStatus[]>([]);
  const [recentEvents, setRecentEvents] = useState<ProtectionEvent[]>([]);
  const [overallStats, setOverallStats] = useState({
    totalPositions: 0,
    activeProtections: 0,
    totalSaved: 0,
    successRate: 0,
  });

  useEffect(() => {
    // Simulate real-time data
    const interval = setInterval(() => {
      updateProtectionData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const updateProtectionData = () => {
    // Mock data - in production, this would come from the protection system
    const mockPositions: ProtectionStatus[] = [
      {
        positionId: 'pos_1',
        token: 'SOL/USDC',
        entryPrice: 145.50,
        currentPrice: 152.30,
        unrealizedPnL: 0.047,
        trailingStop: {
          active: true,
          currentStop: 149.20,
          trailDistance: 0.02,
        },
        circuitBreaker: {
          active: true,
          triggeredCount: 0,
        },
        hedgePositions: [
          {
            id: 'hedge_1',
            token: 'BONK',
            size: 1000000,
            effectiveness: 0.75,
          }
        ],
        insurance: {
          active: true,
          coverage: 0.8,
          premium: 150,
        },
        mlOptimization: {
          confidence: 0.87,
          lastUpdate: '2 minutes ago',
          recommendedChanges: ['Tighten trail to 1.5%', 'Increase hedge ratio'],
        },
      },
      {
        positionId: 'pos_2',
        token: 'PUMP/SOL',
        entryPrice: 0.000025,
        currentPrice: 0.000031,
        unrealizedPnL: 0.24,
        trailingStop: {
          active: true,
          currentStop: 0.000028,
          trailDistance: 0.015,
        },
        circuitBreaker: {
          active: true,
          triggeredCount: 1,
          lastTrigger: '1 hour ago',
        },
        hedgePositions: [],
        insurance: {
          active: false,
          coverage: 0,
          premium: 0,
        },
        mlOptimization: {
          confidence: 0.92,
          lastUpdate: '30 seconds ago',
          recommendedChanges: ['Current parameters optimal'],
        },
      },
    ];

    const mockEvents: ProtectionEvent[] = [
      {
        id: 'evt_1',
        timestamp: '2 minutes ago',
        type: 'trail_updated',
        severity: 'low',
        message: 'Trailing stop updated for SOL/USDC position',
        positionId: 'pos_1',
      },
      {
        id: 'evt_2',
        timestamp: '5 minutes ago',
        type: 'params_optimized',
        severity: 'medium',
        message: 'ML optimizer updated protection parameters',
        positionId: 'pos_1',
      },
      {
        id: 'evt_3',
        timestamp: '1 hour ago',
        type: 'emergency_exit',
        severity: 'high',
        message: 'Circuit breaker triggered emergency exit for PUMP/SOL',
        positionId: 'pos_2',
      },
    ];

    setProtectedPositions(mockPositions);
    setRecentEvents(mockEvents);
    setOverallStats({
      totalPositions: mockPositions.length,
      activeProtections: mockPositions.filter(p => p.trailingStop.active).length,
      totalSaved: 15420, // Mock savings in USD
      successRate: 94.2,
    });
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'stop_triggered': return <XCircle className="h-4 w-4" />;
      case 'trail_updated': return <TrendingUp className="h-4 w-4" />;
      case 'hedge_executed': return <Shield className="h-4 w-4" />;
      case 'emergency_exit': return <AlertTriangle className="h-4 w-4" />;
      case 'params_optimized': return <Zap className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Protection Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor and manage advanced trading protections
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            <Shield className="mr-2 h-4 w-4" />
            Configure Protection
          </Button>
          <Button>
            <Target className="mr-2 h-4 w-4" />
            Emergency Stop All
          </Button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Protected Positions</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overallStats.totalPositions}</div>
            <p className="text-xs text-muted-foreground">
              {overallStats.activeProtections} active protections
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Saved</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${overallStats.totalSaved.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Losses prevented by protection systems
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overallStats.successRate}%</div>
            <p className="text-xs text-muted-foreground">
              Protection trigger accuracy
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Confidence</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {protectedPositions.length > 0 ? 
                (protectedPositions.reduce((sum, p) => sum + p.mlOptimization.confidence, 0) / protectedPositions.length * 100).toFixed(1) : 0}%
            </div>
            <p className="text-xs text-muted-foreground">
              Average ML model confidence
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Protected Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Protected Positions</CardTitle>
          <CardDescription>
            Real-time monitoring of all positions with active protection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {protectedPositions.map((position) => (
              <div key={position.positionId} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <Badge variant="outline" className="font-mono">
                      {position.token}
                    </Badge>
                    <div className="flex items-center space-x-1">
                      {position.unrealizedPnL >= 0 ? (
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                      )}
                      <span className={`font-medium ${position.unrealizedPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {(position.unrealizedPnL * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    {position.trailingStop.active && (
                      <Badge variant="secondary">Trailing Stop</Badge>
                    )}
                    {position.circuitBreaker.active && (
                      <Badge variant="secondary">Circuit Breaker</Badge>
                    )}
                    {position.insurance.active && (
                      <Badge variant="secondary">Insured</Badge>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Price Info */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Price Information</h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span>Entry:</span>
                        <span className="font-mono">${position.entryPrice}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Current:</span>
                        <span className="font-mono">${position.currentPrice}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Stop:</span>
                        <span className="font-mono">${position.trailingStop.currentStop}</span>
                      </div>
                    </div>
                  </div>

                  {/* Protection Status */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Protection Status</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Trail Distance:</span>
                        <Badge variant="outline">
                          {(position.trailingStop.trailDistance * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      {position.hedgePositions.length > 0 && (
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Hedges:</span>
                          <Badge variant="outline">
                            {position.hedgePositions.length} active
                          </Badge>
                        </div>
                      )}
                      {position.insurance.active && (
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Coverage:</span>
                          <Badge variant="outline">
                            {(position.insurance.coverage * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* ML Optimization */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">ML Optimization</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Confidence:</span>
                        <Progress 
                          value={position.mlOptimization.confidence * 100} 
                          className="w-16 h-2"
                        />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Updated {position.mlOptimization.lastUpdate}
                      </div>
                      {position.mlOptimization.recommendedChanges.length > 0 && (
                        <div className="text-xs">
                          <div className="font-medium">Recommendations:</div>
                          {position.mlOptimization.recommendedChanges.map((change, idx) => (
                            <div key={idx} className="text-muted-foreground">• {change}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Events */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Protection Events</CardTitle>
          <CardDescription>
            Latest protection system activities and alerts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentEvents.map((event) => (
              <Alert key={event.id}>
                <div className={`w-2 h-2 rounded-full ${getSeverityColor(event.severity)} mt-2`} />
                <div className="flex items-start space-x-3 ml-3">
                  {getEventIcon(event.type)}
                  <div className="flex-1">
                    <AlertTitle className="text-sm">{event.message}</AlertTitle>
                    <AlertDescription className="text-xs">
                      {event.timestamp} {event.positionId && `• Position: ${event.positionId}`}
                    </AlertDescription>
                  </div>
                </div>
              </Alert>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}