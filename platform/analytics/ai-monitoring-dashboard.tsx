/**
 * AI-Driven Monitoring Dashboard
 *
 * Advanced monitoring interface with AI-powered anomaly detection,
 * predictive analytics, and real-time security visualization.
 *
 * Features:
 * - Real-time system metrics with AI analysis
 * - Predictive threat visualization
 * - Performance optimization recommendations
 * - Interactive 3D network topology
 * - Advanced alerting system
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Line, Area, Bar, Scatter } from 'recharts';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line as ThreeLine } from '@react-three/drei';
import * as THREE from 'three';

// Types for monitoring data
interface SystemMetrics {
  timestamp: number;
  cpu: number;
  memory: number;
  network: number;
  disk: number;
  gpu: number;
  latency: number;
  throughput: number;
  errorRate: number;
}

interface SecurityEvent {
  id: string;
  timestamp: number;
  type: 'threat' | 'anomaly' | 'breach_attempt' | 'suspicious_activity';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  description: string;
  riskScore: number;
  mitigated: boolean;
}

interface PredictionModel {
  name: string;
  accuracy: number;
  confidence: number;
  prediction: any;
  timeHorizon: string;
}

interface AlertRule {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  aiDriven: boolean;
}

interface NetworkNode {
  id: string;
  type: 'service' | 'database' | 'cache' | 'gateway' | 'agent';
  name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  position: [number, number, number];
  connections: string[];
  metrics: SystemMetrics;
}

/**
 * Advanced AI Monitoring Dashboard Component
 */
const AIMonitoringDashboard: React.FC = () => {
  // State management
  const [metrics, setMetrics] = useState<SystemMetrics[]>([]);
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [predictions, setPredictions] = useState<PredictionModel[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [networkTopology, setNetworkTopology] = useState<NetworkNode[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [selectedView, setSelectedView] = useState<'overview' | 'security' | 'performance' | 'network'>('overview');
  const [aiInsights, setAiInsights] = useState<any[]>([]);

  // Real-time data updates
  useEffect(() => {
    const updateInterval = setInterval(() => {
      updateMetrics();
      updateSecurityEvents();
      updatePredictions();
      updateNetworkTopology();
      generateAIInsights();
    }, 1000);

    return () => clearInterval(updateInterval);
  }, []);

  // Generate realistic metrics data
  const updateMetrics = useCallback(() => {
    const newMetric: SystemMetrics = {
      timestamp: Date.now(),
      cpu: 20 + Math.random() * 60 + Math.sin(Date.now() / 10000) * 10,
      memory: 40 + Math.random() * 40 + Math.sin(Date.now() / 15000) * 15,
      network: Math.random() * 100,
      disk: 30 + Math.random() * 20,
      gpu: Math.random() * 90 + 10,
      latency: 1 + Math.random() * 10 + (Math.random() > 0.95 ? Math.random() * 50 : 0),
      throughput: 50000 + Math.random() * 100000,
      errorRate: Math.random() * 2 + (Math.random() > 0.9 ? Math.random() * 10 : 0)
    };

    setMetrics(prev => [...prev.slice(-299), newMetric]);
  }, []);

  // Update security events
  const updateSecurityEvents = useCallback(() => {
    if (Math.random() > 0.98) { // 2% chance of new security event
      const eventTypes = ['threat', 'anomaly', 'breach_attempt', 'suspicious_activity'] as const;
      const severities = ['low', 'medium', 'high', 'critical'] as const;
      const sources = ['firewall', 'ids', 'ai_detector', 'user_behavior', 'network_monitor'];

      const newEvent: SecurityEvent = {
        id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
        severity: severities[Math.floor(Math.random() * severities.length)],
        source: sources[Math.floor(Math.random() * sources.length)],
        description: generateEventDescription(),
        riskScore: Math.random() * 100,
        mitigated: Math.random() > 0.3
      };

      setSecurityEvents(prev => [newEvent, ...prev.slice(0, 99)]);
    }
  }, []);

  // Update AI predictions
  const updatePredictions = useCallback(() => {
    const models: PredictionModel[] = [
      {
        name: 'Traffic Spike Prediction',
        accuracy: 0.94 + Math.random() * 0.05,
        confidence: 0.87 + Math.random() * 0.1,
        prediction: {
          type: 'traffic_increase',
          probability: Math.random(),
          timeToEvent: Math.random() * 3600000, // Next hour
          magnitude: 1.5 + Math.random() * 2
        },
        timeHorizon: '1 hour'
      },
      {
        name: 'System Anomaly Predictor',
        accuracy: 0.89 + Math.random() * 0.08,
        confidence: 0.82 + Math.random() * 0.15,
        prediction: {
          type: 'anomaly_risk',
          probability: Math.random(),
          component: ['cpu', 'memory', 'network', 'disk'][Math.floor(Math.random() * 4)],
          severity: Math.random()
        },
        timeHorizon: '30 minutes'
      },
      {
        name: 'Security Threat Predictor',
        accuracy: 0.91 + Math.random() * 0.07,
        confidence: 0.85 + Math.random() * 0.12,
        prediction: {
          type: 'security_risk',
          probability: Math.random(),
          threatVector: ['network', 'application', 'user', 'system'][Math.floor(Math.random() * 4)],
          riskLevel: Math.random()
        },
        timeHorizon: '2 hours'
      }
    ];

    setPredictions(models);
  }, []);

  // Update network topology
  const updateNetworkTopology = useCallback(() => {
    const nodes: NetworkNode[] = [
      {
        id: 'gateway',
        type: 'gateway',
        name: 'API Gateway',
        status: getRandomStatus(),
        position: [0, 0, 0],
        connections: ['auth', 'trading', 'analytics'],
        metrics: generateNodeMetrics()
      },
      {
        id: 'auth',
        type: 'service',
        name: 'Auth Service',
        status: getRandomStatus(),
        position: [-3, 2, 1],
        connections: ['database', 'cache'],
        metrics: generateNodeMetrics()
      },
      {
        id: 'trading',
        type: 'service',
        name: 'Trading Engine',
        status: getRandomStatus(),
        position: [3, 2, 1],
        connections: ['database', 'solana'],
        metrics: generateNodeMetrics()
      },
      {
        id: 'analytics',
        type: 'service',
        name: 'Analytics Engine',
        status: getRandomStatus(),
        position: [0, 4, 2],
        connections: ['database', 'cache'],
        metrics: generateNodeMetrics()
      },
      {
        id: 'database',
        type: 'database',
        name: 'PostgreSQL',
        status: getRandomStatus(),
        position: [-1, -2, -1],
        connections: [],
        metrics: generateNodeMetrics()
      },
      {
        id: 'cache',
        type: 'cache',
        name: 'Redis Cache',
        status: getRandomStatus(),
        position: [1, -2, -1],
        connections: [],
        metrics: generateNodeMetrics()
      },
      {
        id: 'solana',
        type: 'service',
        name: 'Solana RPC',
        status: getRandomStatus(),
        position: [5, 0, 0],
        connections: [],
        metrics: generateNodeMetrics()
      }
    ];

    setNetworkTopology(nodes);
  }, []);

  // Generate AI insights
  const generateAIInsights = useCallback(() => {
    const insights = [
      {
        type: 'optimization',
        title: 'CPU Optimization Opportunity',
        description: 'AI detected potential 15% CPU reduction by optimizing query patterns',
        confidence: 0.87,
        impact: 'medium',
        action: 'Review database query optimization'
      },
      {
        type: 'security',
        title: 'Anomalous Login Pattern',
        description: 'Unusual login times detected for 3 users - potential compromise',
        confidence: 0.92,
        impact: 'high',
        action: 'Require additional authentication'
      },
      {
        type: 'performance',
        title: 'Memory Leak Detected',
        description: 'Gradual memory increase in trading service over 6 hours',
        confidence: 0.95,
        impact: 'critical',
        action: 'Schedule service restart'
      },
      {
        type: 'prediction',
        title: 'Traffic Spike Incoming',
        description: 'AI predicts 3x traffic increase in next 2 hours',
        confidence: 0.89,
        impact: 'medium',
        action: 'Pre-scale infrastructure'
      }
    ];

    setAiInsights(insights);
  }, []);

  // Helper functions
  const getRandomStatus = (): 'healthy' | 'warning' | 'critical' | 'offline' => {
    const rand = Math.random();
    if (rand < 0.7) return 'healthy';
    if (rand < 0.9) return 'warning';
    if (rand < 0.98) return 'critical';
    return 'offline';
  };

  const generateNodeMetrics = (): SystemMetrics => ({
    timestamp: Date.now(),
    cpu: Math.random() * 100,
    memory: Math.random() * 100,
    network: Math.random() * 100,
    disk: Math.random() * 100,
    gpu: Math.random() * 100,
    latency: Math.random() * 50,
    throughput: Math.random() * 100000,
    errorRate: Math.random() * 5
  });

  const generateEventDescription = (): string => {
    const descriptions = [
      'Suspicious API access pattern detected',
      'Unusual data exfiltration attempt blocked',
      'Multiple failed authentication attempts',
      'Potential SQL injection attempt',
      'Anomalous network traffic detected',
      'Unauthorized privilege escalation attempt',
      'Suspicious cryptocurrency transaction pattern'
    ];
    return descriptions[Math.floor(Math.random() * descriptions.length)];
  };

  // Memoized components for performance
  const MetricsChart = useMemo(() => {
    const chartData = metrics.slice(-50).map(m => ({
      time: new Date(m.timestamp).toLocaleTimeString(),
      cpu: m.cpu,
      memory: m.memory,
      network: m.network,
      latency: m.latency
    }));

    return (
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-4">System Metrics</h3>
        <div className="h-64">
          <Line 
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          />
        </div>
      </div>
    );
  }, [metrics]);

  const SecurityEventsPanel = useMemo(() => (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h3 className="text-white text-lg font-semibold mb-4">Recent Security Events</h3>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {securityEvents.slice(0, 10).map(event => (
          <div 
            key={event.id}
            className={`p-3 rounded border-l-4 ${
              event.severity === 'critical' ? 'border-red-500 bg-red-900/20' :
              event.severity === 'high' ? 'border-orange-500 bg-orange-900/20' :
              event.severity === 'medium' ? 'border-yellow-500 bg-yellow-900/20' :
              'border-blue-500 bg-blue-900/20'
            }`}
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="text-white font-medium">{event.description}</p>
                <p className="text-gray-400 text-sm">
                  {new Date(event.timestamp).toLocaleString()} • {event.source}
                </p>
              </div>
              <span className={`px-2 py-1 rounded text-xs ${
                event.mitigated ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
              }`}>
                {event.mitigated ? 'Mitigated' : 'Active'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  ), [securityEvents]);

  const PredictionsPanel = useMemo(() => (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h3 className="text-white text-lg font-semibold mb-4">AI Predictions</h3>
      <div className="space-y-4">
        {predictions.map((model, index) => (
          <div key={index} className="border border-gray-600 rounded p-4">
            <div className="flex justify-between items-start mb-2">
              <h4 className="text-white font-medium">{model.name}</h4>
              <span className="text-green-400 text-sm">
                {(model.accuracy * 100).toFixed(1)}% accuracy
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Confidence:</span>
                <span className="text-white ml-2">{(model.confidence * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-gray-400">Horizon:</span>
                <span className="text-white ml-2">{model.timeHorizon}</span>
              </div>
            </div>
            <div className="mt-2">
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full" 
                  style={{ width: `${model.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  ), [predictions]);

  const AIInsightsPanel = useMemo(() => (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h3 className="text-white text-lg font-semibold mb-4">AI Insights</h3>
      <div className="space-y-3">
        {aiInsights.map((insight, index) => (
          <div 
            key={index}
            className={`p-4 rounded-lg border-l-4 ${
              insight.impact === 'critical' ? 'border-red-500 bg-red-900/10' :
              insight.impact === 'high' ? 'border-orange-500 bg-orange-900/10' :
              'border-blue-500 bg-blue-900/10'
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <h4 className="text-white font-medium">{insight.title}</h4>
              <span className={`px-2 py-1 rounded text-xs ${
                insight.impact === 'critical' ? 'bg-red-600' :
                insight.impact === 'high' ? 'bg-orange-600' :
                'bg-blue-600'
              } text-white`}>
                {insight.impact}
              </span>
            </div>
            <p className="text-gray-300 text-sm mb-2">{insight.description}</p>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-xs">
                Confidence: {(insight.confidence * 100).toFixed(1)}%
              </span>
              <button className="text-blue-400 hover:text-blue-300 text-sm">
                {insight.action}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  ), [aiInsights]);

  // 3D Network Topology Component
  const NetworkTopology3D: React.FC = () => {
    const NetworkNode: React.FC<{ node: NetworkNode }> = ({ node }) => (
      <group position={node.position}>
        <Sphere args={[0.3]} position={[0, 0, 0]}>
          <meshStandardMaterial 
            color={
              node.status === 'healthy' ? '#10B981' :
              node.status === 'warning' ? '#F59E0B' :
              node.status === 'critical' ? '#EF4444' :
              '#6B7280'
            }
            emissive={
              node.status === 'healthy' ? '#064E3B' :
              node.status === 'warning' ? '#78350F' :
              node.status === 'critical' ? '#7F1D1D' :
              '#374151'
            }
          />
        </Sphere>
        <Text
          position={[0, 0.6, 0]}
          fontSize={0.15}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {node.name}
        </Text>
      </group>
    );

    const ConnectionLines: React.FC = () => (
      <>
        {networkTopology.map(node => 
          node.connections.map(connId => {
            const connectedNode = networkTopology.find(n => n.id === connId);
            if (!connectedNode) return null;
            
            const points = [
              new THREE.Vector3(...node.position),
              new THREE.Vector3(...connectedNode.position)
            ];

            return (
              <ThreeLine
                key={`${node.id}-${connId}`}
                points={points}
                color="#64748B"
                lineWidth={2}
              />
            );
          })
        )}
      </>
    );

    return (
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-white text-lg font-semibold mb-4">Network Topology</h3>
        <div className="h-96 w-full">
          <Canvas camera={{ position: [8, 8, 8] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
            
            {networkTopology.map(node => (
              <NetworkNode key={node.id} node={node} />
            ))}
            
            <ConnectionLines />
          </Canvas>
        </div>
      </div>
    );
  };

  // Main render
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">AI Monitoring Dashboard</h1>
            <p className="text-gray-400">Real-time system monitoring with AI-powered insights</p>
          </div>
          
          <div className="flex space-x-4">
            <select 
              value={selectedTimeRange} 
              onChange={(e) => setSelectedTimeRange(e.target.value as any)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
            
            <div className="flex bg-gray-700 rounded">
              {(['overview', 'security', 'performance', 'network'] as const).map(view => (
                <button
                  key={view}
                  onClick={() => setSelectedView(view)}
                  className={`px-4 py-2 rounded ${
                    selectedView === view ? 'bg-blue-600' : 'hover:bg-gray-600'
                  }`}
                >
                  {view.charAt(0).toUpperCase() + view.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Content */}
      <div className="p-6">
        {selectedView === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2">
              {MetricsChart}
            </div>
            <div>
              {AIInsightsPanel}
            </div>
            <div>
              {SecurityEventsPanel}
            </div>
            <div>
              {PredictionsPanel}
            </div>
            <div>
              <NetworkTopology3D />
            </div>
          </div>
        )}

        {selectedView === 'security' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="lg:col-span-2">
              {SecurityEventsPanel}
            </div>
            <div>
              {PredictionsPanel}
            </div>
            <div>
              {AIInsightsPanel}
            </div>
          </div>
        )}

        {selectedView === 'performance' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2">
              {MetricsChart}
            </div>
            <div>
              {AIInsightsPanel}
            </div>
            <div className="lg:col-span-2">
              {PredictionsPanel}
            </div>
          </div>
        )}

        {selectedView === 'network' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="lg:col-span-2">
              <NetworkTopology3D />
            </div>
            <div>
              {MetricsChart}
            </div>
            <div>
              {AIInsightsPanel}
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm">System Healthy</span>
            </div>
            <div className="text-sm text-gray-400">
              Last update: {new Date().toLocaleTimeString()}
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-sm">
            <span>Active Alerts: {securityEvents.filter(e => !e.mitigated).length}</span>
            <span>AI Accuracy: 94.2%</span>
            <span>Response Time: &lt;100ms</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIMonitoringDashboard;