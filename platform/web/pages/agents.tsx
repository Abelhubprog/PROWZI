import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Layout from '../components/Layout';
import { 
  Users, 
  Play, 
  Pause, 
  Square, 
  Settings, 
  Activity, 
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Search,
  Filter,
  Plus,
  MoreVertical,
  ArrowLeft
} from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: 'scout' | 'analyst' | 'trader' | 'guardian' | 'curator';
  status: 'active' | 'inactive' | 'paused' | 'error';
  mission?: {
    id: string;
    name: string;
  };
  performance: {
    score: number;
    accuracy: number;
    uptime: number;
  };
  metrics: {
    tasksCompleted: number;
    avgResponseTime: number;
    lastActivity: string;
  };
  resources: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
  config: {
    model: string;
    parameters: Record<string, any>;
  };
}

const mockAgents: Agent[] = [
  {
    id: 'agent_001',
    name: 'Scout Alpha',
    type: 'scout',
    status: 'active',
    mission: { id: 'mission_1', name: 'DeFi Yield Monitoring' },
    performance: { score: 95, accuracy: 92, uptime: 99.2 },
    metrics: { tasksCompleted: 1247, avgResponseTime: 142, lastActivity: '2 min ago' },
    resources: { cpu: 23, memory: 45, gpu: 67 },
    config: { model: 'claude-3.5-sonnet', parameters: { temperature: 0.3 } }
  },
  {
    id: 'agent_002',
    name: 'Guardian Beta',
    type: 'guardian',
    status: 'active',
    mission: { id: 'mission_2', name: 'Portfolio Protection' },
    performance: { score: 89, accuracy: 96, uptime: 98.7 },
    metrics: { tasksCompleted: 892, avgResponseTime: 89, lastActivity: '1 min ago' },
    resources: { cpu: 34, memory: 52 },
    config: { model: 'gpt-4-turbo', parameters: { temperature: 0.1 } }
  },
  {
    id: 'agent_003',
    name: 'Analyst Gamma',
    type: 'analyst',
    status: 'active',
    mission: { id: 'mission_3', name: 'Market Trend Analysis' },
    performance: { score: 92, accuracy: 88, uptime: 97.4 },
    metrics: { tasksCompleted: 2156, avgResponseTime: 234, lastActivity: '30s ago' },
    resources: { cpu: 67, memory: 78, gpu: 45 },
    config: { model: 'claude-3-opus', parameters: { temperature: 0.5 } }
  },
  {
    id: 'agent_004',
    name: 'Trader Delta',
    type: 'trader',
    status: 'paused',
    performance: { score: 87, accuracy: 84, uptime: 95.1 },
    metrics: { tasksCompleted: 456, avgResponseTime: 567, lastActivity: '1h ago' },
    resources: { cpu: 12, memory: 28 },
    config: { model: 'gpt-4', parameters: { temperature: 0.2 } }
  },
  {
    id: 'agent_005',
    name: 'Curator Epsilon',
    type: 'curator',
    status: 'error',
    performance: { score: 73, accuracy: 79, uptime: 89.3 },
    metrics: { tasksCompleted: 234, avgResponseTime: 456, lastActivity: '3h ago' },
    resources: { cpu: 5, memory: 15 },
    config: { model: 'llama-3-70b', parameters: { temperature: 0.4 } }
  }
];

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>(mockAgents);
  const [filteredAgents, setFilteredAgents] = useState<Agent[]>(mockAgents);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    filterAgents();
  }, [searchTerm, statusFilter, typeFilter, agents]);

  const filterAgents = () => {
    let filtered = agents;

    if (searchTerm) {
      filtered = filtered.filter(agent => 
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(agent => agent.status === statusFilter);
    }

    if (typeFilter !== 'all') {
      filtered = filtered.filter(agent => agent.type === typeFilter);
    }

    setFilteredAgents(filtered);
  };

  const handleAgentAction = async (agentId: string, action: 'start' | 'pause' | 'stop' | 'restart') => {
    setLoading(true);
    try {
      // In production, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAgents(prev => prev.map(agent => {
        if (agent.id === agentId) {
          switch (action) {
            case 'start':
              return { ...agent, status: 'active' as const };
            case 'pause':
              return { ...agent, status: 'paused' as const };
            case 'stop':
              return { ...agent, status: 'inactive' as const };
            case 'restart':
              return { ...agent, status: 'active' as const };
            default:
              return agent;
          }
        }
        return agent;
      }));
    } catch (error) {
      console.error('Error performing agent action:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500 text-white';
      case 'paused': return 'bg-yellow-500 text-black';
      case 'error': return 'bg-red-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="h-4 w-4" />;
      case 'paused': return <Pause className="h-4 w-4" />;
      case 'error': return <AlertTriangle className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'scout': return '🔍';
      case 'analyst': return '📊';
      case 'trader': return '💹';
      case 'guardian': return '🛡️';
      case 'curator': return '📝';
      default: return '🤖';
    }
  };

  return (
    <Layout>
      <Head>
        <title>AI Agents - Prowzi</title>
        <meta name="description" content="Manage and monitor your autonomous AI agents" />
      </Head>

      <div className="bg-slate-50 dark:bg-slate-900 min-h-screen">
        {/* Page Header */}
        <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center">
              <div>
                <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">AI Agents</h1>
                <p className="text-slate-600 dark:text-slate-400">Deploy, monitor and manage your autonomous agents</p>
              </div>
              <div className="mt-4 sm:mt-0 flex space-x-3">
                <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white font-medium flex items-center space-x-2 transition-colors">
                  <Plus className="h-4 w-4" />
                  <span>Deploy Agent</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Filters */}
          <div className="mb-8 space-y-4 lg:space-y-0 lg:flex lg:items-center lg:space-x-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search agents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div className="flex space-x-4">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="paused">Paused</option>
                <option value="inactive">Inactive</option>
                <option value="error">Error</option>
              </select>

              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Types</option>
                <option value="scout">Scout</option>
                <option value="analyst">Analyst</option>
                <option value="trader">Trader</option>
                <option value="guardian">Guardian</option>
                <option value="curator">Curator</option>
              </select>
            </div>
          </div>

          {/* Agents Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAgents.map((agent) => (
              <div key={agent.id} className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 shadow-sm hover:shadow-md transition-shadow">
                {/* Agent Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getTypeIcon(agent.type)}</span>
                    <div>
                      <h3 className="text-slate-900 dark:text-white font-semibold">{agent.name}</h3>
                      <p className="text-slate-500 dark:text-slate-400 text-sm capitalize">{agent.type}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getStatusColor(agent.status)}`}>
                      {getStatusIcon(agent.status)}
                      <span className="capitalize">{agent.status}</span>
                    </span>
                    <button className="text-slate-400 hover:text-slate-600 dark:hover:text-white">
                      <MoreVertical className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Mission */}
                {agent.mission && (
                  <div className="mb-4 p-3 bg-slate-100 dark:bg-slate-700 rounded-lg">
                    <p className="text-slate-500 dark:text-slate-300 text-sm">Current Mission</p>
                    <p className="text-slate-900 dark:text-white font-medium">{agent.mission.name}</p>
                  </div>
                )}

                {/* Performance Metrics */}
                <div className="mb-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500 dark:text-slate-400 text-sm">Performance Score</span>
                    <span className="text-slate-900 dark:text-white font-medium">{agent.performance.score}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full" 
                      style={{ width: `${agent.performance.score}%` }}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-slate-500 dark:text-slate-400">Accuracy</p>
                      <p className="text-slate-900 dark:text-white font-medium">{agent.performance.accuracy}%</p>
                    </div>
                    <div>
                      <p className="text-slate-500 dark:text-slate-400">Uptime</p>
                      <p className="text-slate-900 dark:text-white font-medium">{agent.performance.uptime}%</p>
                    </div>
                  </div>
                </div>

                {/* Resource Usage */}
                <div className="mb-4 space-y-2">
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Resource Usage</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-300">CPU</span>
                      <span className="text-slate-900 dark:text-white">{agent.resources.cpu}%</span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                      <div 
                        className="bg-green-500 h-1 rounded-full" 
                        style={{ width: `${agent.resources.cpu}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-300">Memory</span>
                      <span className="text-slate-900 dark:text-white">{agent.resources.memory}%</span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                      <div 
                        className="bg-yellow-500 h-1 rounded-full" 
                        style={{ width: `${agent.resources.memory}%` }}
                      />
                    </div>

                    {agent.resources.gpu && (
                      <>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-slate-600 dark:text-slate-300">GPU</span>
                          <span className="text-slate-900 dark:text-white">{agent.resources.gpu}%</span>
                        </div>
                        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                          <div 
                            className="bg-purple-500 h-1 rounded-full" 
                            style={{ width: `${agent.resources.gpu}%` }}
                          />
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Agent Actions */}
                <div className="flex space-x-2">
                  {agent.status === 'active' ? (
                    <button 
                      onClick={() => handleAgentAction(agent.id, 'pause')}
                      disabled={loading}
                      className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-2 rounded-lg text-sm font-medium flex items-center justify-center space-x-1"
                    >
                      <Pause className="h-4 w-4" />
                      <span>Pause</span>
                    </button>
                  ) : (
                    <button 
                      onClick={() => handleAgentAction(agent.id, 'start')}
                      disabled={loading}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg text-sm font-medium flex items-center justify-center space-x-1"
                    >
                      <Play className="h-4 w-4" />
                      <span>Start</span>
                    </button>
                  )}

                  <button 
                    onClick={() => handleAgentAction(agent.id, 'stop')}
                    disabled={loading}
                    className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm font-medium"
                  >
                    <Square className="h-4 w-4" />
                  </button>

                  <button 
                    onClick={() => setSelectedAgent(agent)}
                    className="bg-slate-600 hover:bg-slate-700 text-white px-3 py-2 rounded-lg text-sm font-medium"
                  >
                    <Settings className="h-4 w-4" />
                  </button>
                </div>

                {/* Last Activity */}
                <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                  <p className="text-slate-500 dark:text-slate-400 text-xs">Last activity: {agent.metrics.lastActivity}</p>
                </div>
              </div>
            ))}
          </div>

          {filteredAgents.length === 0 && (
            <div className="text-center py-12">
              <Users className="mx-auto h-12 w-12 text-slate-400" />
              <h3 className="mt-2 text-sm font-medium text-slate-600 dark:text-slate-300">No agents found</h3>
              <p className="mt-1 text-sm text-slate-500">Try adjusting your search or filters.</p>
            </div>
          )}
        </main>

        {/* Agent Details Modal */}
        {selectedAgent && (
          <div className="fixed inset-0 z-50 overflow-y-auto">
            <div className="flex min-h-screen items-center justify-center p-4">
              <div className="fixed inset-0 bg-black opacity-50" onClick={() => setSelectedAgent(null)} />
              <div className="relative bg-white dark:bg-slate-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Agent Details</h2>
                    <button 
                      onClick={() => setSelectedAgent(null)}
                      className="text-slate-400 hover:text-slate-600 dark:hover:text-white"
                    >
                      ×
                    </button>
                  </div>

                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-3">Configuration</h3>
                      <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-slate-500 dark:text-slate-400">Model</p>
                            <p className="text-slate-900 dark:text-white">{selectedAgent.config.model}</p>
                          </div>
                          <div>
                            <p className="text-slate-500 dark:text-slate-400">Tasks Completed</p>
                            <p className="text-slate-900 dark:text-white">{selectedAgent.metrics.tasksCompleted}</p>
                          </div>
                          <div>
                            <p className="text-slate-500 dark:text-slate-400">Avg Response Time</p>
                            <p className="text-slate-900 dark:text-white">{selectedAgent.metrics.avgResponseTime}ms</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-3">Performance History</h3>
                      <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                        <p className="text-slate-500 dark:text-slate-400 text-sm">Performance tracking chart would go here</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}