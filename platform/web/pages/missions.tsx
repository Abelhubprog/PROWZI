import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { 
  Target, 
  Play, 
  Pause, 
  Square, 
  Plus, 
  Search, 
  Filter,
  Calendar,
  Clock,
  Users,
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  X,
  ArrowLeft,
  MoreVertical
} from 'lucide-react';

interface Mission {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  progress: number;
  createdAt: string;
  completedAt?: string;
  budget: {
    tokens: number;
    used: number;
    compute: 'low' | 'medium' | 'high';
  };
  agents: {
    assigned: number;
    active: number;
  };
  results: {
    briefs: number;
    alerts: number;
    actions: number;
  };
  config: {
    dataSources: string[];
    objectives: string[];
    constraints: string[];
  };
}

const mockMissions: Mission[] = [
  {
    id: 'mission_001',
    name: 'DeFi Yield Farming Monitor',
    description: 'Monitor emerging yield farming opportunities across major DeFi protocols with risk assessment',
    status: 'active',
    priority: 'high',
    progress: 67,
    createdAt: '2024-01-15T10:30:00Z',
    budget: { tokens: 50000, used: 23400, compute: 'medium' },
    agents: { assigned: 4, active: 4 },
    results: { briefs: 12, alerts: 5, actions: 3 },
    config: {
      dataSources: ['solana', 'ethereum', 'polygon'],
      objectives: ['identify high yield opportunities', 'assess risk levels', 'monitor liquidity'],
      constraints: ['min 5% APY', 'TVL > $1M', 'established protocols only']
    }
  },
  {
    id: 'mission_002',
    name: 'MEV Protection Analysis',
    description: 'Analyze and protect against MEV attacks on trading positions',
    status: 'active',
    priority: 'critical',
    progress: 85,
    createdAt: '2024-01-14T08:15:00Z',
    budget: { tokens: 75000, used: 62100, compute: 'high' },
    agents: { assigned: 2, active: 2 },
    results: { briefs: 8, alerts: 15, actions: 7 },
    config: {
      dataSources: ['solana', 'mempool'],
      objectives: ['detect MEV opportunities', 'protect user transactions', 'minimize slippage'],
      constraints: ['real-time monitoring', 'sub-100ms response', 'high accuracy required']
    }
  },
  {
    id: 'mission_003',
    name: 'Social Sentiment Analysis',
    description: 'Track social media sentiment for emerging crypto projects',
    status: 'paused',
    priority: 'medium',
    progress: 34,
    createdAt: '2024-01-13T14:20:00Z',
    budget: { tokens: 30000, used: 12800, compute: 'low' },
    agents: { assigned: 3, active: 0 },
    results: { briefs: 6, alerts: 2, actions: 1 },
    config: {
      dataSources: ['twitter', 'reddit', 'discord'],
      objectives: ['sentiment tracking', 'influencer monitoring', 'trend detection'],
      constraints: ['verified accounts only', 'min 1k followers', 'english language']
    }
  },
  {
    id: 'mission_004',
    name: 'Arbitrage Opportunity Scanner',
    description: 'Scan for arbitrage opportunities across multiple exchanges',
    status: 'completed',
    priority: 'high',
    progress: 100,
    createdAt: '2024-01-10T09:00:00Z',
    completedAt: '2024-01-12T16:45:00Z',
    budget: { tokens: 40000, used: 38900, compute: 'medium' },
    agents: { assigned: 3, active: 0 },
    results: { briefs: 25, alerts: 12, actions: 8 },
    config: {
      dataSources: ['jupiter', 'raydium', 'orca'],
      objectives: ['price difference detection', 'profitability analysis', 'execution timing'],
      constraints: ['min 0.5% profit', 'liquid markets only', 'gas fees considered']
    }
  }
];

export default function MissionsPage() {
  const [missions, setMissions] = useState<Mission[]>(mockMissions);
  const [filteredMissions, setFilteredMissions] = useState<Mission[]>(mockMissions);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [priorityFilter, setPriorityFilter] = useState<string>('all');
  const [selectedMission, setSelectedMission] = useState<Mission | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    filterMissions();
  }, [searchTerm, statusFilter, priorityFilter, missions]);

  const filterMissions = () => {
    let filtered = missions;

    if (searchTerm) {
      filtered = filtered.filter(mission => 
        mission.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        mission.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(mission => mission.status === statusFilter);
    }

    if (priorityFilter !== 'all') {
      filtered = filtered.filter(mission => mission.priority === priorityFilter);
    }

    setFilteredMissions(filtered);
  };

  const handleMissionAction = async (missionId: string, action: 'start' | 'pause' | 'stop') => {
    setLoading(true);
    try {
      // In production, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setMissions(prev => prev.map(mission => {
        if (mission.id === missionId) {
          switch (action) {
            case 'start':
              return { ...mission, status: 'active' as const };
            case 'pause':
              return { ...mission, status: 'paused' as const };
            case 'stop':
              return { ...mission, status: 'completed' as const, progress: 100 };
            default:
              return mission;
          }
        }
        return mission;
      }));
    } catch (error) {
      console.error('Error performing mission action:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500 text-white';
      case 'paused': return 'bg-yellow-500 text-black';
      case 'completed': return 'bg-blue-500 text-white';
      case 'failed': return 'bg-red-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <>
      <Head>
        <title>Missions - Prowzi Dashboard</title>
        <meta name="description" content="Create and manage AI agent missions" />
      </Head>

      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="bg-slate-800 border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <Link href="/" className="text-slate-400 hover:text-white">
                  <ArrowLeft className="h-6 w-6" />
                </Link>
                <h1 className="text-xl font-semibold text-white">Missions</h1>
              </div>
              <button 
                onClick={() => setShowCreateModal(true)}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white font-medium flex items-center space-x-2"
              >
                <Plus className="h-4 w-4" />
                <span>Create Mission</span>
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Filters */}
          <div className="mb-8 space-y-4 lg:space-y-0 lg:flex lg:items-center lg:space-x-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search missions..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div className="flex space-x-4">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="paused">Paused</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>

              <select
                value={priorityFilter}
                onChange={(e) => setPriorityFilter(e.target.value)}
                className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Priorities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>

          {/* Missions Grid */}
          <div className="space-y-6">
            {filteredMissions.map((mission) => (
              <div key={mission.id} className="bg-slate-800 rounded-lg border border-slate-700 p-6">
                {/* Mission Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-xl font-semibold text-white">{mission.name}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(mission.status)}`}>
                        {mission.status}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(mission.priority)}`}>
                        {mission.priority}
                      </span>
                    </div>
                    <p className="text-slate-400 mb-3">{mission.description}</p>
                    <div className="flex items-center space-x-6 text-sm text-slate-400">
                      <div className="flex items-center space-x-1">
                        <Calendar className="h-4 w-4" />
                        <span>Created {formatDate(mission.createdAt)}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Users className="h-4 w-4" />
                        <span>{mission.agents.active}/{mission.agents.assigned} agents</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Activity className="h-4 w-4" />
                        <span>{mission.results.briefs} briefs generated</span>
                      </div>
                    </div>
                  </div>
                  <button className="text-slate-400 hover:text-white">
                    <MoreVertical className="h-5 w-5" />
                  </button>
                </div>

                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-400">Progress</span>
                    <span className="text-sm text-white font-medium">{mission.progress}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${mission.progress}%` }}
                    />
                  </div>
                </div>

                {/* Mission Stats */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  <div className="bg-slate-700 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">Token Usage</p>
                    <p className="text-white font-medium">{mission.budget.used.toLocaleString()}/{mission.budget.tokens.toLocaleString()}</p>
                  </div>
                  <div className="bg-slate-700 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">Briefs</p>
                    <p className="text-white font-medium">{mission.results.briefs}</p>
                  </div>
                  <div className="bg-slate-700 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">Alerts</p>
                    <p className="text-white font-medium">{mission.results.alerts}</p>
                  </div>
                  <div className="bg-slate-700 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">Actions</p>
                    <p className="text-white font-medium">{mission.results.actions}</p>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-3">
                  {mission.status === 'active' ? (
                    <button 
                      onClick={() => handleMissionAction(mission.id, 'pause')}
                      disabled={loading}
                      className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2"
                    >
                      <Pause className="h-4 w-4" />
                      <span>Pause</span>
                    </button>
                  ) : mission.status === 'paused' ? (
                    <button 
                      onClick={() => handleMissionAction(mission.id, 'start')}
                      disabled={loading}
                      className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2"
                    >
                      <Play className="h-4 w-4" />
                      <span>Resume</span>
                    </button>
                  ) : null}

                  {mission.status !== 'completed' && (
                    <button 
                      onClick={() => handleMissionAction(mission.id, 'stop')}
                      disabled={loading}
                      className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2"
                    >
                      <Square className="h-4 w-4" />
                      <span>Stop</span>
                    </button>
                  )}

                  <button 
                    onClick={() => setSelectedMission(mission)}
                    className="bg-slate-600 hover:bg-slate-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
                  >
                    View Details
                  </button>

                  <Link
                    href={`/missions/${mission.id}/stream`}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2"
                  >
                    <Activity className="h-4 w-4" />
                    <span>Live Stream</span>
                  </Link>
                </div>
              </div>
            ))}
          </div>

          {filteredMissions.length === 0 && (
            <div className="text-center py-12">
              <Target className="mx-auto h-12 w-12 text-slate-500" />
              <h3 className="mt-2 text-sm font-medium text-slate-300">No missions found</h3>
              <p className="mt-1 text-sm text-slate-500">Try adjusting your search or filters.</p>
            </div>
          )}
        </main>

        {/* Create Mission Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 z-50 overflow-y-auto">
            <div className="flex min-h-screen items-center justify-center p-4">
              <div className="fixed inset-0 bg-black opacity-50" onClick={() => setShowCreateModal(false)} />
              <div className="relative bg-slate-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold text-white">Create New Mission</h2>
                    <button 
                      onClick={() => setShowCreateModal(false)}
                      className="text-slate-400 hover:text-white"
                    >
                      <X className="h-6 w-6" />
                    </button>
                  </div>

                  <form className="space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Mission Name</label>
                      <input
                        type="text"
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
                        placeholder="Enter mission name..."
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Description</label>
                      <textarea
                        rows={3}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
                        placeholder="Describe what this mission should accomplish..."
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-white mb-2">Priority</label>
                        <select className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500">
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                          <option value="critical">Critical</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-white mb-2">Token Budget</label>
                        <input
                          type="number"
                          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
                          placeholder="50000"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-white mb-2">Data Sources</label>
                      <div className="space-y-2">
                        {['solana', 'ethereum', 'twitter', 'reddit', 'github', 'arxiv'].map((source) => (
                          <label key={source} className="flex items-center space-x-2">
                            <input type="checkbox" className="rounded" />
                            <span className="text-white capitalize">{source}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div className="flex space-x-3">
                      <button
                        type="button"
                        onClick={() => setShowCreateModal(false)}
                        className="flex-1 bg-slate-600 hover:bg-slate-700 text-white px-4 py-2 rounded-lg font-medium"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium"
                      >
                        Create Mission
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Mission Details Modal */}
        {selectedMission && (
          <div className="fixed inset-0 z-50 overflow-y-auto">
            <div className="flex min-h-screen items-center justify-center p-4">
              <div className="fixed inset-0 bg-black opacity-50" onClick={() => setSelectedMission(null)} />
              <div className="relative bg-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold text-white">Mission Details</h2>
                    <button 
                      onClick={() => setSelectedMission(null)}
                      className="text-slate-400 hover:text-white"
                    >
                      <X className="h-6 w-6" />
                    </button>
                  </div>

                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-medium text-white mb-3">Configuration</h3>
                      <div className="bg-slate-700 rounded-lg p-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <p className="text-slate-400 text-sm">Data Sources</p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {selectedMission.config.dataSources.map((source) => (
                                <span key={source} className="px-2 py-1 bg-slate-600 rounded text-xs text-white">
                                  {source}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <p className="text-slate-400 text-sm">Objectives</p>
                            <ul className="text-white text-sm mt-1 space-y-1">
                              {selectedMission.config.objectives.map((objective, index) => (
                                <li key={index} className="flex items-start space-x-2">
                                  <span>•</span>
                                  <span>{objective}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-white mb-3">Performance Metrics</h3>
                      <div className="bg-slate-700 rounded-lg p-4">
                        <p className="text-slate-400 text-sm">Performance tracking chart would go here</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}