import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Layout from '../components/Layout';
import { 
  Activity, 
  Shield, 
  TrendingUp, 
  Users, 
  Zap, 
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Target,
  Settings,
  Bell,
  Menu,
  X
} from 'lucide-react';

interface DashboardStats {
  activeAgents: number;
  activeMissions: number;
  eventsToday: number;
  briefsGenerated: number;
  totalSaved: number;
  successRate: number;
  protectedPositions: number;
}

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'paused' | 'error';
  mission?: string;
  lastActivity: string;
  performance: number;
}

interface Mission {
  id: string;
  name: string;
  status: 'active' | 'completed' | 'paused';
  progress: number;
  createdAt: string;
  agentCount: number;
}

interface Brief {
  id: string;
  title: string;
  summary: string;
  impact: 'critical' | 'high' | 'medium' | 'low';
  confidence: number;
  timestamp: string;
  source: string;
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    activeAgents: 0,
    activeMissions: 0,
    eventsToday: 0,
    briefsGenerated: 0,
    totalSaved: 0,
    successRate: 0,
    protectedPositions: 0
  });
  const [agents, setAgents] = useState<Agent[]>([]);
  const [missions, setMissions] = useState<Mission[]>([]);
  const [recentBriefs, setRecentBriefs] = useState<Brief[]>([]);
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, agentsRes, missionsRes, briefsRes] = await Promise.all([
        fetch('/api/v1/dashboard/stats').catch(() => null),
        fetch('/api/v1/agents').catch(() => null),
        fetch('/api/v1/missions').catch(() => null),
        fetch('/api/v1/briefs?limit=5').catch(() => null)
      ]);

      if (statsRes?.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      } else {
        // Mock data for development
        setStats({
          activeAgents: 12,
          activeMissions: 3,
          eventsToday: 847,
          briefsGenerated: 23,
          totalSaved: 15420,
          successRate: 94.2,
          protectedPositions: 5
        });
      }

      if (agentsRes?.ok) {
        const agentsData = await agentsRes.json();
        setAgents(agentsData.slice(0, 6));
      } else {
        // Mock agents data
        setAgents([
          { id: '1', name: 'Scout Alpha', type: 'Market Scanner', status: 'active', mission: 'DeFi Monitoring', lastActivity: '2 min ago', performance: 95 },
          { id: '2', name: 'Guardian Beta', type: 'Risk Manager', status: 'active', mission: 'Portfolio Protection', lastActivity: '1 min ago', performance: 89 },
          { id: '3', name: 'Analyst Gamma', type: 'Data Processor', status: 'active', mission: 'Trend Analysis', lastActivity: '30s ago', performance: 92 }
        ]);
      }

      if (missionsRes?.ok) {
        const missionsData = await missionsRes.json();
        setMissions(missionsData.slice(0, 3));
      } else {
        // Mock missions data
        setMissions([
          { id: '1', name: 'DeFi Yield Farming Monitor', status: 'active', progress: 67, createdAt: '2024-01-15', agentCount: 4 },
          { id: '2', name: 'MEV Protection Analysis', status: 'active', progress: 85, createdAt: '2024-01-14', agentCount: 2 }
        ]);
      }

      if (briefsRes?.ok) {
        const briefsData = await briefsRes.json();
        setRecentBriefs(briefsData);
      } else {
        // Mock briefs data
        setRecentBriefs([
          { id: '1', title: 'High-yield opportunity detected in SOL staking', summary: 'New validator offering 8.5% APY with solid track record', impact: 'high', confidence: 87, timestamp: '5 min ago', source: 'Solana Network' },
          { id: '2', title: 'Unusual whale activity on BONK', summary: 'Large transfers detected, potential market movement incoming', impact: 'medium', confidence: 73, timestamp: '12 min ago', source: 'On-chain Analysis' }
        ]);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-300">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <Layout>
      <Head>
        <title>Dashboard - Prowzi</title>
        <meta name="description" content="Monitor and control your autonomous AI agents" />
      </Head>

      <div className="bg-slate-50 dark:bg-slate-900 min-h-screen">

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">Dashboard</h1>
            <p className="text-slate-600 dark:text-slate-400">Monitor and control your autonomous AI agents</p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="bg-white dark:bg-slate-800 rounded-lg p-6 border border-slate-200 dark:border-slate-700 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Active Agents</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">{stats.activeAgents}</p>
                </div>
                <Users className="h-8 w-8 text-blue-500" />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Active Missions</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">{stats.activeMissions}</p>
                </div>
                <Target className="h-8 w-8 text-green-500" />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Events Today</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">{stats.eventsToday.toLocaleString()}</p>
                </div>
                <Activity className="h-8 w-8 text-purple-500" />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Total Saved</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">${stats.totalSaved.toLocaleString()}</p>
                </div>
                <DollarSign className="h-8 w-8 text-yellow-500" />
              </div>
            </div>
          </div>

          {/* Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Active Agents */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
              <div className="p-6 border-b border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Active Agents</h2>
                  <Link href="/agents" className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 text-sm">
                    View All
                  </Link>
                </div>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {agents.map((agent) => (
                    <div key={agent.id} className="flex items-center justify-between p-4 bg-slate-700 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status).replace('text-', 'bg-')}`} />
                        <div>
                          <p className="text-white font-medium">{agent.name}</p>
                          <p className="text-slate-400 text-sm">{agent.type}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-slate-300 text-sm">{agent.performance}%</p>
                        <p className="text-slate-400 text-xs">{agent.lastActivity}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Briefs */}
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="p-6 border-b border-slate-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-white">Recent Intelligence Briefs</h2>
                  <Link href="/briefs" className="text-blue-400 hover:text-blue-300 text-sm">
                    View All
                  </Link>
                </div>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {recentBriefs.map((brief) => (
                    <div key={brief.id} className="p-4 bg-slate-700 rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="text-white font-medium text-sm">{brief.title}</h3>
                        <span className={`px-2 py-1 rounded text-xs text-white ${getImpactColor(brief.impact)}`}>
                          {brief.impact}
                        </span>
                      </div>
                      <p className="text-slate-400 text-sm mb-2">{brief.summary}</p>
                      <div className="flex items-center justify-between text-xs text-slate-500">
                        <span>{brief.source}</span>
                        <span>{brief.timestamp}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Active Missions */}
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="p-6 border-b border-slate-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-white">Active Missions</h2>
                  <Link href="/missions" className="text-blue-400 hover:text-blue-300 text-sm">
                    View All
                  </Link>
                </div>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {missions.map((mission) => (
                    <div key={mission.id} className="p-4 bg-slate-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-white font-medium">{mission.name}</h3>
                        <span className="text-blue-400 text-sm">{mission.progress}%</span>
                      </div>
                      <div className="w-full bg-slate-600 rounded-full h-2 mb-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${mission.progress}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span>{mission.agentCount} agents</span>
                        <span>Started {mission.createdAt}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* System Health */}
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="p-6 border-b border-slate-700">
                <h2 className="text-xl font-semibold text-white">System Health</h2>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      <span className="text-white">Agent Runtime</span>
                    </div>
                    <span className="text-green-400 text-sm">Healthy</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      <span className="text-white">Gateway</span>
                    </div>
                    <span className="text-green-400 text-sm">Healthy</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Shield className="h-5 w-5 text-green-500" />
                      <span className="text-white">Protection Systems</span>
                    </div>
                    <span className="text-green-400 text-sm">{stats.protectedPositions} Active</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5 text-green-500" />
                      <span className="text-white">Success Rate</span>
                    </div>
                    <span className="text-green-400 text-sm">{stats.successRate}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Link href="/missions/create" className="bg-blue-600 hover:bg-blue-700 rounded-lg p-4 text-center transition-colors">
              <Target className="h-8 w-8 text-white mx-auto mb-2" />
              <p className="text-white font-medium">Create Mission</p>
            </Link>
            
            <Link href="/agents" className="bg-green-600 hover:bg-green-700 rounded-lg p-4 text-center transition-colors">
              <Users className="h-8 w-8 text-white mx-auto mb-2" />
              <p className="text-white font-medium">Manage Agents</p>
            </Link>
            
            <Link href="/protection" className="bg-purple-600 hover:bg-purple-700 rounded-lg p-4 text-center transition-colors">
              <Shield className="h-8 w-8 text-white mx-auto mb-2" />
              <p className="text-white font-medium">Protection Center</p>
            </Link>
            
            <Link href="/settings" className="bg-gray-600 hover:bg-gray-700 rounded-lg p-4 text-center transition-colors">
              <Settings className="h-8 w-8 text-white mx-auto mb-2" />
              <p className="text-white font-medium">Settings</p>
            </Link>
          </div>
        </main>
      </div>
    </>
  );
}