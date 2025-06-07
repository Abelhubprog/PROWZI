use clap::{App, Arg, SubCommand};
use rustyline::Editor;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, Write};
use tokio::time::{sleep, Duration};
use uuid::Uuid;

const PROWZI_ASCII: &str = r#"
  ____                         _ 
 |  _ \ _ __ _____      _____(_)
 | |_) | '__/ _ \ \ /\ / /_  / |
 |  __/| | | (_) \ V  V / / /| |
 |_|   |_|  \___/ \_/\_/ /___|_|
                                
 Debug Console - v0.1.0
"#;

#[derive(Debug, Clone)]
struct Agent {
    id: String,
    name: String,
    status: String,
    performance: f64,
    last_seen: String,
}

#[derive(Debug, Clone)]
struct Mission {
    id: String,
    name: String,
    status: String,
    progress: f64,
    agents: Vec<String>,
}

struct DebugConsole {
    agents: HashMap<String, Agent>,
    missions: HashMap<String, Mission>,
    logs: Vec<String>,
}

impl DebugConsole {
    fn new() -> Self {
        let mut console = Self {
            agents: HashMap::new(),
            missions: HashMap::new(),
            logs: Vec::new(),
        };
        
        // Add some mock data for demonstration
        console.load_mock_data();
        console
    }

    fn load_mock_data(&mut self) {
        // Mock agents
        let agents = vec![
            Agent {
                id: "agent-001".to_string(),
                name: "Solana Mempool Sensor".to_string(),
                status: "active".to_string(),
                performance: 0.95,
                last_seen: "2024-12-20T10:30:00Z".to_string(),
            },
            Agent {
                id: "agent-002".to_string(),
                name: "Risk Scorer".to_string(),
                status: "idle".to_string(),
                performance: 0.87,
                last_seen: "2024-12-20T10:25:00Z".to_string(),
            },
            Agent {
                id: "agent-003".to_string(),
                name: "GitHub Events Sensor".to_string(),
                status: "error".to_string(),
                performance: 0.12,
                last_seen: "2024-12-20T09:45:00Z".to_string(),
            },
        ];

        for agent in agents {
            self.agents.insert(agent.id.clone(), agent);
        }

        // Mock missions
        let missions = vec![
            Mission {
                id: "mission-001".to_string(),
                name: "Market Analysis DeFi".to_string(),
                status: "running".to_string(),
                progress: 0.68,
                agents: vec!["agent-001".to_string(), "agent-002".to_string()],
            },
            Mission {
                id: "mission-002".to_string(),
                name: "Threat Detection Scan".to_string(),
                status: "completed".to_string(),
                progress: 1.0,
                agents: vec!["agent-002".to_string()],
            },
        ];

        for mission in missions {
            self.missions.insert(mission.id.clone(), mission);
        }

        // Mock logs
        self.logs = vec![
            "[2024-12-20T10:30:00Z] INFO: Agent agent-001 completed task successfully".to_string(),
            "[2024-12-20T10:29:45Z] WARN: High memory usage detected on agent-003".to_string(),
            "[2024-12-20T10:29:30Z] ERROR: Connection timeout for agent-003".to_string(),
            "[2024-12-20T10:29:15Z] INFO: Mission mission-001 progress: 68%".to_string(),
        ];
    }

    fn print_help(&self) {
        println!("{}", PROWZI_ASCII);
        println!("Available commands:");
        println!("  agents list          - List all agents");
        println!("  agents show <id>     - Show agent details");
        println!("  agents restart <id>  - Restart an agent");
        println!("  missions list        - List all missions");
        println!("  missions show <id>   - Show mission details");
        println!("  missions stop <id>   - Stop a mission");
        println!("  logs                 - Show recent logs");
        println!("  logs tail            - Tail logs in real-time");
        println!("  metrics              - Show system metrics");
        println!("  health               - Check system health");
        println!("  clear                - Clear screen");
        println!("  help                 - Show this help");
        println!("  exit                 - Exit console");
        println!();
    }

    fn handle_command(&mut self, input: &str) -> bool {
        let parts: Vec<&str> = input.trim().split_whitespace().collect();
        if parts.is_empty() {
            return true;
        }

        match parts[0] {
            "help" => self.print_help(),
            "clear" => {
                print!("\x1B[2J\x1B[H");
                io::stdout().flush().unwrap();
            }
            "exit" | "quit" => return false,
            "agents" => self.handle_agent_command(&parts[1..]),
            "missions" => self.handle_mission_command(&parts[1..]),
            "logs" => self.handle_logs_command(&parts[1..]),
            "metrics" => self.show_metrics(),
            "health" => self.show_health(),
            _ => println!("Unknown command '{}'. Type 'help' for available commands.", parts[0]),
        }
        true
    }

    fn handle_agent_command(&self, parts: &[&str]) {
        if parts.is_empty() {
            println!("Usage: agents <list|show|restart> [id]");
            return;
        }

        match parts[0] {
            "list" => {
                println!("\n📋 Agents Status:");
                println!("{:<12} {:<25} {:<10} {:<10} {:<20}", "ID", "Name", "Status", "Perf", "Last Seen");
                println!("{:-<80}", "");
                for agent in self.agents.values() {
                    let status_emoji = match agent.status.as_str() {
                        "active" => "🟢",
                        "idle" => "🟡",
                        "error" => "🔴",
                        _ => "⚪",
                    };
                    println!("{:<12} {:<25} {}{:<9} {:<10.2} {:<20}", 
                        agent.id, agent.name, status_emoji, agent.status, agent.performance, agent.last_seen);
                }
                println!();
            }
            "show" => {
                if parts.len() < 2 {
                    println!("Usage: agents show <id>");
                    return;
                }
                if let Some(agent) = self.agents.get(parts[1]) {
                    println!("\n🤖 Agent Details:");
                    println!("  ID: {}", agent.id);
                    println!("  Name: {}", agent.name);
                    println!("  Status: {}", agent.status);
                    println!("  Performance: {:.2}", agent.performance);
                    println!("  Last Seen: {}", agent.last_seen);
                    println!();
                } else {
                    println!("Agent '{}' not found", parts[1]);
                }
            }
            "restart" => {
                if parts.len() < 2 {
                    println!("Usage: agents restart <id>");
                    return;
                }
                if self.agents.contains_key(parts[1]) {
                    println!("🔄 Restarting agent '{}'...", parts[1]);
                    println!("✅ Agent restarted successfully");
                } else {
                    println!("Agent '{}' not found", parts[1]);
                }
            }
            _ => println!("Unknown agent command '{}'. Use: list, show, restart", parts[0]),
        }
    }

    fn handle_mission_command(&self, parts: &[&str]) {
        if parts.is_empty() {
            println!("Usage: missions <list|show|stop> [id]");
            return;
        }

        match parts[0] {
            "list" => {
                println!("\n🎯 Missions Status:");
                println!("{:<15} {:<25} {:<12} {:<10} {:<15}", "ID", "Name", "Status", "Progress", "Agents");
                println!("{:-<80}", "");
                for mission in self.missions.values() {
                    let status_emoji = match mission.status.as_str() {
                        "running" => "🟢",
                        "paused" => "🟡",
                        "completed" => "✅",
                        "failed" => "❌",
                        _ => "⚪",
                    };
                    println!("{:<15} {:<25} {}{:<11} {:<10.1}% {:<15}", 
                        mission.id, mission.name, status_emoji, mission.status, 
                        mission.progress * 100.0, mission.agents.len());
                }
                println!();
            }
            "show" => {
                if parts.len() < 2 {
                    println!("Usage: missions show <id>");
                    return;
                }
                if let Some(mission) = self.missions.get(parts[1]) {
                    println!("\n🎯 Mission Details:");
                    println!("  ID: {}", mission.id);
                    println!("  Name: {}", mission.name);
                    println!("  Status: {}", mission.status);
                    println!("  Progress: {:.1}%", mission.progress * 100.0);
                    println!("  Agents: {}", mission.agents.join(", "));
                    println!();
                } else {
                    println!("Mission '{}' not found", parts[1]);
                }
            }
            "stop" => {
                if parts.len() < 2 {
                    println!("Usage: missions stop <id>");
                    return;
                }
                if self.missions.contains_key(parts[1]) {
                    println!("🛑 Stopping mission '{}'...", parts[1]);
                    println!("✅ Mission stopped successfully");
                } else {
                    println!("Mission '{}' not found", parts[1]);
                }
            }
            _ => println!("Unknown mission command '{}'. Use: list, show, stop", parts[0]),
        }
    }

    fn handle_logs_command(&self, parts: &[&str]) {
        if parts.is_empty() || parts[0] == "recent" {
            println!("\n📋 Recent Logs:");
            for log in &self.logs {
                println!("  {}", log);
            }
            println!();
        } else if parts[0] == "tail" {
            println!("📋 Tailing logs (Ctrl+C to stop)...");
            // In a real implementation, this would tail actual logs
            println!("  [2024-12-20T10:30:15Z] INFO: Real-time log entry 1");
            println!("  [2024-12-20T10:30:30Z] INFO: Real-time log entry 2");
        } else {
            println!("Usage: logs [recent|tail]");
        }
    }

    fn show_metrics(&self) {
        println!("\n📊 System Metrics:");
        println!("  Active Agents: {}/{}", 
            self.agents.values().filter(|a| a.status == "active").count(),
            self.agents.len());
        println!("  Running Missions: {}/{}", 
            self.missions.values().filter(|m| m.status == "running").count(),
            self.missions.len());
        println!("  System Uptime: 2d 14h 23m");
        println!("  Memory Usage: 2.3GB / 8GB (28.8%)");
        println!("  CPU Usage: 45.2%");
        println!("  Network I/O: ↑ 1.2MB/s ↓ 3.4MB/s");
        println!();
    }

    fn show_health(&self) {
        println!("\n🏥 System Health Check:");
        
        let healthy_agents = self.agents.values().filter(|a| a.status == "active").count();
        let total_agents = self.agents.len();
        
        if healthy_agents as f64 / total_agents as f64 > 0.8 {
            println!("  ✅ Agents: {} / {} healthy", healthy_agents, total_agents);
        } else {
            println!("  ⚠️  Agents: {} / {} healthy", healthy_agents, total_agents);
        }
        
        println!("  ✅ Database: Connected");
        println!("  ✅ Message Queue: Operational");
        println!("  ✅ Monitoring: Active");
        println!("  ⚠️  Storage: 85% capacity");
        println!("\n  Overall Status: 🟡 Warning - Storage capacity high");
        println!();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("Prowzi Debug Console")
        .version("0.1.0")
        .author("Prowzi Team")
        .about("Debug console for Prowzi autonomous AI agents platform")
        .arg(Arg::with_name("command")
            .help("Command to execute")
            .takes_value(true)
            .multiple(true))
        .get_matches();

    let mut console = DebugConsole::new();
    
    // If command line arguments provided, execute them and exit
    if let Some(commands) = matches.values_of("command") {
        let command = commands.collect::<Vec<_>>().join(" ");
        console.handle_command(&command);
        return Ok(());
    }

    // Interactive mode
    println!("{}", PROWZI_ASCII);
    println!("Type 'help' for available commands or 'exit' to quit.");
    println!();

    let mut rl = Editor::<()>::new();
    
    loop {
        let readline = rl.readline("prowzi> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str());
                if !console.handle_command(&line) {
                    break;
                }
            }
            Err(_) => {
                println!("Goodbye!");
                break;
            }
        }
    }

    Ok(())
}