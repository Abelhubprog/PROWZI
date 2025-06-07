use rustyline::Editor;
use std::sync::Arc;

pub struct DebugConsole {
    orchestrator: Arc<Orchestrator>,
    tracer: Arc<DistributedTracer>,
    state_inspector: Arc<StateInspector>,
}

impl DebugConsole {
    pub async fn start_interactive_session(&self) {
        let mut rl = Editor::<()>::new();

        println!("Prowzi Debug Console v1.0");
        println!("Type 'help' for available commands\n");

        loop {
            let readline = rl.readline("prowzi> ");

            match readline {
                Ok(line) => {
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();

                    if parts.is_empty() {
                        continue;
                    }

                    match parts[0] {
                        "help" => self.print_help(),
                        "mission" => self.handle_mission_command(&parts[1..]).await,
                        "agent" => self.handle_agent_command(&parts[1..]).await,
                        "trace" => self.handle_trace_command(&parts[1..]).await,
                        "perf" => self.handle_performance_command(&parts[1..]).await,
                        "state" => self.handle_state_command(&parts[1..]).await,
                        "replay" => self.handle_replay_command(&parts[1..]).await,
                        "exit" => break,
                        _ => println!("Unknown command. Type 'help' for available commands."),
                    }
                }
                Err(_) => break,
            }
        }
    }

    async fn handle_trace_command(&self, args: &[&str]) {
        if args.is_empty() {
            println!("Usage: trace <event_id|mission_id>");
            return;
        }

        let id = args[0];

        // Fetch trace data
        let trace = self.tracer.get_trace(id).await;

        match trace {
            Ok(trace_data) => {
                // Print trace tree
                println!("\nTrace for {}", id);
                println!("{}", "-".repeat(80));

                self.print_trace_tree(&trace_data.root_span, 0);

                // Print summary
                println!("\nSummary:");
                println!("  Total Duration: {:.2}ms", trace_data.total_duration_ms);
                println!("  Span Count: {}", trace_data.span_count);
                println!("  Error Count: {}", trace_data.error_count);

                // Critical path
                println!("\nCritical Path:");
                for (i, span) in trace_data.critical_path.iter().enumerate() {
                    println!("  {}. {} ({:.2}ms)", i + 1, span.name, span.duration_ms);
                }
            }
            Err(e) => println!("Error fetching trace: {}", e),
        }
    }

    async fn handle_state_command(&self, args: &[&str]) {
        if args.len() < 2 {
            println!("Usage: state <inspect|dump|diff> <entity_id>");
            return;
        }

        let action = args[0];
        let entity_id = args[1];

        match action {
            "inspect" => {
                let state = self.state_inspector.inspect_entity(entity_id).await;

                match state {
                    Ok(state_data) => {
                        println!("\nState for {}", entity_id);
                        println!("{}", "-".repeat(80));
                        println!("{}", serde_json::to_string_pretty(&state_data).unwrap());
                    }
                    Err(e) => println!("Error inspecting state: {}", e),
                }
            }
            "dump" => {
                let filename = format!("{}_state_{}.json", entity_id, chrono::Utc::now().timestamp());

                match self.state_inspector.dump_to_file(entity_id, &filename).await {
                    Ok(_) => println!("State dumped to {}", filename),
                    Err(e) => println!("Error dumping state: {}", e),
                }
            }
            "diff" => {
                if args.len() < 3 {
                    println!("Usage: state diff <entity_id> <timestamp>");
                    return;
                }

                let timestamp = args[2].parse::<i64>().unwrap_or(0);
                let diff = self.state_inspector.diff_from_timestamp(entity_id, timestamp).await;

                match diff {
                    Ok(changes) => {
                        println!("\nState changes since {}", timestamp);
                        for change in changes {
                            println!("  {} {} = {}", 
                                change.operation, 
                                change.field, 
                                change.value
                            );
                        }
                    }
                    Err(e) => println!("Error computing diff: {}", e),
                }
            }
            _ => println!("Unknown state command"),
        }
    }

    async fn handle_replay_command(&self, args: &[&str]) {
        if args.is_empty() {
            println!("Usage: replay <event_id> [--speed <1-10>] [--breakpoint <operation>]");
            return;
        }

        let event_id = args[0];
        let mut speed = 1.0;
        let mut breakpoint = None;

        // Parse options
        let mut i = 1;
        while i < args.len() {
            match args[i] {
                "--speed" => {
                    if i + 1 < args.len() {
                        speed = args[i + 1].parse().unwrap_or(1.0);
                    }
                    i += 2;
                }
                "--breakpoint" => {
                    if i + 1 < args.len() {
                        breakpoint = Some(args[i + 1].to_string());
                    }
                    i += 2;
                }
                _ => i += 1,
            }
        }

        println!("Replaying event {} at {}x speed", event_id, speed);

        // Start replay
        let replay_stream = self.state_inspector.replay_event(event_id, speed).await;

        match replay_stream {
            Ok(mut stream) => {
                while let Some(step) = stream.next().await {
                    // Print step
                    println!("[{:.3}s] {} - {}", 
                        step.elapsed_seconds,
                        step.operation,
                        step.description
                    );

                    // Check breakpoint
                    if let Some(ref bp) = breakpoint {
                        if step.operation.contains(bp) {
                            println!("\nBreakpoint hit! Press Enter to continue...");
                            let mut input = String::new();
                            std::io::stdin().read_line(&mut input).unwrap();
                        }
                    }

                    // Add delay based on speed
                    if speed < 10.0 {
                        tokio::time::sleep(Duration::from_millis(
                            (step.duration_ms / speed) as u64
                        )).await;
                    }
                }

                println!("\nReplay complete!");
            }
            Err(e) => println!("Error starting replay: {}", e),
        }
    }
}
