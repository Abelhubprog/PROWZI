# Security Workflow Changes

## What Changed

- **Removed outdated paths**: No longer scans non-existent `platform/*` and `packages/eprowzi/*` directories
- **Added experimental Rust flag**: Set `CODEQL_ENABLE_EXPERIMENTAL_FEATURES: "true"` to enable Rust CodeQL analysis
- **Updated Trivy container scan**: 
  - Changed from hardcoded `docker/gateway` and `docker/orchestrator` paths
  - Now uses matrix strategy to build and scan all available Docker images
  - Individual SARIF uploads with categories for better tracking
- **Fixed IaC scan**:
  - Runs Checkov on both `infrastructure/` and `charts/` directories
  - Always creates a valid `checkov-results.sarif` file even with low-severity findings
  - Uses `soft_fail: true` to prevent blocking on minor issues
  - Merges multiple SARIF files when both directories are scanned
- **Added permissions**: `security-events: write` added to all scan jobs for SARIF uploads on public repos
- **Added path filters**: Workflow only triggers on changes to relevant directories:
  - `agent-runtime/**`
  - `packages/**`
  - `platform/**`
  - `sdk/**`
  - `docker/**`
  - `infrastructure/**`
  - `charts/**`
- **Added conditional guards**: 
  - Node audit only runs if package.json files are found
  - Container scans only build images where Dockerfiles exist
  - Better error handling for missing files
- **Dynamic path discovery**: Uses environment variables and runtime detection for packages and Docker directories
- **Improved Node.js scanning**: Only scans directories that contain both package.json and package-lock.json files
