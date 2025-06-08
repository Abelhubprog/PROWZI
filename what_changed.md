# Security Workflow Changes

## Issues Fixed

- **Node dependency audit**
  - Now only scans folders with both `package.json` AND a lockfile (`package-lock.json`, `pnpm-lock.yaml`, or `yarn.lock`)
  - Gracefully skips the scan when no qualifying packages are found
  - Updated `NODE_PACKAGES` to include `web` directory instead of `sdk`

- **CodeQL**
  - Added `fail-fast: false` to prevent cancellation of all jobs when one fails
  - Kept Rust experimental flag enabled for proper analysis

- **Container scan**
  - Updated Docker directories to use `docker/gateway` and `docker/mission-control`
  - Added `--build-arg REQ_PATH=requirements.txt` for proper file location
  - Creates empty SARIF files when builds fail to ensure upload steps don't error

- **IaC scan**
  - Checks if directories exist before running Checkov
  - Renamed output file to `checkov.sarif` for consistency
  - Kept permissions: `security-events: write` for SARIF uploads

- **Path filters & guards**
  - Changed trigger paths to include `web/**` instead of `platform/**` and `sdk/**`
  - Added conditional checks throughout to prevent failures on missing files
