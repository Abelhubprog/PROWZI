//scripts/production-readiness-check.ts
import { execSync } from 'child_process'
import * as fs from 'fs'

interface CheckResult {
  name: string
  passed: boolean
  details?: string
}

class ProductionReadinessChecker {
  private checks: CheckResult[] = []

  async runAllChecks(): Promise<void> {
    console.log('🚀 Running Production Readiness Checks...\n')

    // Security checks
    await this.checkAuthentication()
    await this.checkRLS()
    await this.checkWebhookSigning()
    await this.checkPromptGuard()

    // Operational checks
    await this.checkDisasterRecovery()
    await this.checkPlanHotSwap()
    await this.checkFinOpsAlerts()
    await this.checkChaosResilience()

    // Compliance checks
    await this.checkGDPRCompliance()
    await this.checkSOC2Controls()
    await this.checkSBOM()

    // Performance checks
    await this.checkLatencySLO()
    await this.checkCostPerUser()

    // Display results
    this.displayResults()
  }

  private async checkAuthentication(): Promise<void> {
    try {
      // Test JWT generation
      const authTest = execSync('npm run test:auth', { encoding: 'utf8' })

      // Test wallet signatures
      const walletTest = execSync('npm run test:wallet-auth', { encoding: 'utf8' })

      this.checks.push({
        name: 'Authentication & JWT',
        passed: true,
        details: 'JWT issuance and wallet auth working',
      })
    } catch (error) {
      this.checks.push({
        name: 'Authentication & JWT',
        passed: false,
        details: error.message,
      })
    }
  }

  private async checkRLS(): Promise<void> {
    try {
      const result = execSync(
        'psql $DATABASE_URL -c "SELECT * FROM prowzi.test_tenant_isolation(\'tenant-a\', \'tenant-b\')"',
        { encoding: 'utf8' }
      )

      const passed = result.includes('true') && !result.includes('false')

      this.checks.push({
        name: 'PostgreSQL RLS',
        passed,
        details: passed ? 'Tenant isolation verified' : 'RLS test failed',
      })
    } catch (error) {
      this.checks.push({
        name: 'PostgreSQL RLS',
        passed: false,
        details: error.message,
      })
    }
  }

  private async checkLatencySLO(): Promise<void> {
    try {
      const metrics = await this.queryPrometheus(
        'histogram_quantile(0.99, prowzi_brief_generation_seconds_bucket[1h])'
      )

      const p99Latency = parseFloat(metrics.data.result[0].value[1])
      const passed = p99Latency < 1.0 // Under 1 second

      this.checks.push({
        name: 'Latency SLO (P99 < 1s)',
        passed,
        details: `Current P99: ${p99Latency.toFixed(3)}s`,
      })
    } catch (error) {
      this.checks.push({
        name: 'Latency SLO (P99 < 1s)',
        passed: false,
        details: error.message,
      })
    }
  }

  private async checkCostPerUser(): Promise<void> {
    try {
      const metrics = await this.queryPrometheus(
        'sum(prowzi_user_cost_dollars) by (user_id)'
      )

      const maxCost = Math.max(...metrics.data.result.map(r => parseFloat(r.value[1])))
      const passed = maxCost <= 2.0

      this.checks.push({
        name: 'Cost per User (≤$2/month)',
        passed,
        details: `Max user cost: $${maxCost.toFixed(2)}`,
      })
    } catch (error) {
      this.checks.push({
        name: 'Cost per User (≤$2/month)',
        passed: false,
        details: error.message,
      })
    }
  }

  private displayResults(): void {
    console.log('\n📊 Production Readiness Results:\n')

    const passed = this.checks.filter(c => c.passed).length
    const total = this.checks.length
    const percentage = (passed / total * 100).toFixed(1)

    for (const check of this.checks) {
      const icon = check.passed ? '✅' : '❌'
      console.log(`${icon} ${check.name}`)
      if (check.details) {
        console.log(`   ${check.details}`)
      }
    }

    console.log(`\n📈 Overall Score: ${passed}/${total} (${percentage}%)`)

    if (passed === total) {
      console.log('\n🎉 All checks passed! Prowzi is ready for production.')
    } else {
      console.log('\n⚠️  Some checks failed. Please address issues before production deployment.')
      process.exit(1)
    }
  }
}

// Run checks
const checker = new ProductionReadinessChecker()
checker.runAllChecks().catch(console.error)
```

### 10. Deployment Script

```bash
#!/bin/bash
# deploy-production.sh

set -euo pipefail

echo "🚀 Deploying Prowzi to Production"

# Run production readiness checks
echo "📋 Running readiness checks..."
npm run check:production-ready

# Build and sign all images
echo "🔨 Building and signing images..."
make build-all-images
make sign-images

# Generate SBOM
echo "📦 Generating SBOM..."
make generate-sbom

# Run security scan
echo "🔍 Running security scan..."
make security-scan

# Deploy with Flux
echo "🚢 Deploying via GitOps..."
git tag -a "v1.0.0-alpha" -m "Private alpha release"
git push origin v1.0.0-alpha

# Wait for Flux sync
echo "⏳ Waiting for Flux sync..."
flux reconcile kustomization prowzi-production --with-source

# Run smoke tests
echo "🧪 Running smoke tests..."
npm run test:smoke

# Update status page
echo "📊 Updating status page..."
curl -X POST https://api.statuspage.io/v1/pages/${PAGE_ID}/incidents \
  -H "Authorization: OAuth ${STATUSPAGE_TOKEN}" \
  -d '{
    "incident": {
      "name": "Prowzi v1.0.0-alpha deployed",
      "status": "resolved",
      "impact": "none",
      "component_ids": ["all"]
    }
  }'

echo "✅ Deployment complete!"
echo "🔗 Access at: https://api.prowzi.io"
echo "📊 Monitoring: https://grafana.prowzi.io"
