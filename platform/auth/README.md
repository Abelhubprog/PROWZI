# Prowzi Authentication Service

A secure, multi-tenant authentication service for the Prowzi platform supporting wallet-based authentication (Ethereum & Solana), JWT tokens, and row-level security.

## Features

- 🔐 **Wallet Authentication**: Support for Ethereum (SIWE) and Solana wallets
- 🏢 **Multi-Tenant Architecture**: Complete tenant isolation with Row-Level Security (RLS)
- 🎫 **JWT Tokens**: Secure access and refresh token management
- 🚀 **High Performance**: Built with Rust/Axum for maximum performance
- 🔒 **Security First**: Rate limiting, token revocation, secure key management
- 📊 **Observability**: Comprehensive logging and metrics
- 🌐 **Standards Compliant**: OAuth 2.0 token introspection, JWKS endpoint

## Quick Start

### Prerequisites

- Rust 1.70+ ([Install Rust](https://rustup.rs/))
- PostgreSQL 14+
- Redis 6+
- OpenSSL (for key generation)

### Installation

1. **Clone and navigate to the auth service:**
   ```bash
   cd platform/auth
   ```

2. **Run the setup script:**
   ```bash
   # Linux/macOS
   chmod +x setup.sh
   ./setup.sh development

   # Windows (PowerShell as Administrator)
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\setup.ps1 -Environment development
   ```

3. **Configure environment:**
   ```bash
   # Edit the generated .env file
   nano .env
   
   # Update these critical values:
   DATABASE_URL=postgresql://user:pass@localhost:5432/prowzi
   REDIS_URL=redis://localhost:6379
   ```

4. **Run database migrations:**
   ```bash
   # From project root
   sqlx migrate run --database-url $DATABASE_URL
   ```

5. **Start the service:**
   ```bash
   cargo run
   ```

The authentication service will be available at `http://localhost:3001`.

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/auth/nonce` | Generate nonce for wallet authentication |
| `POST` | `/auth/wallet` | Authenticate with wallet signature |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/auth/introspect` | Introspect token validity |
| `POST` | `/auth/revoke` | Revoke token |

### Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/.well-known/jwks.json` | JSON Web Key Set |
| `GET` | `/health` | Health check |

### Example: Wallet Authentication

1. **Get nonce:**
   ```bash
   curl -X GET "http://localhost:3001/auth/nonce?address=0x742d35Cc6610C7532C8"
   ```

2. **Sign message with wallet and authenticate:**
   ```bash
   curl -X POST "http://localhost:3001/auth/wallet" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "ethereum",
       "address": "0x742d35Cc6610C7532C8",
       "message": "Sign in to Prowzi...",
       "signature": "0x1234...",
       "tenant_id": "my-org"
     }'
   ```

3. **Use access token:**
   ```bash
   curl -X GET "http://localhost:3001/api/protected" \
     -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
   ```

## SDK Usage

### TypeScript/JavaScript

```bash
npm install @prowzi/auth-sdk
```

```typescript
import ProwziAuth from '@prowzi/auth-sdk';

const auth = new ProwziAuth({
  apiUrl: 'http://localhost:3001',
  tenantId: 'my-org'
});

// Ethereum authentication
const tokens = await auth.authenticateEthereum(signer);

// Solana authentication
const tokens = await auth.authenticateSolana(wallet);

// Use tokens for API calls
const authFetch = createAuthenticatedFetch(auth);
const response = await authFetch('/api/missions');
```

### Rust

```toml
[dependencies]
prowzi-auth = { path = "../platform/auth" }
```

```rust
use prowzi_auth::{AuthService, AuthConfig, middleware::auth_middleware};

// Initialize auth service
let config = AuthConfig::from_env()?;
let auth_service = Arc::new(AuthService::new(config).await?);

// Use in Axum router
let app = Router::new()
    .route("/protected", get(protected_handler))
    .layer(axum::middleware::from_fn_with_state(
        auth_service.clone(),
        auth_middleware,
    ));
```

## Architecture

### Multi-Tenant Security

The service implements comprehensive tenant isolation:

- **Row-Level Security (RLS)**: Database-level tenant isolation
- **JWT Claims**: Tenant context in every token
- **Middleware**: Automatic tenant context setting
- **API Scoping**: All queries automatically scoped to tenant

### Token Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Auth Service
    participant D as Database
    participant R as Redis

    C->>A: Wallet signature
    A->>A: Verify signature
    A->>D: Get/create user
    A->>A: Generate tokens
    A->>R: Store access token
    A->>C: Return tokens
    
    Note over C,A: Later API calls
    C->>A: API call with token
    A->>R: Check if revoked
    A->>A: Validate token
    A->>D: Set tenant context
    A->>C: Process request
```

### Database Schema

Key tables with RLS enabled:

- `users` - User accounts with tenant isolation
- `tenants` - Tenant configuration
- `user_sessions` - Token tracking and revocation
- `missions`, `briefs`, `events` - Business data (tenant-scoped)

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | ✅ | - |
| `REDIS_URL` | Redis connection string | ❌ | `redis://localhost:6379` |
| `JWT_PRIVATE_KEY` | RSA private key for JWT signing | ✅ | - |
| `JWT_PUBLIC_KEY` | RSA public key for JWT verification | ✅ | - |
| `REFRESH_PRIVATE_KEY` | RSA private key for refresh tokens | ✅ | - |
| `REFRESH_PUBLIC_KEY` | RSA public key for refresh tokens | ✅ | - |
| `JWT_EXPIRY_SECONDS` | Access token expiry | ❌ | `3600` (1 hour) |
| `REFRESH_EXPIRY_SECONDS` | Refresh token expiry | ❌ | `604800` (7 days) |
| `RATE_LIMIT_REQUESTS` | Rate limit per window | ❌ | `100` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window | ❌ | `60` |
| `CORS_ORIGINS` | Allowed CORS origins | ❌ | Environment-dependent |
| `ENVIRONMENT` | Deployment environment | ❌ | `development` |

### Key Generation

Generate RSA key pairs for JWT signing:

```bash
# JWT keys
openssl genrsa -out jwt_private.pem 2048
openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

# Refresh token keys  
openssl genrsa -out refresh_private.pem 2048
openssl rsa -in refresh_private.pem -pubout -out refresh_public.pem
```

## Deployment

### Development

```bash
cargo run
```

### Production

1. **Build release binary:**
   ```bash
   cargo build --release
   ```

2. **Set up systemd service (Linux):**
   ```bash
   sudo cp prowzi-auth.service /etc/systemd/system/
   sudo systemctl enable prowzi-auth
   sudo systemctl start prowzi-auth
   ```

3. **Set up Windows service:**
   ```powershell
   .\install-service.ps1
   ```

### Docker

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/prowzi-auth /usr/local/bin/
EXPOSE 3001
CMD ["prowzi-auth"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prowzi-auth
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prowzi-auth
  template:
    metadata:
      labels:
        app: prowzi-auth
    spec:
      containers:
      - name: prowzi-auth
        image: prowzi/auth:latest
        ports:
        - containerPort: 3001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: prowzi-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## Security Considerations

### Production Checklist

- [ ] Use strong RSA keys (2048+ bits)
- [ ] Store private keys securely (Vault/HSM)
- [ ] Enable TLS/HTTPS in production
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting and DDoS protection
- [ ] Monitor for security events
- [ ] Regular security audits
- [ ] Implement key rotation
- [ ] Use secure Redis configuration
- [ ] Enable database connection encryption

### Rate Limiting

The service implements multiple layers of rate limiting:

- **IP-based**: 100 requests per minute per IP
- **User-based**: 1000 requests per hour per user
- **Endpoint-specific**: Special limits for auth endpoints

### Token Security

- **Short-lived access tokens**: 1 hour default expiry
- **Sliding refresh tokens**: 7 day expiry with renewal
- **Token revocation**: Immediate invalidation via Redis
- **Secure storage**: httpOnly cookies for web clients

## Monitoring and Observability

### Metrics

The service exposes Prometheus metrics:

- `prowzi_auth_requests_total` - Total requests by endpoint
- `prowzi_auth_request_duration_seconds` - Request duration
- `prowzi_auth_active_tokens` - Number of active tokens
- `prowzi_auth_failed_authentications` - Failed auth attempts

### Logging

Structured logging with configurable levels:

```bash
RUST_LOG=prowzi_auth=debug,tower_http=info
```

### Health Checks

- **Liveness**: `GET /health`
- **Readiness**: Checks database and Redis connectivity
- **Dependencies**: PostgreSQL and Redis status

## Troubleshooting

### Common Issues

1. **"Invalid JWT key" errors:**
   - Verify RSA keys are properly formatted
   - Check for newline characters in environment variables
   - Ensure keys match (private/public pairs)

2. **Database connection errors:**
   - Verify DATABASE_URL is correct
   - Check PostgreSQL is running and accessible
   - Ensure database exists and migrations are applied

3. **Redis connection errors:**
   - Verify REDIS_URL is correct
   - Check Redis is running and accessible
   - Test connection with `redis-cli ping`

4. **Authentication failures:**
   - Verify wallet signature format
   - Check message format for SIWE compliance
   - Ensure clock synchronization for token expiry

### Debug Mode

Enable debug logging for troubleshooting:

```bash
RUST_LOG=debug cargo run
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests: `cargo test`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: engineering@prowzi.com
- 🐛 Issues: [GitHub Issues](https://github.com/prowzi/platform/issues)
- 📖 Documentation: [Prowzi Docs](https://docs.prowzi.io)
- 💬 Discord: [Prowzi Community](https://discord.gg/prowzi)
