use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct TenantIsolationManager {
    tenants: Arc<RwLock<HashMap<String, TenantContext>>>,
    key_manager: Arc<KeyManager>,
    network_policies: Arc<NetworkPolicyEnforcer>,
}

#[derive(Clone)]
struct TenantContext {
    id: String,
    tier: TenantTier,
    encryption_key: LessSafeKey,
    resource_limits: ResourceLimits,
    data_residency: DataResidency,
    isolation_level: IsolationLevel,
}

impl TenantIsolationManager {
    pub async fn create_tenant(
        &self,
        tenant_id: String,
        config: TenantConfig,
    ) -> Result<(), TenantError> {
        // Generate tenant-specific encryption key
        let master_key = self.key_manager.get_master_key().await?;
        let tenant_key = self.derive_tenant_key(&master_key, &tenant_id)?;

        // Create isolated namespace
        self.create_namespace(&tenant_id, &config).await?;

        // Set up network policies
        self.network_policies.create_tenant_policies(
            &tenant_id,
            &config.allowed_endpoints,
            config.isolation_level,
        ).await?;

        // Configure resource quotas
        self.set_resource_quotas(&tenant_id, &config.resource_limits).await?;

        // Create tenant context
        let context = TenantContext {
            id: tenant_id.clone(),
            tier: config.tier,
            encryption_key: tenant_key,
            resource_limits: config.resource_limits,
            data_residency: config.data_residency,
            isolation_level: config.isolation_level,
        };

        self.tenants.write().await.insert(tenant_id, context);

        Ok(())
    }

    pub async fn encrypt_tenant_data(
        &self,
        tenant_id: &str,
        data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        let tenants = self.tenants.read().await;
        let context = tenants.get(tenant_id)
            .ok_or(CryptoError::TenantNotFound)?;

        // Generate nonce
        let nonce = self.generate_nonce()?;

        // Encrypt with tenant key
        let mut encrypted = data.to_vec();
        context.encryption_key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(&[]),
            &mut encrypted,
        )?;

        // Prepend nonce
        let mut result = nonce.as_ref().to_vec();
        result.extend_from_slice(&encrypted);

        Ok(result)
    }

    pub async fn validate_cross_tenant_access(
        &self,
        source_tenant: &str,
        target_tenant: &str,
        operation: &str,
    ) -> Result<bool, AccessError> {
        // Check if cross-tenant access is allowed
        let tenants = self.tenants.read().await;

        let source = tenants.get(source_tenant)
            .ok_or(AccessError::TenantNotFound)?;
        let target = tenants.get(target_tenant)
            .ok_or(AccessError::TenantNotFound)?;

        // Enterprise tier can access shared resources
        if source.tier == TenantTier::Enterprise 
            && target.tier == TenantTier::Shared 
            && operation == "read" {
            return Ok(true);
        }

        // No cross-tenant access by default
        Ok(false)
    }

    async fn create_namespace(
        &self,
        tenant_id: &str,
        config: &TenantConfig,
    ) -> Result<(), NamespaceError> {
        // Create Nomad namespace
        let nomad_namespace = format!("prowzi-tenant-{}", tenant_id);
        self.nomad_client.create_namespace(&nomad_namespace).await?;

        // Create database schema
        let db_schema = format!("tenant_{}", tenant_id);
        sqlx::query(&format!("CREATE SCHEMA IF NOT EXISTS {}", db_schema))
            .execute(&self.db_pool)
            .await?;

        // Create Qdrant collection
        let collection_name = format!("tenant_{}_events", tenant_id);
        self.qdrant_client.create_collection(&collection_name).await?;

        // Set up S3 bucket with encryption
        if config.isolation_level == IsolationLevel::Dedicated {
            let bucket_name = format!("prowzi-tenant-{}", tenant_id);
            self.create_encrypted_bucket(&bucket_name, &config.data_residency).await?;
        }

        Ok(())
    }
}

// Secure API Gateway middleware
pub struct TenantAuthMiddleware {
    tenant_manager: Arc<TenantIsolationManager>,
    token_validator: Arc<TokenValidator>,
}

impl<S> Service<Request<Body>> for TenantAuthMiddleware<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let tenant_manager = self.tenant_manager.clone();
        let token_validator = self.token_validator.clone();
        let next = self.next.clone();

        Box::pin(async move {
            // Extract and validate token
            let token = extract_bearer_token(&req)?;
            let claims = token_validator.validate(&token).await?;

            // Verify tenant access
            let tenant_id = claims.tenant_id;
            let requested_tenant = extract_tenant_from_path(&req);

            if tenant_id != requested_tenant {
                let allowed = tenant_manager.validate_cross_tenant_access(
                    &tenant_id,
                    &requested_tenant,
                    req.method().as_str(),
                ).await?;

                if !allowed {
                    return Err(ErrorForbidden("Cross-tenant access denied"));
                }
            }

            // Add tenant context to request
            let mut req = req;
            req.extensions_mut().insert(TenantId(tenant_id));

            // Continue with request
            next.call(req).await
        })
    }
}
