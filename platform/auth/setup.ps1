# Prowzi Authentication Service Setup Script for Windows
# This script sets up the authentication service for development/production

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development"
)

Write-Host "🚀 Setting up Prowzi Authentication Service..." -ForegroundColor Green

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if required tools are installed
function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error "Rust/Cargo is not installed. Please install from https://rustup.rs/"
        exit 1
    }
    
    if (!(Get-Command psql -ErrorAction SilentlyContinue)) {
        Write-Warning "PostgreSQL client not found. Install PostgreSQL for database operations."
    }
    
    if (!(Get-Command redis-cli -ErrorAction SilentlyContinue)) {
        Write-Warning "Redis client not found. Install Redis for cache operations."
    }
    
    if (!(Get-Command openssl -ErrorAction SilentlyContinue)) {
        Write-Warning "OpenSSL not found. Install OpenSSL or use Git Bash for key generation."
    }
    
    Write-Status "Dependencies check completed."
}

# Generate RSA key pairs for JWT tokens
function New-RSAKeys {
    Write-Status "Generating RSA key pairs for JWT tokens..."
    
    $KeysDir = ".\keys"
    if (!(Test-Path $KeysDir)) {
        New-Item -ItemType Directory -Path $KeysDir | Out-Null
    }
    
    # Generate JWT keys
    if (!(Test-Path "$KeysDir\jwt_private.pem")) {
        if (Get-Command openssl -ErrorAction SilentlyContinue) {
            openssl genrsa -out "$KeysDir\jwt_private.pem" 2048
            openssl rsa -in "$KeysDir\jwt_private.pem" -pubout -out "$KeysDir\jwt_public.pem"
            Write-Status "JWT keys generated"
        } else {
            Write-Warning "OpenSSL not available. Please generate RSA keys manually:"
            Write-Warning "  openssl genrsa -out keys/jwt_private.pem 2048"
            Write-Warning "  openssl rsa -in keys/jwt_private.pem -pubout -out keys/jwt_public.pem"
            return $false
        }
    } else {
        Write-Warning "JWT keys already exist, skipping generation"
    }
    
    # Generate refresh token keys
    if (!(Test-Path "$KeysDir\refresh_private.pem")) {
        if (Get-Command openssl -ErrorAction SilentlyContinue) {
            openssl genrsa -out "$KeysDir\refresh_private.pem" 2048
            openssl rsa -in "$KeysDir\refresh_private.pem" -pubout -out "$KeysDir\refresh_public.pem"
            Write-Status "Refresh token keys generated"
        } else {
            Write-Warning "Please generate refresh token keys manually:"
            Write-Warning "  openssl genrsa -out keys/refresh_private.pem 2048"
            Write-Warning "  openssl rsa -in keys/refresh_private.pem -pubout -out keys/refresh_public.pem"
            return $false
        }
    } else {
        Write-Warning "Refresh token keys already exist, skipping generation"
    }
    
    return $true
}

# Create environment file
function New-EnvironmentFile {
    Write-Status "Creating environment configuration..."
    
    if (!(Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        
        # Try to populate with generated keys
        if ((Test-Path "keys\jwt_private.pem") -and (Test-Path "keys\jwt_public.pem")) {
            $jwtPrivate = (Get-Content "keys\jwt_private.pem" -Raw) -replace "`r`n", "\n" -replace "`n", "\n"
            $jwtPublic = (Get-Content "keys\jwt_public.pem" -Raw) -replace "`r`n", "\n" -replace "`n", "\n"
            $refreshPrivate = (Get-Content "keys\refresh_private.pem" -Raw) -replace "`r`n", "\n" -replace "`n", "\n"
            $refreshPublic = (Get-Content "keys\refresh_public.pem" -Raw) -replace "`r`n", "\n" -replace "`n", "\n"
            
            # Read .env file and replace placeholders
            $envContent = Get-Content ".env" -Raw
            $envContent = $envContent -replace "YOUR_JWT_PRIVATE_KEY_HERE", $jwtPrivate
            $envContent = $envContent -replace "YOUR_JWT_PUBLIC_KEY_HERE", $jwtPublic
            $envContent = $envContent -replace "YOUR_REFRESH_PRIVATE_KEY_HERE", $refreshPrivate
            $envContent = $envContent -replace "YOUR_REFRESH_PUBLIC_KEY_HERE", $refreshPublic
            
            Set-Content ".env" $envContent
        }
        
        Write-Status "Environment file created. Please review and update .env with your specific configuration."
        Write-Warning "Make sure to update DATABASE_URL and REDIS_URL in .env file!"
    } else {
        Write-Warning ".env file already exists, skipping creation"
    }
}

# Build the service
function Build-Service {
    Write-Status "Building authentication service..."
    
    cargo build --release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Build completed successfully"
        return $true
    } else {
        Write-Error "Build failed"
        return $false
    }
}

# Create Windows service installer (for production)
function New-WindowsService {
    if ($Environment -eq "production") {
        Write-Status "Creating Windows service installer..."
        
        $serviceScript = @"
# Windows Service Installation Script for Prowzi Auth
# Run as Administrator

`$serviceName = "ProwziAuth"
`$serviceDisplayName = "Prowzi Authentication Service"
`$servicePath = "`$PSScriptRoot\target\release\prowzi-auth.exe"
`$serviceDescription = "Prowzi platform authentication and authorization service"

# Stop and remove existing service if it exists
if (Get-Service -Name `$serviceName -ErrorAction SilentlyContinue) {
    Stop-Service -Name `$serviceName -Force
    Remove-Service -Name `$serviceName
    Write-Host "Removed existing service"
}

# Create new service
New-Service -Name `$serviceName ``
    -DisplayName `$serviceDisplayName ``
    -BinaryPathName `$servicePath ``
    -Description `$serviceDescription ``
    -StartupType Automatic

Write-Host "Service `$serviceName created successfully"
Write-Host "To start the service, run: Start-Service -Name `$serviceName"
"@
        
        Set-Content "install-service.ps1" $serviceScript
        Write-Status "Windows service installer created: install-service.ps1"
        Write-Warning "Run install-service.ps1 as Administrator to install the service"
    }
}

# Main setup function
function Start-Setup {
    Write-Status "Starting Prowzi Authentication Service Setup for $Environment environment"
    
    # Run setup steps
    Test-Dependencies
    
    $keysGenerated = New-RSAKeys
    New-EnvironmentFile
    
    $buildSuccess = Build-Service
    
    if ($Environment -eq "production") {
        New-WindowsService
        Write-Warning "Additional production setup required:"
        Write-Warning "1. Set up reverse proxy (IIS/nginx)"
        Write-Warning "2. Configure TLS certificates"
        Write-Warning "3. Set up monitoring and logging"
        Write-Warning "4. Configure Windows Firewall rules"
    }
    
    if ($buildSuccess) {
        Write-Status "Setup completed! 🎉"
        Write-Status ""
        Write-Status "Next steps:"
        Write-Status "1. Review and update .env file with your configuration"
        
        if (!$keysGenerated) {
            Write-Status "2. Generate RSA keys manually using OpenSSL"
            Write-Status "3. Update .env file with the generated keys"
        }
        
        Write-Status "2. Ensure PostgreSQL and Redis are running"
        Write-Status "3. Run database migrations manually if needed"
        Write-Status "4. Start the service:"
        Write-Status "   Development: cargo run"
        Write-Status "   Production: install-service.ps1 (as Administrator)"
        Write-Status ""
        Write-Status "Service will be available at: http://localhost:3001"
    } else {
        Write-Error "Setup failed during build step"
        exit 1
    }
}

# Run main setup
Start-Setup
