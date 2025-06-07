# HashiCorp Vault configuration for Prowzi secrets management
# Manages API keys, database credentials, and other sensitive data

# Vault cluster configuration
resource "aws_kms_key" "vault" {
  description             = "Prowzi Vault encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name        = "prowzi-vault-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
    Purpose     = "vault-encryption"
  }
}

resource "aws_kms_alias" "vault" {
  name          = "alias/prowzi-vault-${var.environment}"
  target_key_id = aws_kms_key.vault.key_id
}

# S3 bucket for Vault storage backend
resource "aws_s3_bucket" "vault_storage" {
  bucket = "prowzi-vault-storage-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "prowzi-vault-storage-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
    Purpose     = "vault-backend-storage"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}

resource "aws_s3_bucket_versioning" "vault_storage" {
  bucket = aws_s3_bucket.vault_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "vault_storage" {
  bucket = aws_s3_bucket.vault_storage.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.vault.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "vault_storage" {
  bucket = aws_s3_bucket.vault_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role for Vault instances
resource "aws_iam_role" "vault_instance" {
  name = "prowzi-vault-instance-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "prowzi-vault-instance-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
  }
}

resource "aws_iam_role_policy" "vault_s3_access" {
  name = "vault-s3-access"
  role = aws_iam_role.vault_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.vault_storage.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.vault_storage.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.vault.arn
      }
    ]
  })
}

resource "aws_iam_instance_profile" "vault" {
  name = "prowzi-vault-${var.environment}"
  role = aws_iam_role.vault_instance.name
}

# Security group for Vault cluster
resource "aws_security_group" "vault" {
  name_prefix = "prowzi-vault-${var.environment}"
  vpc_id      = var.vpc_id

  # Vault API port
  ingress {
    from_port       = 8200
    to_port         = 8200
    protocol        = "tcp"
    security_groups = [aws_security_group.vault_alb.id]
  }

  # Vault cluster communication
  ingress {
    from_port = 8201
    to_port   = 8201
    protocol  = "tcp"
    self      = true
  }

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr_blocks]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "prowzi-vault-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
  }
}

# Application Load Balancer for Vault
resource "aws_security_group" "vault_alb" {
  name_prefix = "prowzi-vault-alb-${var.environment}"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "prowzi-vault-alb-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
  }
}

resource "aws_lb" "vault" {
  name               = "prowzi-vault-${var.environment}"
  internal           = true
  load_balancer_type = "application"
  security_groups    = [aws_security_group.vault_alb.id]
  subnets            = var.private_subnet_ids

  enable_deletion_protection = var.environment == "production"

  tags = {
    Name        = "prowzi-vault-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
  }
}

resource "aws_lb_target_group" "vault" {
  name     = "prowzi-vault-${var.environment}"
  port     = 8200
  protocol = "HTTPS"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/v1/sys/health"
    port                = "traffic-port"
    protocol            = "HTTPS"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name        = "prowzi-vault-${var.environment}"
    Environment = var.environment
    Project     = "prowzi"
  }
}

resource "aws_lb_listener" "vault" {
  load_balancer_arn = aws_lb.vault.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.vault.arn
  }
}

# Launch template for Vault instances
resource "aws_launch_template" "vault" {
  name_prefix   = "prowzi-vault-${var.environment}"
  image_id      = data.aws_ami.vault.id
  instance_type = var.vault_instance_type
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.vault.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.vault.name
  }

  user_data = base64encode(templatefile("${path.module}/scripts/vault-init.sh", {
    s3_bucket    = aws_s3_bucket.vault_storage.id
    kms_key_id   = aws_kms_key.vault.key_id
    environment  = var.environment
    vault_domain = "vault.${var.domain_name}"
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "prowzi-vault-${var.environment}"
      Environment = var.environment
      Project     = "prowzi"
      Role        = "vault"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group for Vault cluster
resource "aws_autoscaling_group" "vault" {
  name                = "prowzi-vault-${var.environment}"
  vpc_zone_identifier = var.private_subnet_ids
  target_group_arns   = [aws_lb_target_group.vault.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = var.vault_min_size
  max_size         = var.vault_max_size
  desired_capacity = var.vault_desired_capacity

  launch_template {
    id      = aws_launch_template.vault.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "prowzi-vault-${var.environment}"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = "prowzi"
    propagate_at_launch = true
  }

  tag {
    key                 = "Role"
    value               = "vault"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Data source for Vault AMI
data "aws_ami" "vault" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Route53 record for Vault load balancer
resource "aws_route53_record" "vault" {
  zone_id = var.route53_zone_id
  name    = "vault.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.vault.dns_name
    zone_id                = aws_lb.vault.zone_id
    evaluate_target_health = true
  }
}

# Outputs
output "vault_endpoint" {
  description = "Vault cluster endpoint"
  value       = "https://vault.${var.domain_name}"
}

output "vault_kms_key_id" {
  description = "KMS key ID used for Vault encryption"
  value       = aws_kms_key.vault.key_id
}

output "vault_s3_bucket" {
  description = "S3 bucket used for Vault backend storage"
  value       = aws_s3_bucket.vault_storage.id
}

output "vault_load_balancer_dns" {
  description = "DNS name of the Vault load balancer"
  value       = aws_lb.vault.dns_name
}