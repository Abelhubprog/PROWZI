hcl
resource "aws_s3_bucket" "wal_archive" {
  bucket = "prowzi-wal-archive-${var.environment}"

  lifecycle_rule {
    enabled = true

    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 90
    }
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_iam_role" "wal_archiver" {
  name = "prowzi-wal-archiver-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "wal_archiver" {
  role = aws_iam_role.wal_archiver.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.wal_archive.arn,
          "${aws_s3_bucket.wal_archive.arn}/*"
        ]
      }
    ]
  })
}
