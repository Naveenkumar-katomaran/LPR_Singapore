import boto3
from botocore.client import Config
import logging as log
from urllib.parse import urljoin
logging = log.getLogger(__name__)
log.getLogger("boto3").setLevel(log.WARNING)
log.getLogger("botocore").setLevel(log.WARNING)
log.getLogger("urllib3").setLevel(log.WARNING)

class MinioClient:
    def __init__(self, config):
        self.endpoint = config.get("host", "127.0.0.1:9000").replace("http://", "").replace("https://", "")
        self.access_key = config.get("username", "minioadmin")
        self.secret_key = config.get("password", "minioadmin")
        self.bucket = config.get("bucket", "traffic-api")
        self.secure = config.get("secure", False)
        self.expire_seconds = config.get("expire_seconds", 3600 * 24 * 7) # Default 7 days
        self.region = config.get("region", "us-east-1")
        self.public_url = config.get("public_url", "").rstrip("/")
        
        protocol = "https" if self.secure else "http"
        self.endpoint_url = f"{protocol}://{self.endpoint}"
        
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 1},
                connect_timeout=3,
                read_timeout=3
            ),
        )
        try:
            self.ensure_bucket_exists()
        except Exception:
            pass # Exception is handled and logged inside ensure_bucket_exists

    def ensure_bucket_exists(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.client.create_bucket(Bucket=self.bucket)
                logging.info(f"Created MinIO bucket: {self.bucket}")
            else:
                logging.error(f"Error checking bucket: {e}")
                raise
        except Exception as e:
            logging.error(f"MinIO network/timeout error: {e}")
            raise

    def upload_bytes(self, data, object_key, content_type="image/jpeg"):
        """Upload bytes directly from memory."""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=data,
                ContentType=content_type,
            )
            logging.info(f"Uploaded (bytes) to MinIO -> {self.bucket}/{object_key}")
            return object_key
        except Exception as exc:
            logging.error(f"MinIO byte-upload failed: {self.bucket}/{object_key} - {exc}")
            return None

    def get_public_url(self, object_key):
        """Returns direct public HTTP URL without signatures."""
        base = self.public_url if self.public_url else self.endpoint_url
        return f"{base}/{self.bucket}/{object_key}"

    def get_url(self, object_key, presign=True):
        """Returns HTTP URL (Presigned by default)."""
        if presign:
            try:
                url = self.client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": object_key},
                    ExpiresIn=self.expire_seconds,
                )
                return url
            except Exception as e:
                logging.warning(f"Presigned URL failed for {object_key}: {e}")
        
        return self.get_public_url(object_key)
