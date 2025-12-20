"""
RQ Client for enqueuing jobs to the Redis queue.
"""
from redis import Redis
from rq import Queue
import uuid


class RQClient:
    """Client for managing RQ job queue."""
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        """
        Initialize the RQ client.
        
        Args:
            host: Redis/Valkey host
            port: Redis/Valkey port
        """
        self.redis_conn = Redis(host=host, port=port)
        self.queue = Queue("rag_queries", connection=self.redis_conn)
    
    def enqueue_query(self, query: str) -> str:
        """
        Enqueue a query for processing.
        
        Args:
            query: The user's question/message
            
        Returns:
            str: The job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        # Import here to avoid circular imports
        from queues.worker import process_query
        
        job = self.queue.enqueue(
            process_query,
            job_id,
            query,
            job_id=job_id
        )
        
        return job_id
    
    def get_job_status(self, job_id: str) -> dict:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            dict: Job status information
        """
        from rq.job import Job
        
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            return {
                "job_id": job_id,
                "status": job.get_status(),
                "result": job.result if job.is_finished else None
            }
        except Exception:
            return {
                "job_id": job_id,
                "status": "not_found",
                "result": None
            }


# Singleton instance
rq_client = RQClient()