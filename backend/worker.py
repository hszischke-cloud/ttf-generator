"""
worker.py — Runs a single processing job in isolation.
Called as: python worker.py <job_id>
"""
import sys
import traceback

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: worker.py <job_id>", file=sys.stderr)
        sys.exit(1)

    job_id = sys.argv[1]

    try:
        from main import _process_job
        _process_job(job_id)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            from job_store import job_store
            job_store.update_state(job_id, status="error", error_message=tb[-2000:])
        except Exception as e2:
            print(f"Failed to update job state: {e2}", file=sys.stderr, flush=True)
        sys.exit(1)
