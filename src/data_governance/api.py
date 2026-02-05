"""
REST API for Data Governance Framework.

Provides HTTP endpoints for integrating the governance framework
with enterprise data platforms and monitoring systems.

Reference: Section 4.3 - Integration Architecture
"""

import logging
from typing import Dict, Optional
from dataclasses import asdict
import json
import time

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.info("FastAPI not installed. REST API features disabled.")


if HAS_FASTAPI:
    
    class EventRequest(BaseModel):
        """Request model for processing events."""
        event_type: str
        source_system: str
        metadata: Dict
        event_id: Optional[str] = None
    
    class DatasetRequest(BaseModel):
        """Request model for adding datasets."""
        dataset_id: str
        name: str
        metadata: Dict
    
    class TransformationRequest(BaseModel):
        """Request model for adding transformations."""
        transform_id: str
        name: str
        input_datasets: list
        output_datasets: list
        metadata: Dict
    
    class PolicyRequest(BaseModel):
        """Request model for policy operations."""
        policy_id: str
        enabled: Optional[bool] = None


def create_api(framework) -> 'FastAPI':
    """
    Create FastAPI application for governance framework.
    
    Args:
        framework: GovernanceAutomationFramework instance
        
    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Data Governance Automation API",
        description="ML-based framework for lineage tracking, anomaly detection, and policy enforcement",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Check API health status."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "initialized": framework._initialized
        }
    
    # Event processing
    @app.post("/events")
    async def process_event(request: EventRequest):
        """
        Process a governance event.
        
        Analyzes the event for anomalies and policy violations.
        """
        try:
            result = framework.process_event(
                event_type=request.event_type,
                source_system=request.source_system,
                metadata=request.metadata,
                event_id=request.event_id
            )
            
            return {
                "event_id": result.event_id,
                "anomaly_detected": result.anomaly_detected,
                "violations": result.violations,
                "lineage_updated": result.lineage_updated
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/events/batch")
    async def process_batch(events: list):
        """Process multiple events in batch."""
        try:
            results = framework.process_batch(events)
            return {
                "processed": len(results),
                "anomalies": sum(1 for r in results if r.anomaly_detected),
                "violations": sum(len(r.violations) for r in results)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Lineage endpoints
    @app.post("/lineage/datasets")
    async def add_dataset(request: DatasetRequest):
        """Add a dataset to the lineage graph."""
        try:
            framework.lineage_engine.add_dataset(
                dataset_id=request.dataset_id,
                name=request.name,
                metadata=request.metadata
            )
            return {"status": "success", "dataset_id": request.dataset_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/lineage/transformations")
    async def add_transformation(request: TransformationRequest):
        """Add a transformation to the lineage graph."""
        try:
            framework.lineage_engine.add_transformation(
                transform_id=request.transform_id,
                name=request.name,
                input_datasets=request.input_datasets,
                output_datasets=request.output_datasets,
                metadata=request.metadata
            )
            return {"status": "success", "transform_id": request.transform_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/lineage/{dataset_id}")
    async def get_lineage(dataset_id: str):
        """Get full lineage for a dataset."""
        try:
            lineage = framework.get_dataset_lineage(dataset_id)
            return lineage
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/lineage/infer")
    async def infer_lineage(background_tasks: BackgroundTasks):
        """Trigger lineage inference (runs in background)."""
        background_tasks.add_task(framework.infer_lineage)
        return {"status": "inference_started"}
    
    # Anomaly endpoints
    @app.get("/anomalies")
    async def get_anomalies(limit: int = 100):
        """Get recent anomaly events."""
        anomalies = framework.anomaly_engine.anomaly_history[-limit:]
        return {
            "count": len(anomalies),
            "anomalies": [
                {
                    "event_id": a.event_id,
                    "timestamp": a.timestamp,
                    "type": a.anomaly_type.value,
                    "score": a.anomaly_score,
                    "is_critical": a.is_critical
                }
                for a in anomalies
            ]
        }
    
    @app.get("/anomalies/statistics")
    async def get_anomaly_stats():
        """Get anomaly detection statistics."""
        return framework.anomaly_engine.get_anomaly_statistics()
    
    # Policy endpoints
    @app.get("/policies")
    async def list_policies():
        """List all registered policies."""
        return {
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "name": p.name,
                    "description": p.description,
                    "action": p.action.value,
                    "priority": p.priority,
                    "enabled": p.enabled
                }
                for p in framework.policy_engine.policies.values()
            ]
        }
    
    @app.put("/policies/{policy_id}")
    async def update_policy(policy_id: str, request: PolicyRequest):
        """Enable or disable a policy."""
        if policy_id not in framework.policy_engine.policies:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        if request.enabled is not None:
            framework.policy_engine.policies[policy_id].enabled = request.enabled
        
        return {"status": "updated", "policy_id": policy_id}
    
    @app.get("/violations")
    async def get_violations(limit: int = 100, unresolved_only: bool = False):
        """Get policy violations."""
        violations = framework.policy_engine.violations
        
        if unresolved_only:
            violations = [v for v in violations if not v.resolved]
        
        violations = violations[-limit:]
        
        return {
            "count": len(violations),
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "policy_id": v.policy.policy_id,
                    "policy_name": v.policy.name,
                    "timestamp": v.timestamp,
                    "status": v.status.value,
                    "action_taken": v.action_taken.value,
                    "resolved": v.resolved
                }
                for v in violations
            ]
        }
    
    @app.post("/violations/{violation_id}/resolve")
    async def resolve_violation(violation_id: str, notes: str = ""):
        """Mark a violation as resolved."""
        success = framework.policy_engine.resolve_violation(violation_id, notes)
        if not success:
            raise HTTPException(status_code=404, detail="Violation not found")
        return {"status": "resolved", "violation_id": violation_id}
    
    # Reports
    @app.get("/reports/governance")
    async def get_governance_report():
        """Get comprehensive governance report."""
        return framework.get_governance_report()
    
    @app.get("/reports/compliance")
    async def get_compliance_report():
        """Get compliance report."""
        return framework.policy_engine.get_compliance_report()
    
    return app


def run_server(framework, host: str = "0.0.0.0", port: int = 8000):
    """
    Run the API server.
    
    Args:
        framework: GovernanceAutomationFramework instance
        host: Server host
        port: Server port
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required. Install with: pip install uvicorn")
    
    app = create_api(framework)
    uvicorn.run(app, host=host, port=port)
