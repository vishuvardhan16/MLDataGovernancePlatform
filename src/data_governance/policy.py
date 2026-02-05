"""
Policy Enforcement Engine for Data Governance.

Implements automated compliance checking and policy enforcement
combining ML-based triggers with rule-based actions.

Reference: Section 3.4 of the paper - Policy Enforcement Engine
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from .config import PolicyConfig, PolicyAction, AnomalyType
from .anomaly import AnomalyEvent


logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance evaluation status."""
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"
    PENDING_REVIEW = "pending_review"


@dataclass
class PolicyCondition:
    """
    Represents a policy condition (predicate).
    
    P_anomaly: Anomaly or lineage-based trigger
    """
    
    condition_type: str  # 'anomaly', 'lineage', 'access', 'schema'
    predicate: Callable[[Dict], bool]
    description: str
    
    def evaluate(self, context: Dict) -> bool:
        """Evaluate the condition against context."""
        try:
            return self.predicate(context)
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False


@dataclass
class ComplianceRequirement:
    """
    Represents a compliance requirement.
    
    C_compliance: Regulatory or organizational compliance condition
    """
    
    requirement_id: str
    name: str
    regulation: str  # 'GDPR', 'HIPAA', 'SOX', 'internal'
    description: str
    check_function: Callable[[Dict], bool]
    severity: str = "high"  # 'low', 'medium', 'high', 'critical'
    
    def check(self, context: Dict) -> bool:
        """Check if requirement is satisfied."""
        try:
            return self.check_function(context)
        except Exception as e:
            logger.error(f"Error checking requirement {self.requirement_id}: {e}")
            return False


@dataclass
class GovernancePolicy:
    """
    Represents a governance policy as a triple (P, C, A).
    
    P: Predicate (anomaly/lineage trigger)
    C: Compliance condition
    A: Action to take upon violation
    """
    
    policy_id: str
    name: str
    description: str
    condition: PolicyCondition
    compliance: ComplianceRequirement
    action: PolicyAction
    priority: int = 5  # 1-10, higher = more important
    enabled: bool = True
    
    def evaluate(self, context: Dict) -> bool:
        """
        Evaluate policy activation.
        
        Policy activates when: P_anomaly(context) AND C_compliance(context)
        
        Args:
            context: Evaluation context dictionary
            
        Returns:
            True if policy should be activated
        """
        if not self.enabled:
            return False
        
        condition_met = self.condition.evaluate(context)
        compliance_violated = not self.compliance.check(context)
        
        return condition_met and compliance_violated


@dataclass
class PolicyViolation:
    """Represents a detected policy violation."""
    
    violation_id: str
    policy: GovernancePolicy
    timestamp: float
    context: Dict
    status: ComplianceStatus
    action_taken: PolicyAction
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class EnforcementAction:
    """Represents an enforcement action taken."""
    
    action_id: str
    action_type: PolicyAction
    violation_id: str
    timestamp: float
    success: bool
    details: Dict


class PolicyEnforcementEngine:
    """
    Engine for automated policy enforcement.
    
    Evaluates data operations against governance rules and compliance
    requirements, triggering appropriate actions upon violations.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize policy enforcement engine.
        
        Args:
            config: Policy configuration
        """
        self.config = config
        
        # Registered policies
        self.policies: Dict[str, GovernancePolicy] = {}
        
        # Violation history
        self.violations: List[PolicyViolation] = []
        
        # Action history
        self.actions: List[EnforcementAction] = []
        
        # Alert handlers
        self._alert_handlers: List[Callable[[PolicyViolation], None]] = []
        
        # Initialize default policies
        self._register_default_policies()
    
    def _register_default_policies(self) -> None:
        """Register default governance policies."""
        
        # Policy 1: PII Exposure Detection
        pii_condition = PolicyCondition(
            condition_type='anomaly',
            predicate=lambda ctx: ctx.get('anomaly_type') == AnomalyType.PII_EXPOSURE.value,
            description="Detects potential PII exposure in data pipelines"
        )
        
        pii_compliance = ComplianceRequirement(
            requirement_id='GDPR-001',
            name='PII Protection',
            regulation='GDPR',
            description='Personal data must be protected and not exposed',
            check_function=lambda ctx: not ctx.get('pii_detected', False),
            severity='critical'
        )
        
        self.register_policy(GovernancePolicy(
            policy_id='POL-001',
            name='PII Exposure Prevention',
            description='Blocks operations that may expose PII',
            condition=pii_condition,
            compliance=pii_compliance,
            action=PolicyAction.BLOCK,
            priority=10
        ))
        
        # Policy 2: Unauthorized Access Detection
        access_condition = PolicyCondition(
            condition_type='access',
            predicate=lambda ctx: ctx.get('anomaly_type') == AnomalyType.UNAUTHORIZED_ACCESS.value,
            description="Detects unauthorized data access attempts"
        )
        
        access_compliance = ComplianceRequirement(
            requirement_id='SOX-001',
            name='Access Control',
            regulation='SOX',
            description='Data access must be authorized',
            check_function=lambda ctx: ctx.get('user_authorized', True),
            severity='high'
        )
        
        self.register_policy(GovernancePolicy(
            policy_id='POL-002',
            name='Unauthorized Access Prevention',
            description='Alerts on unauthorized access attempts',
            condition=access_condition,
            compliance=access_compliance,
            action=PolicyAction.ALERT,
            priority=9
        ))
        
        # Policy 3: Data Retention Compliance
        retention_condition = PolicyCondition(
            condition_type='lineage',
            predicate=lambda ctx: ctx.get('anomaly_type') == AnomalyType.RETENTION_VIOLATION.value,
            description="Detects data retention policy violations"
        )
        
        retention_compliance = ComplianceRequirement(
            requirement_id='GDPR-002',
            name='Data Retention',
            regulation='GDPR',
            description='Data must not be retained beyond allowed period',
            check_function=lambda ctx: not ctx.get('retention_exceeded', False),
            severity='high'
        )
        
        self.register_policy(GovernancePolicy(
            policy_id='POL-003',
            name='Data Retention Enforcement',
            description='Enforces data retention policies',
            condition=retention_condition,
            compliance=retention_compliance,
            action=PolicyAction.QUARANTINE,
            priority=8
        ))
        
        # Policy 4: Schema Drift Monitoring
        schema_condition = PolicyCondition(
            condition_type='schema',
            predicate=lambda ctx: ctx.get('anomaly_type') == AnomalyType.SCHEMA_DRIFT.value,
            description="Detects unexpected schema changes"
        )
        
        schema_compliance = ComplianceRequirement(
            requirement_id='INT-001',
            name='Schema Stability',
            regulation='internal',
            description='Schema changes must be approved',
            check_function=lambda ctx: ctx.get('schema_change_approved', True),
            severity='medium'
        )
        
        self.register_policy(GovernancePolicy(
            policy_id='POL-004',
            name='Schema Change Control',
            description='Monitors and alerts on schema changes',
            condition=schema_condition,
            compliance=schema_compliance,
            action=PolicyAction.NOTIFY_STEWARD,
            priority=6
        ))
        
        # Policy 5: Incomplete Lineage Detection
        lineage_condition = PolicyCondition(
            condition_type='lineage',
            predicate=lambda ctx: ctx.get('lineage_completeness', 1.0) < self.config.lineage_completeness_threshold,
            description="Detects datasets with incomplete lineage"
        )
        
        lineage_compliance = ComplianceRequirement(
            requirement_id='INT-002',
            name='Lineage Documentation',
            regulation='internal',
            description='All datasets must have documented lineage',
            check_function=lambda ctx: ctx.get('lineage_completeness', 0) >= self.config.lineage_completeness_threshold,
            severity='medium'
        )
        
        self.register_policy(GovernancePolicy(
            policy_id='POL-005',
            name='Lineage Completeness',
            description='Ensures all datasets have complete lineage',
            condition=lineage_condition,
            compliance=lineage_compliance,
            action=PolicyAction.LOG,
            priority=5
        ))
    
    def register_policy(self, policy: GovernancePolicy) -> None:
        """Register a new governance policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered policy: {policy.name} ({policy.policy_id})")
    
    def unregister_policy(self, policy_id: str) -> bool:
        """Unregister a policy by ID."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Unregistered policy: {policy_id}")
            return True
        return False
    
    def add_alert_handler(
        self, 
        handler: Callable[[PolicyViolation], None]
    ) -> None:
        """Add a handler for policy violation alerts."""
        self._alert_handlers.append(handler)
    
    def evaluate_policies(
        self,
        context: Dict,
        anomaly_event: Optional[AnomalyEvent] = None
    ) -> List[PolicyViolation]:
        """
        Evaluate all policies against the given context.
        
        Args:
            context: Evaluation context
            anomaly_event: Optional anomaly event that triggered evaluation
            
        Returns:
            List of policy violations detected
        """
        violations = []
        
        # Enrich context with anomaly information
        if anomaly_event:
            context['anomaly_type'] = anomaly_event.anomaly_type.value
            context['anomaly_score'] = anomaly_event.anomaly_score
            context['reconstruction_error'] = anomaly_event.reconstruction_error
        
        # Sort policies by priority (highest first)
        sorted_policies = sorted(
            self.policies.values(),
            key=lambda p: p.priority,
            reverse=True
        )
        
        for policy in sorted_policies:
            if policy.evaluate(context):
                # Check confidence threshold
                anomaly_score = context.get('anomaly_score', 1.0)
                
                if anomaly_score >= self.config.min_confidence_for_action:
                    violation = PolicyViolation(
                        violation_id=f"VIO-{int(time.time() * 1000)}",
                        policy=policy,
                        timestamp=time.time(),
                        context=context.copy(),
                        status=ComplianceStatus.VIOLATION,
                        action_taken=policy.action
                    )
                    
                    violations.append(violation)
                    self.violations.append(violation)
                    
                    # Execute enforcement action
                    self._execute_action(violation)
                    
                    logger.warning(
                        f"Policy violation: {policy.name} "
                        f"(priority={policy.priority}, action={policy.action.value})"
                    )
        
        return violations
    
    def _execute_action(self, violation: PolicyViolation) -> EnforcementAction:
        """
        Execute enforcement action for a violation.
        
        Args:
            violation: Policy violation
            
        Returns:
            Enforcement action record
        """
        action_type = violation.action_taken
        
        action = EnforcementAction(
            action_id=f"ACT-{int(time.time() * 1000)}",
            action_type=action_type,
            violation_id=violation.violation_id,
            timestamp=time.time(),
            success=True,
            details={}
        )
        
        try:
            if action_type == PolicyAction.ALERT:
                self._send_alert(violation)
                action.details['alert_sent'] = True
                
            elif action_type == PolicyAction.LOG:
                self._log_violation(violation)
                action.details['logged'] = True
                
            elif action_type == PolicyAction.BLOCK:
                action.details['operation_blocked'] = True
                logger.critical(f"BLOCKED: {violation.policy.name}")
                
            elif action_type == PolicyAction.QUARANTINE:
                action.details['data_quarantined'] = True
                logger.warning(f"QUARANTINED: Data flagged for review")
                
            elif action_type == PolicyAction.NOTIFY_STEWARD:
                self._notify_steward(violation)
                action.details['steward_notified'] = True
                
        except Exception as e:
            action.success = False
            action.details['error'] = str(e)
            logger.error(f"Failed to execute action: {e}")
        
        self.actions.append(action)
        return action
    
    def _send_alert(self, violation: PolicyViolation) -> None:
        """Send alert for policy violation."""
        for handler in self._alert_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(
            f"ALERT: Policy '{violation.policy.name}' violated. "
            f"Severity: {violation.policy.compliance.severity}"
        )
    
    def _log_violation(self, violation: PolicyViolation) -> None:
        """Log policy violation for audit."""
        logger.info(
            f"AUDIT LOG: Violation {violation.violation_id} | "
            f"Policy: {violation.policy.policy_id} | "
            f"Time: {datetime.fromtimestamp(violation.timestamp).isoformat()} | "
            f"Status: {violation.status.value}"
        )
    
    def _notify_steward(self, violation: PolicyViolation) -> None:
        """Notify data steward of violation."""
        logger.info(
            f"STEWARD NOTIFICATION: {violation.policy.name} requires review. "
            f"Violation ID: {violation.violation_id}"
        )
    
    def resolve_violation(
        self,
        violation_id: str,
        resolution_notes: str
    ) -> bool:
        """
        Mark a violation as resolved.
        
        Args:
            violation_id: Violation identifier
            resolution_notes: Notes about resolution
            
        Returns:
            True if violation was found and resolved
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                violation.resolution_notes = resolution_notes
                violation.status = ComplianceStatus.COMPLIANT
                
                logger.info(f"Violation {violation_id} resolved: {resolution_notes}")
                return True
        
        return False
    
    def get_compliance_report(self) -> Dict:
        """
        Generate compliance report.
        
        Returns:
            Dictionary with compliance statistics
        """
        total_violations = len(self.violations)
        resolved = sum(1 for v in self.violations if v.resolved)
        unresolved = total_violations - resolved
        
        # Group by policy
        by_policy = {}
        for v in self.violations:
            pid = v.policy.policy_id
            by_policy[pid] = by_policy.get(pid, 0) + 1
        
        # Group by severity
        by_severity = {}
        for v in self.violations:
            sev = v.policy.compliance.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        # Group by action
        by_action = {}
        for v in self.violations:
            act = v.action_taken.value
            by_action[act] = by_action.get(act, 0) + 1
        
        # Calculate false positive rate (violations resolved as false positives)
        false_positives = sum(
            1 for v in self.violations 
            if v.resolved and 'false positive' in v.resolution_notes.lower()
        )
        fp_rate = false_positives / total_violations if total_violations > 0 else 0
        
        return {
            'total_violations': total_violations,
            'resolved': resolved,
            'unresolved': unresolved,
            'false_positive_rate': fp_rate,
            'by_policy': by_policy,
            'by_severity': by_severity,
            'by_action': by_action,
            'total_actions': len(self.actions),
            'successful_actions': sum(1 for a in self.actions if a.success)
        }
